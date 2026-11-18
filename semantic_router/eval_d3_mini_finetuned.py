#!/usr/bin/env python3
"""评测微调后的 d3-mini（LoRA + readout）在 jev-mixed 验证集上的表现。

与 zero-shot 评测同口径（state=中文规则 + images=曲线图 + choice 题），
但模型用微调后的 checkpoint（LoRA adapter + readout.safetensors），
前向走 noncausal 路径（与训练一致）。

用法：
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python semantic_router/eval_d3_mini_finetuned.py \
      --ckpt outputs/d3_mini_finetuned
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

D3_DIR_DEFAULT = os.environ.get("D3_DIR", "checkpoints/d3-mini")
sys.path.insert(0, D3_DIR_DEFAULT)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="微调 checkpoint 目录")
    p.add_argument("--d3-dir", default=D3_DIR_DEFAULT)
    p.add_argument("--mixed-file",
                   default="datas/aiops/noise/jev-mixed.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.d3_dir)  # d3 runtime 目录按 --d3-dir 动态加入（flash/mini 共用）
    from d3_runtime import D3, PromptEncoder
    from safetensors.torch import load_file
    from peft import PeftModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    # 1. 加载 backbone + 挂微调后的 LoRA
    backbone = Qwen3_5Model.from_pretrained(
        args.d3_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=args.device)
    backbone.language_model = PeftModel.from_pretrained(
        backbone.language_model, args.ckpt)
    backbone.eval()

    # 2. 加载微调后的 readout
    readout = load_file(os.path.join(args.ckpt, "readout.safetensors"))["weight"]
    readout = readout.to(args.device).float()

    # 3. 加载 d3 runtime（用于 processor + prepare）
    d3 = D3.from_pretrained(args.d3_dir, device=args.device, verify="none",
                            batch_size=8)

    encoder = PromptEncoder(backbone.language_model)

    # 4. 数据集
    d = json.load(open(args.mixed_file, encoding="utf-8"))
    schema = d["schema"]
    recs = d["splits"]["eval"]
    if args.limit:
        recs = recs[: args.limit]

    results = defaultdict(lambda: {"correct": 0, "total": 0,
                                   "per_class": defaultdict(lambda: {"c": 0, "t": 0})})

    t0 = time.time()
    for i, rec in enumerate(recs):
        tid = rec["task_id"]
        task = schema["tasks"][tid]
        system = schema["systems"][task["system_ref"]]
        data_root = schema["data_roots"][task["data_root_ref"]]
        img_path = os.path.join(data_root, rec["state"]["images"][0])
        choices = task["choices"]
        question = {"type": "choice", "instructions": task["question"],
                    "criteria": {c: None for c in choices}}

        prepared = d3.prepare(state=system, questions={"q": question},
                              images=[img_path])
        if "q" not in prepared.texts or not prepared.images:
            continue

        images = prepared.images
        texts = prepared.texts["q"]
        encoded = d3.processor(text=texts, images=images, padding=True,
                               return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in encoded.items()}

        with torch.no_grad():
            embeds = backbone.get_input_embeddings()(inputs["input_ids"])
            pooled = backbone.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"],
                return_dict=True).pooler_output
            pooled = torch.cat(pooled, dim=0).to(embeds.device, embeds.dtype)
            slots, _ = backbone.get_placeholder_mask(
                inputs["input_ids"], inputs_embeds=embeds, image_features=pooled)
            embeds = embeds.masked_scatter(slots, pooled)
            mrope = backbone.compute_3d_position_ids(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                video_grid_thw=None, inputs_embeds=embeds,
                attention_mask=inputs["attention_mask"],
                past_key_values=None,
                mm_token_type_ids=inputs["mm_token_type_ids"])
            hidden = encoder(embeds, inputs["attention_mask"], mrope)
            logits = hidden[:, -1].float() @ readout.T

        logits = logits[0, : len(choices)]
        pred = int(logits.argmax().item())
        label = rec["answer"]

        results[tid]["total"] += 1
        results[tid]["per_class"][choices[label]]["t"] += 1
        if pred == label:
            results[tid]["correct"] += 1
            results[tid]["per_class"][choices[label]]["c"] += 1

        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(recs) - i - 1) / rate
            print(f"  {i+1}/{len(recs)}  速率 {rate:.2f}/s  ETA {eta/60:.1f}min",
                  flush=True)

    per_task = {}
    tc = tn = 0
    for tid in sorted(results):
        r = results[tid]
        acc = r["correct"] / r["total"] if r["total"] else 0.0
        tc += r["correct"]; tn += r["total"]
        per_task[tid] = {
            "acc": acc, "correct": r["correct"], "total": r["total"],
            "per_class": {lab: {"correct": pc["c"], "total": pc["t"],
                                "acc": pc["c"] / pc["t"] if pc["t"] else 0.0}
                          for lab, pc in sorted(r["per_class"].items())},
        }
    out = {"model": "d3-mini-finetuned", "overall": {"acc": tc / tn if tn else 0.0,
                                                      "correct": tc, "total": tn},
           "per_task": per_task}

    if args.out is None:
        args.out = os.path.join(_REPO_ROOT, "semantic_router", "results",
                                "d3_mini_finetuned_eval.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print("\n========== d3-mini 微调后结果 ==========", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已存: {args.out}")


if __name__ == "__main__":
    main()
