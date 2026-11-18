#!/usr/bin/env python3
"""评测「基座未微调」的 jev_kev 决策模型在 jev-mixed 验证集上的效果。

口径：与 jev_4b_mixed 训练完全一致（Qwen3.5-4B 多模态基座 + LoRA r=16 +
PointerHead dp=256），但**不加载任何训练 checkpoint**。

关键点（为何这能代表「未微调」）：
  - LoRA 初始化 delta=0 → backbone 前向输出 = 原始 Qwen3.5-4B 基座。
  - PointerHead 是 nn.Linear 默认 Kaiming uniform 随机初始化 → 随机打分头。
  所以本脚本评测的 = 冻结基座特征 + 随机分类头，即「微调前」的真实基线。

用法：
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python semantic_router/eval_jev_base.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_kev.source.data import load_mixed_records
from jev_kev.source.model import build_model
from transformers import AutoImageProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=os.environ.get("BACKBONE", "checkpoints/Qwen3.5-4B"))
    ap.add_argument("--mixed-file", default="datas/aiops/noise/jev-mixed.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-state", type=int, default=3072)
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="semantic_router/results/jev_4b_base_eval.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok = __import__("jev_kev.source.model", fromlist=["load_tokenizer"]).load_tokenizer(args.backbone)
    image_processor = AutoImageProcessor.from_pretrained(args.backbone, local_files_only=True)

    print("加载基座 + 随机 PointerHead（LoRA delta=0）...", flush=True)
    # build_model 内部默认冻结 vision tower（只挂 LoRA + PointerHead）
    model = build_model(args.backbone, args.device, lora=16, head_dim=256,
                        multimodal=True, image_processor=image_processor)
    model.eval()

    recs = load_mixed_records(args.mixed_file, "eval", args.limit)
    print(f"eval 记录数: {len(recs)}", flush=True)

    # 按 task 聚合
    per_task = {}
    total_correct = total_n = 0
    t0 = time.time()
    with torch.no_grad():
        for i, rec in enumerate(recs):
            enc = model.encode(tok, rec, max_state=args.max_state,
                               max_branch=2560, strict=True,
                               max_pixels=args.max_pixels)
            logits_b = model.forward_batch([enc], shared_prefix=False)
            z = logits_b[0][0]
            q = rec["questions"][0]
            t = torch.tensor(q["target"], device=args.device, dtype=torch.float32)
            p = F.log_softmax(z.float(), -1).exp()
            pred = int(p.argmax().item())
            gold = int(t.argmax().item())
            tid = q["qid"]
            per_task.setdefault(tid, {"correct": 0, "total": 0})
            per_task[tid]["total"] += 1
            per_task[tid]["correct"] += int(pred == gold)
            total_correct += int(pred == gold)
            total_n += 1
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(recs) - i - 1) / rate
                print(f"  {i+1}/{len(recs)}  acc_sofar={total_correct/total_n:.4f}  "
                      f"速率 {rate:.2f}/s ETA {eta/60:.1f}min", flush=True)

    out = {
        "model": "jev_4b_base (Qwen3.5-4B + random PointerHead, 未微调)",
        "overall": {"acc": total_correct / total_n, "correct": total_correct, "total": total_n},
        "per_task": {tid: {"acc": v["correct"] / v["total"], "correct": v["correct"],
                           "total": v["total"]} for tid, v in sorted(per_task.items())},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n========== 基座未微调结果 ==========", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已存: {args.out}")


if __name__ == "__main__":
    main()
