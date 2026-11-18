#!/usr/bin/env python3
"""d3-flash zero-shot 评测 jev-mixed 验证集（2600 条）。

与 eval_d3_mini_zero_shot.py 完全同口径，仅换模型路径为 d3-flash（Qwen3.5-9B
backbone + 255 路 code-readout，readout [255,4096]）。

对齐口径：
  - 输入 state = jev 的中文分类规则全文（jev-mixed.json schema.systems）
  - 输入 images = 曲线图绝对路径（schema.data_roots + state.images[0]）
  - question = choice 类型，选项名做 key、描述留 None
  - 预测 = d3 system_one 返回 probabilities 的 argmax
  - label = rec["answer"]（选项下标）

注意：这是「换 prompt 框架」的 zero-shot，不是复现 jev 训练口径。

用法：
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python semantic_router/eval_d3_flash_zero_shot.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# d3-flash runtime 所在目录（d3_runtime.py / d3_format.py / d3_fast.py 等）
D3_DIR = os.environ.get("D3_DIR", "checkpoints/d3-flash")
# 项目根（jev-mixed.json 相对路径的基准）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIXED_FILE = os.path.join(ROOT, "datas/aiops/noise/jev-mixed.json")
DEFAULT_OUT = os.path.join(ROOT, "semantic_router", "results", "d3_flash_zero_shot_eval.json")

sys.path.insert(0, D3_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="d3-flash zero-shot eval on jev-mixed eval split")
    p.add_argument("--d3-dir", default=D3_DIR, help="d3-flash checkpoint/runtime directory")
    p.add_argument("--device", default="cuda:0", help="device for the model (default cuda:0)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="questions per forward pass (default 8)")
    p.add_argument("--mixed-file", default=MIXED_FILE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None,
                   help="只评前 N 条（调试用，默认全部 2600）")
    p.add_argument("--verify", default="none", choices=["none", "fast", "full"],
                   help="d3 manifest 校验强度（none 最快）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("加载 d3-flash ...", flush=True)
    from d3_runtime import D3  # noqa: E402
    d3 = D3.from_pretrained(
        args.d3_dir, device=args.device, verify=args.verify, batch_size=args.batch_size
    )
    print(f"加载完成: model={d3.model_name} backbone={d3.backbone_type} "
          f"attention={d3.attention_mode} kernels={d3.kernels}", flush=True)

    d = json.load(open(args.mixed_file, encoding="utf-8"))
    schema = d["schema"]
    eval_recs = d["splits"]["eval"]
    if args.limit:
        eval_recs = eval_recs[: args.limit]
    print(f"eval 记录数: {len(eval_recs)}", flush=True)

    task_cache = {
        tid: {
            "system": schema["systems"][t["system_ref"]],
            "data_root": schema["data_roots"][t["data_root_ref"]],
            "choices": t["choices"],
            "question": t["question"],
        }
        for tid, t in schema["tasks"].items()
    }

    results = defaultdict(
        lambda: {"correct": 0, "total": 0,
                 "per_class": defaultdict(lambda: {"correct": 0, "total": 0})}
    )

    t0 = time.time()
    for i, rec in enumerate(eval_recs):
        tid = rec["task_id"]
        t = task_cache[tid]
        img_path = os.path.join(t["data_root"], rec["state"]["images"][0])
        label = rec["answer"]
        choices = t["choices"]

        question = {
            "type": "choice",
            "instructions": t["question"],
            "criteria": {c: None for c in choices},
        }
        try:
            result = d3.system_one(state=t["system"], questions={"q": question},
                                   images=[img_path])
            pred_key = result["answers"]["q"]["choice"]
            pred = choices.index(pred_key)
        except Exception as exc:  # noqa: BLE001
            results[tid]["total"] += 1
            results[tid]["per_class"][choices[label]]["total"] += 1
            if i < 5:
                print(f"  [warn] 样本 {i} ({tid}) 调用失败: {exc}", flush=True)
            continue

        results[tid]["total"] += 1
        results[tid]["per_class"][choices[label]]["total"] += 1
        if pred == label:
            results[tid]["correct"] += 1
            results[tid]["per_class"][choices[label]]["correct"] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(eval_recs) - i - 1) / rate
            print(f"  {i + 1}/{len(eval_recs)}  速率 {rate:.2f}/s  ETA {eta / 60:.1f}min",
                  flush=True)

    per_task = {}
    total_correct = total_n = 0
    for tid in sorted(results):
        r = results[tid]
        acc = r["correct"] / r["total"] if r["total"] else 0.0
        total_correct += r["correct"]
        total_n += r["total"]
        per_task[tid] = {
            "acc": acc,
            "correct": r["correct"],
            "total": r["total"],
            "per_class": {
                label: {
                    "correct": pc["correct"],
                    "total": pc["total"],
                    "acc": pc["correct"] / pc["total"] if pc["total"] else 0.0,
                }
                for label, pc in sorted(r["per_class"].items())
            },
        }

    out = {
        "model": "d3-flash",
        "mode": "zero-shot (no argos training)",
        "overall": {
            "acc": total_correct / total_n if total_n else 0.0,
            "correct": total_correct,
            "total": total_n,
        },
        "per_task": per_task,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n========== d3-flash zero-shot 结果 ==========", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n结果已存: {args.out}")


if __name__ == "__main__":
    main()
