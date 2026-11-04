"""jev_kev 评测：在 typed-decisions 上评测，对齐 jev train.py 的指标口径。

指标（与 jev text_jev_train/scripts/train.py metrics() 一致）：
  accuracy / soft_cross_entropy / brier_sum / gold_mass_at_selected_option / score_expectation_mae / ECE
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_split_materialized
from .model import build_model
from .train import encode_record


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max()
    p = np.exp(z)
    return (p / p.sum()).tolist()


def metrics(rows):
    """rows: [{type, workflow, target, probs}] -> 与 jev metrics() 一致的指标。"""
    correct, ces, briers, soft, confidence, score_errors = [], [], [], [], [], []
    for r in rows:
        p = np.asarray(r["probs"], dtype=np.float64)
        y = np.asarray(r["target"], dtype=np.float64)
        guess, gold = int(p.argmax()), int(y.argmax())
        correct.append(float(guess == gold))
        ces.append(float(-(y * np.log(np.clip(p, 1e-12, 1))).sum()))
        briers.append(float(((p - y) ** 2).sum()))
        soft.append(float(y[guess]))
        confidence.append(float(p.max()))
        if r["type"] == "score":
            score_errors.append(abs(float(np.arange(len(p)) @ (p - y))))
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        ix = [i for i, c in enumerate(confidence) if lo <= c < (lo + 0.1 if lo < 0.9 else 1.0000001)]
        if ix:
            ece += len(ix) / len(rows) * abs(np.mean([correct[i] for i in ix]) - np.mean([confidence[i] for i in ix]))
    out = {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "soft_cross_entropy": float(np.mean(ces)),
        "brier_sum": float(np.mean(briers)),
        "gold_mass_at_selected_option": float(np.mean(soft)),
        "ece_10_bins_vs_gold_argmax": float(ece),
    }
    if score_errors:
        out["score_expectation_mae"] = float(np.mean(score_errors))
    return out


def report(rows):
    return {
        "overall": metrics(rows),
        "by_type": {k: metrics([r for r in rows if r["type"] == k]) for k in sorted({r["type"] for r in rows})},
        "by_workflow": {k: metrics([r for r in rows if r["workflow"] == k]) for k in sorted({r["workflow"] for r in rows})},
    }


def load_model_from_checkpoint(backbone, checkpoint_dir, device):
    """用 kev 的 Checkpoint.load() 加载训练好的 checkpoint（LoRA adapter + head）。

    checkpoint_dir 需是 kev 兼容布局：根目录有 adapter_config.json + adapter_model.safetensors
    + head.pt（Meta 格式，含 base/base_revision/head/lora/head_dim/weights_dtype）。
    """
    from .checkpoint import Checkpoint
    tok, model = Checkpoint(str(checkpoint_dir)).load(device)
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--checkpoint", required=True, help="训练产出的 checkpoint 目录（含 adapter/ + head.pt）")
    ap.add_argument("--prepared", default="datas/typed-decisions/prepared")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "calibration", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="", help="输出 JSON 路径（默认打印到 stdout）")
    a = ap.parse_args()

    from .device import select
    dev = select(a.device, dtype=torch.bfloat16)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()

    model, tok = load_model_from_checkpoint(a.backbone, a.checkpoint, dev)

    recs = load_split_materialized(a.split, a.prepared)
    print(f"eval {a.split}: {len(recs)} records", file=sys.stderr)

    # 展开成 question 级 rows（对齐 jev 评测口径）
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(recs), 8):
            chunk = recs[start:start + 8]
            encs = [encode_record(model, tok, r) for r in chunk]
            with autocast:
                logits_b = model.forward_batch(encs, shared_prefix=False)
            for logits, rec in zip(logits_b, chunk):
                for z, q in zip(logits, rec["questions"]):
                    target = q["target"]
                    p = probabilities(z.float().cpu().tolist()[:len(target)])
                    rows.append({
                        "type": q["qtype"], "workflow": q["src"],
                        "target": target, "probs": p,
                    })

    rep = report(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
