"""jev_kev 训练：typed-decisions 软标签 LoRA + PointerHead 训练。

流程：
  1. 加载 typed-decisions train split（软标签 target）
  2. materialize 成 kev 内部 record
  3. encode（kev.model.encode，packed block-causal + pointer readout）
  4. LoRA 微调 + 从零训练 PointerHead，loss = soft cross-entropy（对齐 target 概率分布）
  5. 每 N step 在 dev 上评测，选 soft CE 最低的 checkpoint

参考 kev.train 的核心（question_loss 软 CE、AdamW 分组 lr），但去掉了 full_ft/FSDP/多源/augmentation
等复杂逻辑，专注 typed-decisions 这一任务。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from kev.device import select
from kev.model import MAX_STATE, MAX_BRANCH, MAX_PACKED

from .model import build_model
from .data import load_split, load_split_materialized

# typed-decisions 的 state 最长约 530 token，默认 384 不够；1024 足够且远小于 packed 上限 2048
JEV_MAX_STATE = 1024


def question_loss(z, target, dev, brier_w=0.1):
    """soft cross-entropy + 0.1 * sum-candidate Brier，对齐 agentjev_v1 的训练协议。

    ce = -sum(target * log_softmax(z)); brier = sum((softmax(z) - target)^2)
    """
    t = torch.tensor(target, device=dev, dtype=torch.float32)
    logp = F.log_softmax(z.float(), -1)
    ce = -(t * logp).sum()
    if brier_w:
        p = logp.exp()
        brier = ((p - t) ** 2).sum()
        return ce + brier_w * brier
    return ce


def encode_record(model, tok, rec, max_state=JEV_MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    enc = model.encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)
    if len(enc["ids"]) > max_packed:
        raise ValueError(f"record exceeds {max_packed} packed tokens")
    return enc


def batch_loss(model, encs, recs, dev, autocast):
    """一个 batch 的 loss：对每个 question 算 soft CE，取平均。"""
    with autocast:
        logits_b = model.forward_batch(encs, shared_prefix=False)
    total = 0.0
    n = 0
    for logits, rec in zip(logits_b, recs):
        for z, q in zip(logits, rec["questions"]):
            total = total + question_loss(z, q["target"], dev)
            n += 1
    if not torch.isfinite(total):
        raise ValueError("non-finite training loss")
    return total / n, n


def evaluate(model, tok, recs, dev, autocast, batch=8):
    """dev 评测：算每个 question 的 soft CE，返回 (mean_ce, per-record预测)。"""
    model.eval()
    total_ce = 0.0
    n = 0
    with torch.no_grad():
        for start in range(0, len(recs), batch):
            chunk = recs[start:start + batch]
            encs = [encode_record(model, tok, r) for r in chunk]
            with autocast:
                logits_b = model.forward_batch(encs, shared_prefix=False)
            for logits, rec in zip(logits_b, chunk):
                for z, q in zip(logits, rec["questions"]):
                    t = torch.tensor(q["target"], device=dev, dtype=torch.float32)
                    total_ce += float(-(t * F.log_softmax(z.float(), -1)).sum())
                    n += 1
    model.train()
    return total_ce / n


def save_checkpoint(model, out: Path, step: int, backbone: str, revision: str | None):
    """保存 kev 兼容 checkpoint：LoRA adapter 在根目录 + head.pt 为 kev Meta 格式。

    kev.checkpoint.Checkpoint.load() 期望：adapter_config.json + adapter_model.safetensors
    直接在 out/ 根目录，head.pt 是 Meta dict（含 base/base_revision/lora/head_dim/head/weights_dtype/weights）。
    """
    out.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(out)   # 直接存根目录：adapter_config.json + adapter_model.safetensors
    from kev.checkpoint import Meta
    meta = Meta(base=backbone, base_revision=revision, lora=16, head_dim=256,
                weights_dtype="bf16", weights="lora", head=model.head.state_dict(),
                extra={"step": step})
    torch.save(meta.to_dict(), out / "head.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--backbone-revision", default="")
    ap.add_argument("--prepared", default="datas/typed-decisions/prepared")
    ap.add_argument("--out", default="runs/jev_kev")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone LoRA 学习率（对齐 agentjev backbone_lr=1e-5）")
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=2)      # records per forward pass
    ap.add_argument("--accum", type=int, default=4)      # micro-batches per optimizer step
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-state", type=int, default=JEV_MAX_STATE)
    ap.add_argument("--checkpointing", type=int, choices=[0, 1], default=1,
                    help="gradient checkpointing on the backbone（hybrid DeltaNet 训练必开，否则 OOM）")
    a = ap.parse_args()

    dev = select(a.device, dtype=torch.bfloat16)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else contextlib.nullcontext()
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    out = Path(a.out)
    if out.exists() and (out / "head.pt").exists():
        raise SystemExit(f"{out} already holds a checkpoint; choose another --out")

    # 加载数据
    print("loading typed-decisions train/dev ...", flush=True)
    train_recs = load_split_materialized("train", a.prepared)
    dev_recs = load_split_materialized("dev", a.prepared)
    print(f"train={len(train_recs)} records, dev={len(dev_recs)} records", flush=True)

    # 构建模型
    revision = a.backbone_revision or None
    model = build_model(a.backbone, dev, lora=16, head_dim=256, revision=revision)
    model.backbone_name = a.backbone
    if a.checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    tok = model.tokenizer if hasattr(model, "tokenizer") else None
    # DecisionModel 不存 tokenizer，用 load_tokenizer
    from kev.model import load_tokenizer
    tok = load_tokenizer(a.backbone, revision=revision)
    print(f"device={dev} trainable params={sum(p.numel() for p in model.trainable_parameters())/1e6:.1f}M", flush=True)

    # 分组 lr
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    groups = [
        {"params": [p for p in model.trainable_parameters() if id(p) not in head_ids], "lr": a.lr},
        {"params": head_params, "lr": a.head_lr},
    ]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    steps = a.max_steps
    warmup = a.warmup_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, steps - warmup))),
    )

    # 预编码 train（typed-decisions state 短，全量编码省训练时开销）
    print("encoding train records ...", flush=True)
    encs = [encode_record(model, tok, r, max_state=a.max_state) for r in train_recs]

    order = list(range(len(encs)))
    random.shuffle(order)
    position = 0
    events = []
    t0 = time.time()

    # 初始 dev 评估（对齐 agentjev：selection 含 initialization，best_step 可为 0）
    model.eval()
    best_ce = evaluate(model, tok, dev_recs, dev, autocast)
    best_step = 0
    print(f"initial_dev_ce={best_ce:.4f}", flush=True)
    model.train()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        total_n = 0
        for _ in range(a.accum):
            ix = []
            for _ in range(a.batch):
                if position == len(order):
                    random.shuffle(order)
                    position = 0
                ix.append(order[position])
                position += 1
            batch_encs = [encs[i] for i in ix]
            batch_recs = [train_recs[i] for i in ix]
            loss, n = batch_loss(model, batch_encs, batch_recs, dev, autocast)
            (loss / a.accum).backward()
            total += float(loss.detach())
            total_n += n
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()

        rec = {"step": step, "loss": total / total_n, "seconds": round(time.time() - t0, 2)}
        if step % a.eval_every == 0:
            ce = evaluate(model, tok, dev_recs, dev, autocast)
            rec["dev_ce"] = ce
            if ce < best_ce:
                best_ce, best_step = ce, step
                save_checkpoint(model, out, step, a.backbone, revision)
            print(json.dumps(rec), flush=True)
        elif step % 20 == 0:
            print(json.dumps(rec), flush=True)
        events.append(rec)

    (out / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (out / "selection.json").write_text(json.dumps({
        "best_step": best_step, "dev_soft_cross_entropy": best_ce,
    }, indent=2) + "\n")
    print(f"done: best_step={best_step} best_dev_ce={best_ce:.4f}", flush=True)


if __name__ == "__main__":
    main()
