"""jev_kev 多模态 argos 曲线分类训练入口（4 卡 DDP）。

参考 train/jev.py 的 DDP 框架（torchrun + init_distributed + 按 rank 切分
microbatches），但模型用 jev_kev 的 DecisionModel（Qwen3_5Model 多模态骨干 +
LoRA + PointerHead），loss 用 soft CE + Brier（对齐 typed-decisions 协议，但
target 为 one-hot）。

用法（4 卡）：
  .venv/bin/torchrun --standalone --nproc_per_node=4 jev_kev/train_argos.py \
      --backbone checkpoints/Qwen3.5-0.8B \
      --train-file datas/aiops/files/jsons/argos-trains.json \
      --dev-file datas/aiops/files/jsons/argos-valids.json \
      --files-root datas/aiops/files \
      --prompt-file datas/aiops/files/prompts/argos-jev-prompt.md \
      --output-dir outputs/jev_kev_argos
"""
from __future__ import annotations

import os

import argparse
import datetime
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from source.data import load_argos_records, load_mixed_records
from source.model import build_model

# argos state 很大：分类规则 ~2041 + 图像 ~244 ≈ 2300 token，默认 3072 足够
ARGO_MAX_STATE = 3072


def move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def question_loss(z, target, dev, brier_w=0.1):
    """soft cross-entropy + Brier（对齐 jev_kev typed-decisions 协议）。"""
    t = torch.tensor(target, device=dev, dtype=torch.float32)
    logp = F.log_softmax(z.float(), -1)
    ce = -(t * logp).sum()
    p = logp.exp()
    brier = ((p - t) ** 2).sum()
    return ce + brier_w * brier


def encode_record(model, tok, rec, max_state, max_branch=2560, max_pixels=None):
    enc = model.encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True,
                       max_pixels=max_pixels)
    return enc


def batch_loss(model, encs, recs, dev, autocast, brier_w):
    with autocast:
        # DDP：model(encs) 走 forward，梯度在 backward 时自动 all-reduce
        logits_b = model(encs)
    total = 0.0
    n = 0
    for logits, rec in zip(logits_b, recs):
        for z, q in zip(logits, rec["questions"]):
            total = total + question_loss(z, q["target"], dev, brier_w)
            n += 1
    return total / n, n


def evaluate(model, tok, recs, dev, autocast, max_state, max_pixels=None,
             rank=0, world=1, distm=None):
    """dev 评测：DDP 下按 rank 切分，all_reduce 汇总。返回 dict(ce, brier, acc)。"""
    model.eval()
    sums = {"ce": 0.0, "brier": 0.0, "acc": 0.0}
    count = 0
    starts = [s for i, s in enumerate(range(0, len(recs), 1)) if i % world == rank]
    with torch.no_grad():
        for s in starts:
            rec = recs[s]
            enc = encode_record(model, tok, rec, max_state, max_pixels=max_pixels)
            with autocast:
                logits_b = model.forward_batch([enc], shared_prefix=False)
            for z, q in zip(logits_b[0], rec["questions"]):
                t = torch.tensor(q["target"], device=dev, dtype=torch.float32)
                logp = F.log_softmax(z.float(), -1)
                p = logp.exp()
                ce = -(t * logp).sum()
                brier = ((p - t) ** 2).sum()
                acc = float((p.argmax() == t.argmax()).item())
                sums["ce"] += ce.item()
                sums["brier"] += brier.item()
                sums["acc"] += acc
                count += 1
    totals = torch.tensor([sums["ce"], sums["brier"], sums["acc"], float(count)], device=dev)
    if world > 1:
        distm.all_reduce(totals)
    ce, brier, acc, n = totals.tolist()
    model.train()
    return {"ce": ce / n, "brier": brier / n, "acc": acc / n}


def init_distributed():
    """torchrun 环境下初始化 DDP；普通单进程启动时返回单卡占位值。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, dist
    return 0, 1, 0, None


def save_checkpoint(model, out_dir, step, backbone, max_state, lora):
    """保存 checkpoint。

    lora > 0：存 LoRA adapter（language_model）+ head.pt（kev 兼容布局）。
    lora == 0：全量保存（含 vision tower + language_model）+ head.pt。
    """
    import torch
    from source.checkpoint import Meta
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    os.makedirs(out_dir, exist_ok=True)
    if lora:
        if raw.multimodal:
            raw.lm.language_model.save_pretrained(out_dir)
        else:
            raw.lm.save_pretrained(out_dir)
        weights = "lora"
    else:
        # 全量：raw.lm 是完整 Qwen3_5Model（含 visual + language_model）。
        raw.lm.save_pretrained(out_dir, safe_serialization=True)
        weights = "full"
    meta = Meta(base=backbone, lora=lora, head_dim=256, weights_dtype="bf16",
                weights=weights, head=raw.head.state_dict(),
                extra={"step": step, "max_state": max_state, "multimodal": raw.multimodal})
    torch.save(meta.to_dict(), os.path.join(out_dir, "head.pt"))


def verify_saved(raw, out_dir, lora):
    """每次 best-save 后自检：磁盘 checkpoint 是否 == 当前内存权重。

    LoRA 模式校验 adapter + head；全量模式校验 model.safetensors 存在且含
    vision tower 权重 + head（防止历史「vision tower 未保存」bug 复发）。
    """
    ok = True
    # head（两种模式都校验）
    cur_head = {k: v.float().cpu() for k, v in raw.head.state_dict().items()}
    saved_head = torch.load(os.path.join(out_dir, "head.pt"), map_location="cpu")["head"]
    for k, v in cur_head.items():
        if k not in saved_head or not torch.allclose(v, saved_head[k].float(), atol=1e-3):
            print(f"  [verify] head 不一致/缺失: {k}", flush=True); ok = False; break
    if lora:
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict
        cur = {k: v.float().cpu() for k, v in get_peft_model_state_dict(raw.lm.language_model).items()}
        saved = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
        for k, v in cur.items():
            if k not in saved or not torch.allclose(v, saved[k].float(), atol=1e-3):
                print(f"  [verify] adapter 不一致/缺失: {k}", flush=True); ok = False; break
    else:
        from safetensors.torch import load_file
        from pathlib import Path
        full = os.path.join(out_dir, "model.safetensors")
        if not Path(full).exists():
            print("  [verify] 全量 model.safetensors 缺失", flush=True); ok = False
        else:
            saved = load_file(full)
            visual_keys = [k for k in saved if ".visual." in k or k.startswith("visual.")]
            if not visual_keys:
                print("  [verify] vision tower 未保存（无 visual keys）", flush=True); ok = False
            else:
                print(f"  [verify] vision tower 已保存 ({len(visual_keys)} keys)", flush=True)
    print(f"  [verify] save/load 权重{'一致 OK' if ok else '不一致 FAILED'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="Qwen3.5 多模态基座目录")
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--mixed-file", default=None,
                    help="jev-mixed.json 混合数据集（单题多任务，替代 --train-file/--dev-file）")
    ap.add_argument("--files-root", required=True)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/jev_kev_argos")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1, help="每卡每步的 record 数")
    ap.add_argument("--accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--brier-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--max-state", type=int, default=ARGO_MAX_STATE)
    ap.add_argument("--max-pixels", type=int, default=512 * 512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--grad-checkpointing", type=int, choices=[0, 1], default=1,
                    help="hybrid DeltaNet 训练必开，否则 OOM")
    args = ap.parse_args()

    rank, world, local_rank, distm = init_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if world > 1 else "cuda"
    amp_dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # tokenizer + image processor
    from source.model import load_tokenizer
    tok = load_tokenizer(args.backbone)
    image_processor = AutoImageProcessor.from_pretrained(args.backbone, local_files_only=True)

    # 数据（每卡都加载全量，microbatch 按 rank 切分）
    if is_main:
        print(f"loading data ...", flush=True)
    if args.mixed_file:
        train_recs = load_mixed_records(args.mixed_file, "train", args.limit)
        dev_recs = load_mixed_records(args.mixed_file, "eval", args.limit)
    else:
        train_recs = load_argos_records(args.train_file, args.files_root, args.prompt_file, args.limit)
        dev_recs = load_argos_records(args.dev_file, args.files_root, args.prompt_file, args.limit) if args.dev_file else None
    if is_main:
        print(f"train={len(train_recs)} records, dev={len(dev_recs) if dev_recs else 0} records", flush=True)

    # 模型
    model = build_model(args.backbone, device, lora=args.lora, head_dim=args.head_dim,
                        multimodal=True, image_processor=image_processor)
    if args.grad_checkpointing:
        if args.lora:
            model.lm.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            # 全量：直接作用在 Qwen3_5Model 上（含 vision tower + language_model）。
            model.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    raw = model.module if world > 1 else model
    if is_main:
        print(f"device={device} trainable params={sum(p.numel() for p in raw.trainable_parameters())/1e6:.1f}M", flush=True)

    # 分组 lr
    head_params = list(raw.head.parameters())
    head_ids = {id(p) for p in head_params}
    groups = [
        {"params": [p for p in raw.trainable_parameters() if id(p) not in head_ids], "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    steps, warmup = args.max_steps, args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5
        * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, steps - warmup))),
    )
    autocast = torch.autocast("cuda", dtype=amp_dtype)

    # 预编码 train（argos state 大，编码一次；图像处理每卡都做）
    if is_main:
        print(f"encoding train records (max_state={args.max_state}) ...", flush=True)
    encs = [encode_record(raw, tok, r, args.max_state, max_pixels=args.max_pixels) for r in train_recs]

    # 按 rank 切分数据（对齐 jev.py 的 microbatches：各 rank 微批数一致）
    rng = random.Random(args.seed)
    order = list(range(len(encs)))
    usable = (len(order) // (args.batch * world)) * args.batch * world
    order = order[:usable]
    per_rank = usable // world
    mine = order[rank * per_rank:(rank + 1) * per_rank]

    best_ce = float("inf")
    best_step = 0
    if dev_recs is not None:
        base = evaluate(raw, tok, dev_recs, device, autocast, args.max_state,
                        max_pixels=args.max_pixels, rank=rank, world=world, distm=distm)
        if is_main:
            print(f"[initial] dev {base}", flush=True)
            best_ce = base["ce"]

    position = 0
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        total_n = 0
        for _ in range(args.accum):
            ix = []
            for _ in range(args.batch):
                if position >= len(mine):
                    rng.shuffle(mine)
                    position = 0
                ix.append(mine[position])
                position += 1
            batch_encs = [encs[i] for i in ix]
            batch_recs = [train_recs[i] for i in ix]
            # 用 model（DDP）调用 forward，梯度在 backward 时自动 all-reduce
            loss, n = batch_loss(model, batch_encs, batch_recs, device, autocast, args.brier_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            (loss / args.accum).backward()
            total += float(loss.detach())
            total_n += n
        torch.nn.utils.clip_grad_norm_(raw.trainable_parameters(), args.grad_clip, error_if_nonfinite=True)
        opt.step()
        sched.step()

        record = {"step": step, "loss": round(total / total_n, 4),
                  "seconds": round(time.monotonic() - start, 1)}
        if dev_recs is not None and step % args.eval_every == 0:
            dev = evaluate(raw, tok, dev_recs, device, autocast, args.max_state,
                           max_pixels=args.max_pixels, rank=rank, world=world, distm=distm)
            if is_main:
                record["dev"] = dev
                if dev["ce"] < best_ce:
                    best_ce, best_step = dev["ce"], step
                    save_checkpoint(model, args.output_dir, step, args.backbone, args.max_state, args.lora)
                    verify_saved(raw, args.output_dir, args.lora)
        if is_main:
            print(json.dumps(record), flush=True)

    if is_main:
        if dev_recs is None:
            save_checkpoint(model, args.output_dir, steps, args.backbone, args.max_state, args.lora)
            best_step = steps
        print(f"done. best_step={best_step} best_dev_ce={best_ce:.4f}", flush=True)
    if world > 1:
        distm.barrier()
        distm.destroy_process_group()


if __name__ == "__main__":
    main()
