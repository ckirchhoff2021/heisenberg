"""JEV 训练入口：多模态 Qwen3.5 基座 + 置换等变候选头，全参微调。

在选择题数据上训练，按 dev 软交叉熵选优；bf16 混合精度，两参组 AdamW
（backbone/head 不同学习率），指标写 tensorboard。

用法：
  python train/jev.py \
      --backbone checkpoints/Qwen3.5-0.8B \
      --train-file datas/aiops/files/jsons/argos-trains.json \
      --dev-file datas/aiops/files/jsons/argos-valids.json

输入 json 既可是标准 JEV 选择题，也可是 argos 标签数据 (curve/recover/image/from)，
后者会自动按每张图 trend/recover 两题转换，并以 --prompt-file 注入分类规则。
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from dataset.jev import (
    DEFAULT_FILES_ROOT,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_FILE,
    JEVDatasets,
    JevTorchDataset,
    build_argos_jev_dataset,
)
from models.jev import JevModel

ARGOS_FIELDS = {"curve", "recover", "image", "from"}


def build_split(path, files_root, prompt_file, tokenizer, image_processor,
                max_tokens, max_pixels, limit=None):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        first = json.load(f)[0]
    if ARGOS_FIELDS <= set(first):
        return build_argos_jev_dataset(path, files_root, tokenizer, image_processor,
                                       prompt_file, max_tokens, max_pixels, limit)
    raw = JEVDatasets(path, files_root=files_root, answer_format="index").build()
    return JevTorchDataset(raw, tokenizer, image_processor,
                           max_tokens=max_tokens, max_pixels=max_pixels)


def move(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def soft_loss(model, batch, brier_weight, perm_weight):
    out = model(batch, perm_reg=perm_weight > 0)
    mask = batch["cand_mask"]
    target = batch["target"]
    logits = out["logits"].float().masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
    probs = logp.exp() * mask
    ce = -(target * logp).sum(-1).mean()
    brier = ((probs - target) ** 2).sum(-1).mean()
    loss = ce + brier_weight * brier
    if perm_weight > 0:
        lp = out["logits_perm"].float().masked_fill(~mask, float("-inf"))
        q, qp = torch.softmax(logits, -1), torch.softmax(lp, -1)
        perm_kl = (q * (torch.log_softmax(logits, -1) - torch.log_softmax(lp, -1))).sum(-1).mean()
        loss = loss + perm_weight * perm_kl
    acc = (probs.argmax(-1) == target.argmax(-1)).float().mean()
    return loss, {"ce": ce.detach(), "brier": brier.detach(), "acc": acc.detach()}


def evaluate(model, dataset, device, brier_weight, amp_dtype, chunk=2,
             rank=0, world=1, distm=None):
    """dev 评测；DDP 下按 rank 切分题目并行评测，再 all_reduce 汇总，避免空 rank 等 barrier 超时。"""
    model.eval()
    sums = {"ce": 0.0, "brier": 0.0, "acc": 0.0}
    count = 0
    starts = [s for i, s in enumerate(range(0, len(dataset), chunk)) if i % world == rank]
    with torch.inference_mode():
        for start in starts:
            batch = move(dataset.collate([dataset[i] for i in range(start, min(start + chunk, len(dataset)))]), device)
            with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"), dtype=amp_dtype):
                _, s = soft_loss(model, batch, brier_weight, 0.0)
            k = batch["cand_mask"].size(0)
            for key in sums:
                sums[key] += s[key].item() * k
            count += k
    totals = torch.tensor([sums["ce"], sums["brier"], sums["acc"], float(count)], device=device)
    if world > 1:
        distm.all_reduce(totals)
    ce, brier, acc, n = totals.tolist()
    model.train()
    return {"ce": ce / n, "brier": brier / n, "acc": acc / n}


def init_distributed():
    """torchrun 环境下初始化 DDP；普通单进程启动时返回单卡占位值。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import datetime
        import torch.distributed as dist
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, dist
    return 0, 1, 0, None


def microbatches(dataset, microbatch, seed, rank=0, world=1):
    """每 epoch 洗牌后按 rank 切分问题；各 rank 微批数严格一致（DDP 同步要求）。"""
    rng = random.Random(seed)
    order = list(range(len(dataset)))
    while True:
        rng.shuffle(order)
        usable = (len(order) // (microbatch * world)) * microbatch * world
        order = order[:usable]
        per_rank = usable // world
        mine = order[rank * per_rank:(rank + 1) * per_rank]
        for start in range(0, len(mine) - microbatch + 1, microbatch):
            yield dataset.collate([dataset[i] for i in mine[start:start + microbatch]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="Qwen3.5 多模态基座目录")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", default=None)
    parser.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE,
                        help="argos 数据注入的分类规则；传空字符串则不加 system")
    parser.add_argument("--limit", type=int, default=None, help="只取每个文件前 N 条 argos 记录（调试用）")
    parser.add_argument("--output-dir", default="outputs/jev")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--microbatch", type=int, default=4, help="每个微批的问题数")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--perm-reg-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                        help="对 backbone 逐层梯度检查点（A40 显存下必须开启；--no-grad-checkpointing 关闭）")
    args = parser.parse_args()

    rank, world, local_rank, dist = init_distributed()
    is_main = rank == 0
    device = args.device
    if device == "auto":
        device = f"cuda:{local_rank}" if world > 1 else (
            "cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(args.backbone, local_files_only=True)

    train_ds = build_split(args.train_file, args.files_root, args.prompt_file, tokenizer,
                           image_processor, args.max_tokens, args.max_pixels, args.limit)
    if train_ds is None:
        raise FileNotFoundError(f"train file not found: {args.train_file}")
    dev_ds = build_split(args.dev_file, args.files_root, args.prompt_file, tokenizer,
                         image_processor, args.max_tokens, args.max_pixels, args.limit)

    model = JevModel(args.backbone, dtype=amp_dtype).to(device)
    if args.grad_checkpointing:
        # torch 参考 GDN kernel 28 层 fp32 中间矩阵在 autograd 下会 OOM，checkpoint 改为反向时重算
        model.path_encoder.backbone.gradient_checkpointing_enable()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    raw = model.module if world > 1 else model
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if ".backbone." in n], "lr": args.backbone_lr},
        {"params": [p for n, p in model.named_parameters() if ".backbone." not in n], "lr": args.head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    steps, warmup = args.max_steps, args.warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup))
        * 0.5 * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, steps - warmup))),
    )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(args.output_dir, "tb")) if is_main else None
    batches = microbatches(train_ds, args.microbatch, args.seed, rank, world)
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    best_ce, best_step = float("inf"), 0
    if dev_ds is not None:
        base = evaluate(raw, dev_ds, device, args.brier_weight, amp_dtype,
                        rank=rank, world=world, distm=dist)
        if is_main:
            print(f"[initial] dev {base}", flush=True)
            best_ce = base["ce"]

    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = 0.0
        agg = {}
        for _ in range(args.grad_accum):
            batch = move(next(batches), device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                loss, stats = soft_loss(model, batch, args.brier_weight, args.perm_reg_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            (loss / args.grad_accum).backward()
            totals += loss.detach().item() / args.grad_accum
            for k, v in stats.items():
                agg.setdefault(k, 0.0)
                agg[k] += v.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("nonfinite gradient")
        optimizer.step()
        scheduler.step()

        if is_main:
            writer.add_scalar("train/loss", totals, step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            for k, v in agg.items():
                writer.add_scalar(f"train/{k}", v, step)

        record = {"step": step, "loss": round(totals, 4), "acc": round(agg["acc"], 4),
                  "seconds": round(time.monotonic() - start, 1)}
        if dev_ds is not None and step % args.eval_every == 0:
            dev = evaluate(raw, dev_ds, device, args.brier_weight, amp_dtype,
                           rank=rank, world=world, distm=dist)
            if is_main:
                writer.add_scalar("dev/ce", dev["ce"], step)
                writer.add_scalar("dev/acc", dev["acc"], step)
                record["dev"] = dev
                if dev["ce"] < best_ce:
                    best_ce, best_step = dev["ce"], step
                    torch.save({"state_dict": raw.state_dict(), "step": step,
                                "input_schema": "jev.multimodal.v1", "backbone": args.backbone,
                                "max_pixels": args.max_pixels, "max_tokens": args.max_tokens},
                               os.path.join(args.output_dir, "best.pt"))
        if is_main:
            print(record, flush=True)

    if is_main:
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        if dev_ds is None:  # 无 dev 时保存最后一步
            torch.save({"state_dict": raw.state_dict(), "step": steps,
                        "input_schema": "jev.multimodal.v1", "backbone": args.backbone},
                       ckpt_path)
            best_step = steps
        writer.close()
        checkpoint = ckpt_path if (dev_ds is None or best_step > 0) else None
        print(f"done. best_step={best_step} best_dev_ce={best_ce:.4f} checkpoint={checkpoint}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
