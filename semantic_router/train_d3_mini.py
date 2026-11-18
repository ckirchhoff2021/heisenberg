"""d3-mini 微调训练入口（4 卡 DDP，多模态图片版）。

在 vLLM Semantic Router 的 d3-mini（Decision 3.0，Qwen3.5-4B backbone + 255 路
code-readout）基础上，用 jev-mixed 数据集（含曲线图）微调。

训练对象（冻结 vision，已证伪「解冻 vision 会崩」）：
  1. LoRA（r=16）挂在 language_model 上（DeltaNet 混合投影 + 全注意力投影）
  2. readout 头（255 × 2560，d3 自带的 code 打分矩阵，直接微调）

前向复用 d3 的 noncausal_full_attention 路径：
  processor(texts, images) -> image_embeddings(含 masked_scatter 图特征 + mrope)
  -> PromptEncoder(embeds, mask, mrope) -> 最后 token hidden @ readout.T
  -> 255 路 logits，按题目选项数截断后 soft CE + Brier。

关键点：
  - image_embeddings 与 PromptEncoder 均不含 inference_mode 包装，可在 autograd
    下反传（已 smoke 验证 LoRA 496/496 + readout 有梯度、vision 冻结无梯度）。
  - 图片路径与 zero-shot 评测一致：state=中文规则 + d3 英文 system prompt。

用法（4 卡）：
  bash semantic_router/train_d3_mini_4gpu.sh
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

D3_DIR_DEFAULT = os.environ.get("D3_DIR", "checkpoints/d3-mini")
sys.path.insert(0, D3_DIR_DEFAULT)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def question_loss(logits, target, brier_w=0.1):
    """soft cross-entropy + Brier（对齐 jev_kev 协议）。logits 已截断到选项数。"""
    t = torch.tensor(target, device=logits.device, dtype=torch.float32)
    logp = F.log_softmax(logits.float(), -1)
    ce = -(t * logp).sum()
    p = logp.exp()
    brier = ((p - t) ** 2).sum()
    return ce + brier_w * brier


class D3FinetuneModel(torch.nn.Module):
    """d3-mini 微调模型：backbone(LoRA) + PromptEncoder + 可训练 readout。"""

    def __init__(self, d3_dir, device, lora=16, freeze_vision=True):
        super().__init__()
        from d3_runtime import PromptEncoder
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        self.backbone = Qwen3_5Model.from_pretrained(
            d3_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=device)
        if freeze_vision:
            for p in self.backbone.visual.parameters():
                p.requires_grad = False

        if lora:
            from peft import LoraConfig, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                       "out_proj"]
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora,
                             lora_alpha=2 * lora, lora_dropout=0.05,
                             target_modules=targets)
            self.backbone.language_model = get_peft_model(
                self.backbone.language_model, cfg)

        rd = load_file(os.path.join(d3_dir, "readout.safetensors"))["weight"]
        self.readout = torch.nn.Parameter(rd.to(device).float())
        self.encoder = PromptEncoder(self.backbone.language_model)
        self.device = device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, ids, mask, pixel_values=None, image_grid_thw=None,
                mm_token_type_ids=None, mrope=None):
        """DDP 兼容入口：图片/文本统一前向，返回 [B, 255] logits。"""
        return self.forward_logits(ids, mask, pixel_values, image_grid_thw,
                                   mm_token_type_ids, mrope)

    def forward_logits(self, ids, mask, pixel_values=None, image_grid_thw=None,
                       mm_token_type_ids=None, mrope=None):
        """noncausal 前向，返回每行最后 token 的 255 路 logits [B, 255]。

        图片路径：embeds 用 get_input_embeddings + masked_scatter 图特征，
        mrope 用 compute_3d_position_ids；文本路径 mrope=None。
        """
        embeds = self.backbone.get_input_embeddings()(ids)
        if pixel_values is not None:
            pooled = self.backbone.get_image_features(
                pixel_values, image_grid_thw, return_dict=True).pooler_output
            pooled = torch.cat(pooled, dim=0).to(embeds.device, embeds.dtype)
            slots, _ = self.backbone.get_placeholder_mask(
                ids, inputs_embeds=embeds, image_features=pooled)
            embeds = embeds.masked_scatter(slots, pooled)
            mrope = self.backbone.compute_3d_position_ids(
                input_ids=ids, image_grid_thw=image_grid_thw,
                video_grid_thw=None, inputs_embeds=embeds,
                attention_mask=mask, past_key_values=None,
                mm_token_type_ids=mm_token_type_ids)
        hidden = self.encoder(embeds, mask, mrope)  # [B, L, D]
        last = hidden[:, -1].float()  # [B, D]
        return last @ self.readout.T  # [B, 255]


def build_dataset(d3, mixed_file, split, limit=None):
    """用 d3 的 prepare 逻辑把 jev-mixed 记录转成训练样本（含图片）。

    返回 list of dict：{images(PIL), texts(str), counts(int), target(list), task_id}。
    每样本一条 choice 题；texts 含 image placeholder + d3 system prompt +
    中文规则 + 问题 + 选项 code；target 为选项下标 one-hot。
    """
    d = json.load(open(mixed_file, encoding="utf-8"))
    schema = d["schema"]
    recs = d["splits"][split]
    if limit:
        recs = recs[:limit]

    samples = []
    for rec in recs:
        task_id = rec["task_id"]
        task = schema["tasks"][task_id]
        system = schema["systems"][task["system_ref"]]
        data_root = schema["data_roots"][task["data_root_ref"]]
        img_path = os.path.join(data_root, rec["state"]["images"][0])
        choices = task["choices"]
        question = {"type": "choice", "instructions": task["question"],
                    "criteria": {c: None for c in choices}}
        try:
            prepared = d3.prepare(state=system, questions={"q": question},
                                  images=[img_path])
        except Exception:
            continue
        if "q" not in prepared.texts or not prepared.images:
            continue
        target = [0.0] * len(choices)
        target[rec["answer"]] = 1.0
        samples.append({"images": prepared.images, "texts": prepared.texts["q"],
                        "count": len(choices), "target": target,
                        "task_id": task_id, "img_path": img_path})
    return samples


def forward_batch(d3, model, batch, device, autocast, is_ddp):
    """对一批样本（共享一张图 + 一条 prompt）做前向，返回 [B, counts_max] logits 列表。"""
    # d3 图片路径：一次处理一个样本的图 + prompt（图不共享，逐样本前向）
    # 为效率，这里每条样本单独前向（batch=1 时无损失；batch>1 时逐条）
    logits_out = []
    for s in batch:
        images = s["images"]
        texts = s["texts"]
        encoded = d3.processor(text=texts, images=images, padding=True,
                               return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        kw = {}
        if "pixel_values" in inputs:
            kw["pixel_values"] = inputs["pixel_values"]
            kw["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            kw["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        with autocast:
            logits = model(ids, mask, **kw) if is_ddp else model.forward_logits(ids, mask, **kw)
        logits_out.append(logits[0])
    return logits_out


def evaluate(d3, model, samples, device, autocast, is_ddp, rank=0, world=1,
             distm=None, eval_limit=None):
    """dev 评测（逐样本前向）。

    DDP 下所有 rank 分片评测后 all_reduce 汇总 —— 保证每个 rank 都参与评测，
    避免「rank 0 在 eval、其余 rank 在训练 backward」的 NCCL 死锁。
    """
    model.eval()
    local = samples
    if eval_limit:
        # 稳定子集：取前 eval_limit 条（各 rank 一致）
        local = local[:eval_limit]
    if world > 1:
        local = local[rank::world]  # 分片
    sums = {"ce": 0.0, "brier": 0.0, "acc": 0.0}
    n = 0
    with torch.no_grad():
        for s in local:
            logits = forward_batch(d3, model, [s], device, autocast, is_ddp)[0]
            z = logits[: s["count"]]
            t = torch.tensor(s["target"], device=device, dtype=torch.float32)
            logp = F.log_softmax(z.float(), -1)
            p = logp.exp()
            sums["ce"] += (-(t * logp).sum()).item()
            sums["brier"] += ((p - t) ** 2).sum().item()
            sums["acc"] += float((p.argmax() == t.argmax()).item())
            n += 1
    if world > 1:
        stats = torch.tensor([sums["ce"], sums["brier"], sums["acc"], float(n)],
                             device=device, dtype=torch.float32)
        distm.all_reduce(stats)
        sums = {"ce": stats[0].item(), "brier": stats[1].item(),
                "acc": stats[2].item()}
        n = int(stats[3].item())
    model.train()
    return {"ce": sums["ce"] / n, "brier": sums["brier"] / n, "acc": sums["acc"] / n}


def init_distributed():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-dir", default=D3_DIR_DEFAULT)
    ap.add_argument("--mixed-file", required=True)
    ap.add_argument("--output-dir", default="outputs/d3_mini_finetuned")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--readout-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--brier-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=256)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--freeze-vision", type=int, choices=[0, 1], default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank, world, local_rank, distm = init_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if world > 1 else "cuda:0"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sys.path.insert(0, args.d3_dir)  # d3 runtime 目录按 --d3-dir 动态加入（flash/mini 共用）
    from d3_runtime import D3
    d3_runner = D3.from_pretrained(args.d3_dir, device=device, verify="none",
                                   batch_size=8)

    if is_main:
        print("loading data ...", flush=True)
    train_samples = build_dataset(d3_runner, args.mixed_file, "train", args.limit)
    dev_samples = build_dataset(d3_runner, args.mixed_file, "eval", args.limit)
    if is_main:
        print(f"train={len(train_samples)} dev={len(dev_samples)}", flush=True)

    model = D3FinetuneModel(args.d3_dir, device, lora=args.lora,
                            freeze_vision=bool(args.freeze_vision))
    model.backbone.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone.config.use_cache = False
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    raw = model.module if world > 1 else model
    if is_main:
        tp = sum(p.numel() for p in raw.trainable_parameters()) / 1e6
        print(f"trainable params={tp:.1f}M", flush=True)

    readout_ids = {id(raw.readout)}
    groups = [
        {"params": [p for p in raw.trainable_parameters()
                    if id(p) not in readout_ids], "lr": args.lr},
        {"params": [raw.readout], "lr": args.readout_lr},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * 0.5
        * (1 + math.cos(math.pi * max(0, s - args.warmup)
                        / max(1, args.max_steps - args.warmup))),
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    rng = random.Random(args.seed)
    order = list(range(len(train_samples)))
    usable = (len(order) // (args.batch * world)) * args.batch * world
    order = order[:usable]
    per_rank = usable // world
    mine = order[rank * per_rank:(rank + 1) * per_rank]

    best_ce = float("inf")
    best_step = 0
    if dev_samples:
        base = evaluate(d3_runner, raw, dev_samples, device, autocast, world > 1,
                        rank=rank, world=world, distm=distm,
                        eval_limit=args.eval_limit)
        if is_main:
            print(f"[initial] dev {base}", flush=True)
            best_ce = base["ce"]

    def save(step):
        from safetensors.torch import save_file
        os.makedirs(args.output_dir, exist_ok=True)
        raw.backbone.language_model.save_pretrained(args.output_dir)
        save_file({"weight": raw.readout.detach().float().cpu()},
                  os.path.join(args.output_dir, "readout.safetensors"))
        meta = {"step": step, "base": args.d3_dir, "lora": args.lora,
                "freeze_vision": bool(args.freeze_vision),
                "readout_shape": list(raw.readout.shape)}
        torch.save(meta, os.path.join(args.output_dir, "finetune_meta.pt"))
        print(f"  [save] step={step} -> {args.output_dir}", flush=True)

    position = 0
    start = time.monotonic()
    for step in range(1, args.max_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum):
            ix = []
            for _ in range(args.batch):
                if position >= len(mine):
                    rng.shuffle(mine)
                    position = 0
                ix.append(mine[position])
                position += 1
            batch = [train_samples[i] for i in ix]
            logits = forward_batch(d3_runner, model, batch, device, autocast,
                                   world > 1)
            loss = 0.0
            for z, s in zip(logits, batch):
                loss = loss + question_loss(z[: s["count"]], s["target"],
                                            args.brier_weight)
            loss = loss / len(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            (loss / args.accum).backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(raw.trainable_parameters(), args.grad_clip,
                                       error_if_nonfinite=True)
        opt.step()
        sched.step()

        record = {"step": step, "loss": round(total / args.accum, 4),
                  "seconds": round(time.monotonic() - start, 1)}
        if dev_samples and step % args.eval_every == 0:
            dev = evaluate(d3_runner, raw, dev_samples, device, autocast, world > 1,
                           rank=rank, world=world, distm=distm,
                           eval_limit=args.eval_limit)
            if is_main:
                record["dev"] = dev
                if dev["ce"] < best_ce:
                    best_ce, best_step = dev["ce"], step
                    save(step)
        if is_main:
            print(json.dumps(record), flush=True)

    if is_main:
        if not dev_samples:
            save(args.max_steps)
            best_step = args.max_steps
        print(f"done. best_step={best_step} best_dev_ce={best_ce:.4f}", flush=True)
    if world > 1:
        distm.barrier()
        distm.destroy_process_group()


if __name__ == "__main__":
    main()
