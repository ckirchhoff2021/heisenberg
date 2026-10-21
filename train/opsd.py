"""OPSD / 全参数 SFT 训练入口（DDP 版，torchrun 启动）。

两种模式：
  --mode sft   全参数微调 + hard CE（纯生成式 baseline，无教师、无采样）
               loss = cross_entropy(学生 logits[prompt:] , 黄金标签 ids)
  --mode opsd  On-policy Privileged Self-Distillation
               教师 = 与学生同架构 base（原始权重冻结）
               特权信息 = 黄金标签注入教师上下文
               采样 = 学生 on-policy 生成轨迹 <trend>..</trend>\n<recover>..</recover>
               监督 = reverse KL(学生 logits ∥ 教师 logits)，作用在轨迹 token
               更新 = 只更新学生，全参数（解冻 vision）

序列构造（特权注入）：
  学生 input = [img_block][prompt][traj]            —— 预测 traj
  教师 input = [img_block][prompt][gold][traj]      —— 已知答案后预测 traj
  两者轨迹 logits 对齐（都预测 traj 第 i 个 token），直接算 reverse KL。

DDP 说明：
  学生模型用 DDP 包装（loss 经 wrapper forward 以触发梯度 all_reduce）；
  教师模型不 DDP（每个 rank 独立持有冻结副本，只用于 no_grad forward）；
  采样用 raw student（no_grad 下避免 DDP reducer 开销）。

用法（4 卡）：
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
      train/opsd.py --mode sft --backbone ... --train-file ... --dev-file ...
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
      train/opsd.py --mode opsd --backbone ... --train-file ... --dev-file ...
"""
import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_FILES_ROOT = "datas/aiops/files"
DEFAULT_PROMPT_FILE = "datas/aiops/files/prompts/argos-dp-prompt-noreason.md"
DEFAULT_MAX_PIXELS = 512 * 512

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"

TREND_TAG = re.compile(r"<trend>(.*?)</trend>", re.DOTALL)
RECOVER_TAG = re.compile(r"<recover>(.*?)</recover>", re.DOTALL)

VALID_TREND = {"normal", "offset", "increase", "decline", "spike", "fluctuate", "other"}
VALID_RECOVER = {"recovered", "recovering", "no_recover", "other"}


def gold_text(curve: str, recover: str) -> str:
    return f"<trend>{curve}</trend>\n<recover>{recover}</recover>"


def encode_image(img_path: str, image_processor, max_pixels: int):
    img = Image.open(img_path).convert("RGB")
    feat = image_processor(images=[img], return_tensors="pt", max_pixels=max_pixels)
    pixel_values = feat["pixel_values"]
    image_grid_thw = feat["image_grid_thw"]
    merge = getattr(image_processor, "merge_size", 2)
    n_pad = int(image_grid_thw[0].prod().item() // (merge ** 2))
    return pixel_values, image_grid_thw, n_pad


class ModelHolder:
    """持有 tokenizer / image_processor / 学生（raw）/ 教师（可选冻结）。"""

    def __init__(self, backbone: str, device: str, grad_checkpointing: bool,
                 load_teacher: bool):
        self.device = device
        self.amp = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
        self.ip = AutoImageProcessor.from_pretrained(backbone, local_files_only=True)
        self.pad_id = self.tok.pad_token_id
        self.eos_id = self.tok.eos_token_id
        self.vs = self.tok.convert_tokens_to_ids(VISION_START)
        self.vpad = self.tok.convert_tokens_to_ids(IMAGE_PAD)
        self.ve = self.tok.convert_tokens_to_ids(VISION_END)

        print(f"[{os.environ.get('RANK','0')}] loading student (trainable) ...", flush=True)
        self.student = Qwen3_5ForConditionalGeneration.from_pretrained(
            backbone, torch_dtype=self.amp, local_files_only=True).to(device)
        if grad_checkpointing:
            self.student.gradient_checkpointing_enable()
        self.student.train()

        self.teacher = None
        if load_teacher:
            print(f"[{os.environ.get('RANK','0')}] loading teacher (frozen) ...", flush=True)
            self.teacher = Qwen3_5ForConditionalGeneration.from_pretrained(
                backbone, torch_dtype=self.amp, local_files_only=True).to(device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def _img_block(self, n_pad: int):
        return [self.vs] + [self.vpad] * n_pad + [self.ve]

    def _prompt_ids(self, prompt_text: str, img_block):
        return img_block + self.tok.encode(prompt_text, add_special_tokens=False)

    def forward_logits(self, model, input_ids, pixel_values, image_grid_thw):
        mm = (input_ids == self.vpad).long()
        attn = torch.ones_like(input_ids)
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm,
            use_cache=False,
        )
        return out.logits.float()

    @torch.no_grad()
    def sample(self, prompt_ids, pixel_values, image_grid_thw,
               max_new_tokens: int, temperature: float):
        """学生 on-policy 采样，返回轨迹 token id 列表。用 raw student（无 DDP 开销）。"""
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        mm = (input_ids == self.vpad).long()
        gen = self.student.generate(
            input_ids=input_ids, attention_mask=attn,
            pixel_values=pixel_values.to(self.device),
            image_grid_thw=image_grid_thw.to(self.device),
            mm_token_type_ids=mm,
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=self.pad_id, eos_token_id=self.eos_id,
        )
        return gen[0][len(prompt_ids):].tolist()


def sft_loss(holder, model, prompt_text, img_path, curve, recover, max_pixels):
    """纯 hard CE：学生预测黄金标签。model 可为 DDP wrapper 或 raw。"""
    device = holder.device
    pixel_values, image_grid_thw, n_pad = encode_image(img_path, holder.ip, max_pixels)
    pv = pixel_values.to(device)
    gthw = image_grid_thw.to(device)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    Lp = len(prompt_ids)
    gold_ids = holder.tok.encode(gold_text(curve, recover), add_special_tokens=False)
    Lg = len(gold_ids)

    student_ids = torch.tensor([prompt_ids + gold_ids], device=device)
    s_logits = holder.forward_logits(model, student_ids, pv, gthw)
    s_logits = s_logits[0, Lp - 1: Lp - 1 + Lg]  # [Lg, V]
    gold_t = torch.tensor(gold_ids, device=device)
    ce = F.cross_entropy(s_logits, gold_t)
    return ce, {"ce": ce.detach()}, gold_ids


def opsd_loss(holder, model, prompt_text, img_path, curve, recover,
              max_new_tokens, sample_temperature, kl_temperature,
              sft_weight, max_pixels):
    """reverse KL(学生 ∥ 教师) 作用在 on-policy 轨迹 token，可选叠加 hard CE。"""
    device = holder.device
    pixel_values, image_grid_thw, n_pad = encode_image(img_path, holder.ip, max_pixels)
    pv = pixel_values.to(device)
    gthw = image_grid_thw.to(device)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    Lp = len(prompt_ids)

    # 1. 学生 on-policy 采样轨迹
    traj_ids = holder.sample(prompt_ids, pv, gthw, max_new_tokens, sample_temperature)
    Lt = len(traj_ids)
    if Lt == 0:
        traj_ids = holder.tok.encode(gold_text(curve, recover), add_special_tokens=False)
        Lt = len(traj_ids)

    # 2. 教师输入 = prompt + gold + traj；学生输入 = prompt + traj
    gold_ids = holder.tok.encode(gold_text(curve, recover), add_special_tokens=False)
    Lg = len(gold_ids)
    student_ids = torch.tensor([prompt_ids + traj_ids], device=device)
    teacher_ids = torch.tensor([prompt_ids + gold_ids + traj_ids], device=device)

    # 3. 教师 forward（no_grad），轨迹 logits 在 [Lp+Lg-1, Lp+Lg-1+Lt)
    with torch.no_grad():
        t_logits = holder.forward_logits(holder.teacher, teacher_ids, pv, gthw)
        teacher_traj = t_logits[0, Lp + Lg - 1: Lp + Lg - 1 + Lt]

    # 4. 学生 forward（带梯度），轨迹 logits 在 [Lp-1, Lp-1+Lt)
    s_logits = holder.forward_logits(model, student_ids, pv, gthw)
    student_traj = s_logits[0, Lp - 1: Lp - 1 + Lt]

    # 5. reverse KL(学生 ∥ 教师)
    s = student_traj / kl_temperature
    t = teacher_traj / kl_temperature
    log_ps = torch.log_softmax(s, dim=-1)
    log_pt = torch.log_softmax(t, dim=-1)
    ps = log_ps.exp()
    per_token_kl = (ps * (log_ps - log_pt)).sum(-1)
    kl_loss = per_token_kl.mean()

    loss = kl_loss
    info = {"kl": kl_loss.detach()}

    # 6. 可选 hard CE
    if sft_weight > 0:
        sft_ids = torch.tensor([prompt_ids + gold_ids], device=device)
        sft_logits = holder.forward_logits(model, sft_ids, pv, gthw)
        sft_logits = sft_logits[0, Lp - 1: Lp - 1 + Lg]
        gold_t = torch.tensor(gold_ids, device=device)
        ce = F.cross_entropy(sft_logits, gold_t)
        loss = loss + sft_weight * ce
        info["ce"] = ce.detach()

    return loss, info, traj_ids


def parse_tags(text: str):
    m1 = TREND_TAG.search(text)
    m2 = RECOVER_TAG.search(text)
    trend = m1.group(1).strip() if m1 else ""
    recover = m2.group(1).strip() if m2 else ""
    return trend, recover


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


def record_batches(records, grad_accum, seed, rank, world):
    """每 epoch 洗牌后按 rank 分片，yield 每步 grad_accum 个记录的列表。"""
    rng = random.Random(seed)
    order = list(range(len(records)))
    while True:
        rng.shuffle(order)
        usable = (len(order) // (grad_accum * world)) * grad_accum * world
        order = order[:usable]
        per_rank = usable // world
        mine = order[rank * per_rank:(rank + 1) * per_rank]
        for start in range(0, len(mine) - grad_accum + 1, grad_accum):
            yield [records[i] for i in mine[start:start + grad_accum]]


@torch.no_grad()
def evaluate(holder, raw_student, records, img_path, prompt_text, max_new_tokens,
             eval_samples, seed, rank, world, distm):
    """dev 集：学生贪心采样 → 解析标签 → joint 准确率；DDP 下按 rank 分片 + all_reduce。"""
    raw_student.eval()
    rng = random.Random(seed)
    subset = rng.sample(records, min(eval_samples, len(records)))
    correct, total = 0, 0
    for rec in subset:
        pixel_values, image_grid_thw, n_pad = encode_image(img_path(rec), holder.ip, DEFAULT_MAX_PIXELS)
        prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
        input_ids = torch.tensor([prompt_ids], device=holder.device)
        attn = torch.ones_like(input_ids)
        mm = (input_ids == holder.vpad).long()
        gen = raw_student.generate(
            input_ids=input_ids, attention_mask=attn,
            pixel_values=pixel_values.to(holder.device),
            image_grid_thw=image_grid_thw.to(holder.device),
            mm_token_type_ids=mm,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=holder.pad_id, eos_token_id=holder.eos_id,
        )
        out_text = holder.tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
        trend, recover = parse_tags(out_text)
        total += 1
        if trend == rec["curve"] and recover == rec["recover"]:
            correct += 1
    raw_student.train()
    if world > 1:
        t = torch.tensor([float(correct), float(total)], device=holder.device)
        distm.all_reduce(t)
        correct, total = int(t[0].item()), int(t[1].item())
    return correct / max(1, total)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "opsd"], required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", default=None)
    ap.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--output-dir", default="outputs/opsd")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--sample-temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--sft-weight", type=float, default=0.0,
                    help="opsd 模式下叠加 hard CE 权重；0=纯 reverse KL")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=200)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    rank, world, local_rank, dist = init_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if world > 1 else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    with open(args.train_file, encoding="utf-8") as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]
    if is_main:
        print(f"[{args.mode}] train records: {len(records)}", flush=True)

    dev_records = None
    if args.dev_file and os.path.isfile(args.dev_file):
        with open(args.dev_file, encoding="utf-8") as f:
            dev_records = json.load(f)
        if args.limit:
            dev_records = dev_records[: args.limit]

    load_teacher = (args.mode == "opsd")
    holder = ModelHolder(args.backbone, device, args.grad_checkpointing, load_teacher)

    model = holder.student
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    raw_student = holder.student

    optimizer = torch.optim.AdamW(raw_student.parameters(),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    steps, warmup = args.max_steps, args.warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, warmup))
        * 0.5 * (1 + math.cos(math.pi * max(0, s - warmup) / max(1, steps - warmup))),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    def img_path(rec):
        return os.path.join(args.files_root, rec["from"], rec["image"])

    batches = record_batches(records, args.grad_accum, args.seed, rank, world)
    start = time.monotonic()
    best_acc, best_step = -1.0, 0

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        agg = {}
        recs = next(batches)
        for rec in recs:
            if args.mode == "sft":
                loss, info, _ = sft_loss(holder, model, prompt_text, img_path(rec),
                                         rec["curve"], rec["recover"], args.max_pixels)
            else:
                loss, info, _ = opsd_loss(holder, model, prompt_text, img_path(rec),
                                          rec["curve"], rec["recover"],
                                          args.max_new_tokens, args.sample_temperature,
                                          args.kl_temperature, args.sft_weight,
                                          args.max_pixels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}")
            (loss / args.grad_accum).backward()
            total_loss += loss.detach().item() / args.grad_accum
            for k, v in info.items():
                agg.setdefault(k, 0.0)
                agg[k] += v.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_student.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"nonfinite grad at step {step}")
        optimizer.step()
        scheduler.step()

        record = {"step": step, "loss": round(total_loss, 4),
                  "seconds": round(time.monotonic() - start, 1)}
        for k, v in agg.items():
            record[k] = round(v, 4)

        if dev_records is not None and step % args.eval_every == 0:
            acc = evaluate(holder, raw_student, dev_records, img_path, prompt_text,
                           args.max_new_tokens, args.eval_samples, args.seed + step,
                           rank, world, dist)
            if is_main:
                record["dev_acc"] = round(acc, 4)
                if acc > best_acc:
                    best_acc, best_step = acc, step
                    torch.save({"state_dict": raw_student.state_dict(),
                                "step": step, "mode": args.mode,
                                "backbone": args.backbone},
                               os.path.join(args.output_dir, "best.pt"))

        if is_main:
            print(record, flush=True)

    if is_main:
        print(f"done. mode={args.mode} best_step={best_step} best_dev_acc={best_acc:.4f}",
              flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
