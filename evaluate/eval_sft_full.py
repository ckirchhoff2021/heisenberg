"""全参数 SFT 模型验证集评估脚本。

口径与 train/opsd.py --mode sft 完全一致：
  - 手动编码（img_block + prompt_text，无 chat_template）
  - noreason prompt（argos-dp-prompt-noreason.md）
  - 贪心解码轨迹 <trend>..</trend>\n<recover>..</recover>
  - best.pt 是 state_dict 格式，需 from_pretrained(backbone) 重建结构再 load_state_dict

输出：curve / recover / joint 准确率 + 分类报告 + 混淆矩阵。

用法（单卡，8 卡现全空闲）：
  .venv/bin/python evaluate/eval_sft_full.py \
      --checkpoint outputs/sft_full_4gpu/best.pt \
      --valid-data-path datas/aiops/files/jsons/argos-valids.json \
      --prompt-file datas/aiops/files/prompts/argos-dp-prompt-noreason.md \
      --files-root datas/aiops/files \
      --output-file outputs/sft_full_4gpu/eval_metrics.json
"""
import argparse
import json
import os
import re
import sys

import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from train.opsd import ModelHolder, encode_image, parse_tags  # 复用训练编码逻辑

CURVE_LABELS = ["normal", "offset", "increase", "decline", "spike", "fluctuate", "other"]
RECOVER_LABELS = ["recovered", "recovering", "no_recover", "other"]


def load_checkpoint(ckpt_path: str, device: str, grad_checkpointing: bool = False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = ck["backbone"]
    holder = ModelHolder(backbone, device, grad_checkpointing, load_teacher=False)
    missing, unexpected = holder.student.load_state_dict(ck["state_dict"], strict=True)
    if missing or unexpected:
        print(f"[WARN] missing={len(missing)} unexpected={len(unexpected)}")
    holder.student.eval()
    return holder, ck


def evaluate(holder, records, files_root, prompt_text, max_new_tokens):
    preds_curve, preds_recover = [], []
    gts_curve, gts_recover = [], []
    invalid = 0

    for rec in tqdm(records, desc="Evaluating"):
        img_path = os.path.join(files_root, rec["from"], rec["image"])
        pixel_values, image_grid_thw, n_pad = encode_image(
            img_path, holder.ip, 512 * 512)
        prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
        input_ids = torch.tensor([prompt_ids], device=holder.device)
        attn = torch.ones_like(input_ids)
        mm = (input_ids == holder.vpad).long()
        with torch.no_grad():
            gen = holder.student.generate(
                input_ids=input_ids, attention_mask=attn,
                pixel_values=pixel_values.to(holder.device),
                image_grid_thw=image_grid_thw.to(holder.device),
                mm_token_type_ids=mm,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=holder.pad_id, eos_token_id=holder.eos_id,
            )
        out_text = holder.tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
        trend, recover = parse_tags(out_text)
        if not trend or not recover:
            invalid += 1
            trend = trend or "other"
            recover = recover or "other"

        preds_curve.append(trend)
        preds_recover.append(recover)
        gts_curve.append(rec["curve"])
        gts_recover.append(rec["recover"])

    return gts_curve, preds_curve, gts_recover, preds_recover, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid-data-path", default="datas/aiops/files/jsons/argos-valids.json")
    ap.add_argument("--prompt-file", default="datas/aiops/files/prompts/argos-dp-prompt-noreason.md")
    ap.add_argument("--files-root", default="datas/aiops/files")
    ap.add_argument("--output-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    records = json.load(open(args.valid_data_path, encoding="utf-8"))
    print(f"验证集记录数: {len(records)}")

    holder, ck = load_checkpoint(args.checkpoint, args.device)
    print(f"checkpoint step={ck.get('step')} mode={ck.get('mode')} backbone={ck['backbone']}")

    gts_curve, preds_curve, gts_recover, preds_recover, invalid = evaluate(
        holder, records, args.files_root, prompt_text, args.max_new_tokens)

    invalid_rate = invalid / len(records)
    metrics = {}
    metrics["invalid_count"] = invalid
    metrics["invalid_rate"] = invalid_rate
    metrics["curve_accuracy"] = accuracy_score(gts_curve, preds_curve)
    metrics["recover_accuracy"] = accuracy_score(gts_recover, preds_recover)
    metrics["joint_accuracy"] = float(np.mean([
        1 if c == cp and r == rp else 0
        for c, cp, r, rp in zip(gts_curve, preds_curve, gts_recover, preds_recover)
    ]))

    print(f"\n无效输出率: {invalid_rate:.4f} ({invalid}/{len(records)})")
    print(f"\n===== Curve Trend (7类) ===== accuracy={metrics['curve_accuracy']:.4f}")
    print(classification_report(gts_curve, preds_curve, labels=CURVE_LABELS,
                                target_names=CURVE_LABELS, zero_division=0))
    metrics["curve_report"] = classification_report(
        gts_curve, preds_curve, labels=CURVE_LABELS, target_names=CURVE_LABELS,
        zero_division=0, output_dict=True)
    metrics["curve_confusion"] = confusion_matrix(
        gts_curve, preds_curve, labels=CURVE_LABELS).tolist()

    print(f"\n===== Recover State (4类) ===== accuracy={metrics['recover_accuracy']:.4f}")
    print(classification_report(gts_recover, preds_recover, labels=RECOVER_LABELS,
                                target_names=RECOVER_LABELS, zero_division=0))
    metrics["recover_report"] = classification_report(
        gts_recover, preds_recover, labels=RECOVER_LABELS, target_names=RECOVER_LABELS,
        zero_division=0, output_dict=True)
    metrics["recover_confusion"] = confusion_matrix(
        gts_recover, preds_recover, labels=RECOVER_LABELS).tolist()

    print(f"\n===== Joint ===== joint_accuracy={metrics['joint_accuracy']:.4f}")
    print(f"\ncurve 混淆矩阵 (行=GT, 列=pred):\n{np.array(metrics['curve_confusion'])}")
    print(f"recover 混淆矩阵 (行=GT, 列=pred):\n{np.array(metrics['recover_confusion'])}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n指标已保存: {args.output_file}")


if __name__ == "__main__":
    main()
