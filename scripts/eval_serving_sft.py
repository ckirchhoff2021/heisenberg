"""SFT 全参数模型的 vLLM serving 验证集评测（对齐训练口径）。

口径与 train/opsd.py --mode sft 一致：
  - noreason prompt（argos-dp-prompt-noreason.md）
  - 输出 <trend>..</trend>\n<recover>..</recover>
  - 贪心解码（temperature=0）

但走 vLLM OpenAI chat 接口（chat_template 自动包 role 标记），
与 eval_sft_full.py 的手动编码口径存在 token 布局差异，精度可能有偏差。
本脚本用于对比 serving 口径下的精度，基线为 eval_sft_full.py 的 71.75%。
"""
import argparse
import base64
import json
import os
import re

import numpy as np
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

CURVE_LABELS = ["normal", "offset", "increase", "decline", "spike", "fluctuate", "other"]
RECOVER_LABELS = ["recovered", "recovering", "no_recover", "other"]

RESULT_TAG_PATTERN = r"<trend>(.*?)</trend>"
RECOVER_TAG_PATTERN = r"<recover>(.*?)</recover>"


def extract_tag(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_tags(text: str):
    trend = extract_tag(text, RESULT_TAG_PATTERN)
    recover = extract_tag(text, RECOVER_TAG_PATTERN)
    return trend, recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8007/v1")
    ap.add_argument("--model-name", default="sft-full")
    ap.add_argument("--valid-data-path", default="datas/aiops/files/jsons/argos-valids.json")
    ap.add_argument("--prompt-file", default="datas/aiops/files/prompts/argos-dp-prompt-noreason.md")
    ap.add_argument("--files-root", default="datas/aiops/files")
    ap.add_argument("--output-file", default="outputs/sft_full_4gpu/eval_metrics_serving.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    records = json.load(open(args.valid_data_path, encoding="utf-8"))
    print(f"验证集记录数: {len(records)}")

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    gts_curve, gts_recover = [], []
    preds_curve, preds_recover = [], []
    invalid = 0

    for rec in tqdm(records, desc="Serving eval"):
        img_path = os.path.join(args.files_root, rec["from"], rec["image"])
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        resp = client.chat.completions.create(
            model=args.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        out = resp.choices[0].message.content or ""
        trend, recover = parse_tags(out)
        if not trend or not recover:
            invalid += 1
            trend = trend or "other"
            recover = recover or "other"

        preds_curve.append(trend)
        preds_recover.append(recover)
        gts_curve.append(rec["curve"])
        gts_recover.append(rec["recover"])

    metrics = {
        "invalid_count": invalid,
        "invalid_rate": invalid / len(records),
        "curve_accuracy": accuracy_score(gts_curve, preds_curve),
        "recover_accuracy": accuracy_score(gts_recover, preds_recover),
        "joint_accuracy": float(np.mean([
            1 if c == cp and r == rp else 0
            for c, cp, r, rp in zip(gts_curve, preds_curve, gts_recover, preds_recover)
        ])),
    }
    metrics["curve_report"] = classification_report(
        gts_curve, preds_curve, labels=CURVE_LABELS, target_names=CURVE_LABELS,
        zero_division=0, output_dict=True)
    metrics["recover_report"] = classification_report(
        gts_recover, preds_recover, labels=RECOVER_LABELS, target_names=RECOVER_LABELS,
        zero_division=0, output_dict=True)
    metrics["curve_confusion"] = confusion_matrix(
        gts_curve, preds_curve, labels=CURVE_LABELS).tolist()
    metrics["recover_confusion"] = confusion_matrix(
        gts_recover, preds_recover, labels=RECOVER_LABELS).tolist()

    print(f"\n无效输出: {invalid}/{len(records)} ({metrics['invalid_rate']:.4f})")
    print(f"curve_accuracy: {metrics['curve_accuracy']:.4f}")
    print(f"recover_accuracy: {metrics['recover_accuracy']:.4f}")
    print(f"joint_accuracy: {metrics['joint_accuracy']:.4f}")
    print(f"\nrecover 混淆矩阵 (行=GT, 列=pred):\n{np.array(metrics['recover_confusion'])}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n指标已保存: {args.output_file}")


if __name__ == "__main__":
    main()
