"""JEV 模型推理模块。

加载 best.pt（判别式多模态 Qwen3.5 + 置换等变候选头），对单张曲线图推理
trend（7 类）+ recover（4 类）两道选择题。

用法：
  1) 作为库：
       from infer.jev_infer import JevInference
       infer = JevInference("outputs/jev_4gpu/best.pt")
       result = infer.predict_image("/abs/path/to/image.png")
       # result = {"trend": {"label": "normal", "prob": 0.93, "probs": {...}},
       #           "recover": {"label": "no_recover", "prob": 0.81, "probs": {...}}}

  2) 命令行：
       python infer/jev_infer.py /abs/path/to/image.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoImageProcessor, AutoTokenizer

from dataset.jev import (
    CURVE_CHOICES,
    CURVE_QUESTION,
    DEFAULT_FILES_ROOT,
    DEFAULT_PROMPT_FILE,
    RECOVER_CHOICES,
    RECOVER_QUESTION,
    JevTorchDataset,
    load_rule_prompt,
)
from models.jev import JevModel


class JevInference:
    """加载 JEV checkpoint 并提供单图推理。"""

    def __init__(
        self,
        checkpoint: str,
        device: str | None = None,
        prompt_file: str | None = DEFAULT_PROMPT_FILE,
        files_root: str = DEFAULT_FILES_ROOT,
    ):
        """
        Args:
            checkpoint: best.pt 路径（含 state_dict/backbone/max_pixels/max_tokens）
            device: cuda 设备，默认自动（cuda 优先，无则 cpu）
            prompt_file: 分类规则 prompt；传 None 则不注入 system
            files_root: argos 风格相对路径的图像根目录
        """
        self.files_root = files_root
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone = ck["backbone"]
        self.max_pixels = int(ck.get("max_pixels", 512 * 512))
        self.max_tokens = int(ck.get("max_tokens", 3072))
        self.step = ck.get("step")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = torch.bfloat16
        self._last_mask: torch.Tensor | None = None

        print(f"[JEV] backbone={self.backbone}", flush=True)
        print(f"[JEV] checkpoint step={self.step} max_pixels={self.max_pixels} "
              f"max_tokens={self.max_tokens}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(self.backbone, local_files_only=True)

        print(f"[JEV] loading model -> {self.device} ...", flush=True)
        self.model = JevModel(self.backbone, dtype=self.amp_dtype).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

        # 分类规则 system：与训练注入保持一致（load_rule_prompt 去掉生成式"输出格式"）
        self.system = ""
        if prompt_file:
            prompt_path = prompt_file if os.path.isabs(prompt_file) else os.path.join(REPO_ROOT, prompt_file)
            if os.path.isfile(prompt_path):
                self.system = load_rule_prompt(prompt_path)
                print(f"[JEV] system prompt: {prompt_path} ({len(self.system)} chars)", flush=True)
            else:
                print(f"[JEV] WARNING: prompt file not found: {prompt_path}, no system injected", flush=True)

    def _resolve_path(self, image_path: str) -> str:
        """支持绝对路径；相对路径按 argos 风格 (files_root/subdir/name) 解析。"""
        if os.path.isabs(image_path):
            return image_path
        if os.path.isfile(image_path):
            return image_path
        cand = os.path.join(self.files_root, image_path)
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(
            f"image not found: {image_path} (tried files_root={self.files_root})")

    @torch.inference_mode()
    def _forward_questions(self, questions, choices_list, image_paths) -> torch.Tensor:
        """把多个 (question, choices, image) 题目一次编码前向，返回每题 softmax 概率 [Bq, Cmax]。"""
        rows = [
            {
                "system": self.system,
                "state_text": "",
                "state_images": [p],
                "question": q,
                "choices": c,
                "answer": 0,  # 占位，推理不用 answer
            }
            for q, c, p in zip(questions, choices_list, image_paths)
        ]
        ds = JevTorchDataset(
            rows, self.tokenizer, self.image_processor,
            max_tokens=self.max_tokens, max_pixels=self.max_pixels,
        )
        batch = ds.collate([ds[i] for i in range(len(ds))])
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        self._last_mask = batch["cand_mask"].cpu()
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            logits = self.model(batch)["logits"].float()
        mask = batch["cand_mask"]
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _answer(probs_row: torch.Tensor, choices, mask_row) -> dict:
        """单行概率 → {label, index, prob, probs}。"""
        n = int(mask_row.sum())
        p = probs_row[:n]
        best = int(p.argmax(-1))
        return {
            "label": choices[best],
            "index": best,
            "prob": float(p[best]),
            "probs": {choices[i]: float(p[i]) for i in range(n)},
        }

    def predict_image(self, image_path: str) -> dict:
        """对单张曲线图推理 trend + recover。

        Args:
            image_path: 绝对路径，或相对 files_root 的 argos 风格路径
        Returns:
            {"trend": {...}, "recover": {...}}
        """
        path = self._resolve_path(image_path)
        questions = [CURVE_QUESTION, RECOVER_QUESTION]
        choices = [CURVE_CHOICES, RECOVER_CHOICES]
        probs = self._forward_questions(questions, choices, [path, path])
        mask = self._last_mask
        return {
            "trend": self._answer(probs[0], CURVE_CHOICES, mask[0]),
            "recover": self._answer(probs[1], RECOVER_CHOICES, mask[1]),
        }


def main():
    ap = argparse.ArgumentParser(description="JEV 单图推理 CLI")
    ap.add_argument("image", help="图片路径（绝对路径，或相对 files_root 的 argos 风格）")
    ap.add_argument("--checkpoint", default=os.path.join(REPO_ROOT, "outputs/jev_4gpu/best.pt"))
    ap.add_argument("--device", default=None, help="cuda 设备，默认自动")
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    ap.add_argument("--files-root", default=DEFAULT_FILES_ROOT)
    ap.add_argument("--json", action="store_true", help="输出 JSON（默认人类可读）")
    args = ap.parse_args()

    infer = JevInference(args.checkpoint, device=args.device,
                         prompt_file=args.prompt_file, files_root=args.files_root)
    result = infer.predict_image(args.image)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key in ("trend", "recover"):
            r = result[key]
            print(f"[{key}] {r['label']}  (prob={r['prob']:.3f})")
            for label, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
                print(f"    {label:<12} {p:.4f}")


if __name__ == "__main__":
    main()
