"""Checkpoint 保存/加载（自包含，无 kev 依赖）。

支持 jev_kev 训练产出的本地目录布局：
  adapter_config.json + adapter_model.safetensors   # LoRA adapter
  head.pt                                            # Meta：base/head/lora 等元信息

加载行为对齐 kev.checkpoint：本任务保存时 weights_dtype="bf16"，所以在
bf16 base 上以「不合并」方式挂 adapter（kev 对 bf16 LoRA 路径的默认行为），
再载入 PointerHead 权重并切到 eval。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, load_tokenizer


@dataclass
class Meta:
    """head.pt 的内容。KNOWN 之外的键收在 extra 中，读写不丢信息。"""

    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    weights: str = "lora"
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim",
             "weights_dtype", "temperature", "weights")

    @classmethod
    def from_dict(cls, d):
        return cls(
            **{k: d[k] for k in cls.KNOWN if k in d},
            extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}


class Checkpoint:
    def __init__(self, path: str):
        self.path = str(path)
        self.meta = Meta.from_dict(
            torch.load(Path(path) / "head.pt", map_location="cpu"))

    def load(self, device: str):
        """-> (tokenizer, model)，model 为 eval 模式，LoRA 与 PointerHead 均已加载。"""
        meta = self.meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(
            meta.weights_dtype, torch.float32)
        # 先建不带 LoRA 的 base，再从 checkpoint 目录挂 adapter（不 merge）。
        model = DecisionModel(
            meta.base, tok, device, lora=None,
            revision=meta.base_revision, head_dim=meta.head_dim, dtype=dtype)
        from peft import PeftModel
        model.lm = PeftModel.from_pretrained(
            model.lm, self.path, torch_device=str(device)).to(device)
        model.head.load_state_dict(meta.head)
        model.head.temperature = meta.temperature
        model.eval()
        return tok, model


def load(run: str, device: str):
    """便捷函数：Checkpoint(run).load(device)。"""
    return Checkpoint(run).load(device)
