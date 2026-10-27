"""jev_kev：以 Qwen3.5-4B 为 backbone、kev 风格 PointerHead 为分类头的决策模型。

参考 kev（github.com/jaredpalmer/kev）的实现：
- 复用 kev.model 的 DecisionModel / PointerHead / encode（成熟实现，处理 hybrid Qwen3.5
  backbone 的 row-form、DeltaNet cache 等复杂细节）。
- 新增 typed-decisions 软标签数据管线、训练、评测。

架构（与 kev-4b 完全一致）：
  Qwen3.5-4B-Base（32 层，24 Gated DeltaNet + 8 full attention，hidden 2560，frozen）
  + LoRA（rank 16, α 32，作用在 attention/MLP/DeltaNet 投影）
  + PointerHead（dp=256：q(decide token hidden) · k(每个选项 </opt> token hidden) → 每选项标量 logits）
"""
from __future__ import annotations

import torch

from kev.model import DecisionModel, PointerHead, encode, load_tokenizer  # noqa: F401
from kev.checkpoint import LoadOptions  # noqa: F401

# 复用 kev 的模型类与 PointerHead；jev_kev 的增量在数据/训练/评测层。
# DecisionModel(name, tok, device, lora=16, head_dim=256, ...) 即 Qwen3.5-4B + LoRA + PointerHead。
JevKevModel = DecisionModel


def build_model(backbone: str, device: str, lora: int = 16, head_dim: int = 256,
                revision: str | None = None) -> DecisionModel:
    """构建 jev_kev 模型：加载 backbone + 附加 LoRA + 从零初始化的 PointerHead。

    backbone 传 hub id（如 "Qwen/Qwen3.5-4B-Base"），配合 HF_HOME/本地缓存加载。
    """
    tok = load_tokenizer(backbone, revision=revision)
    model = DecisionModel(
        backbone, tok, device, lora=lora, revision=revision, head_dim=head_dim,
        dtype=torch.bfloat16,
    )
    model.lm.config.use_cache = False
    return model
