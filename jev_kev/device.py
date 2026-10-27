"""加速器选择：cuda -> mps -> cpu（自包含，无 kev 依赖）。"""
from __future__ import annotations

import torch


def select(device: str = "auto", *, dtype=None):
    """解析设备字符串。

    auto：依次尝试 CUDA、MPS，均不可用则回退 CPU；
    显式指定 cuda/mps/cpu 时直接返回；其他值报错。
    """
    del dtype
    if device == "cpu":
        return "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device in ("cuda", "mps"):
        return device
    raise ValueError(f"unsupported device {device!r}; choose auto, cuda, mps or cpu")
