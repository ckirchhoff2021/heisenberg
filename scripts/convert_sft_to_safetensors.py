"""将全参数 SFT 的 best.pt（state_dict）转换为可部署的标准 HF 模型目录。

产物：outputs/sft_full_4gpu/safetensors_model/
  - model.safetensors（safe_serialization=True）
  - config.json / generation_config.json
  - tokenizer 全套（tokenizer.json / vocab / merges / tokenizer_config）
  - image processor（preprocessor_config.json 等）
  - chat_template.jinja

注意：
  - best.pt 是裸 state_dict（474 keys，含 lm_head，无 mtp 层）
  - 与 from_pretrained(backbone) 重建的模型 state_dict 严格一致（strict=True）
  - 转换后可直接 vLLM serve，也可 transformers pipeline 加载
"""
import argparse
import json
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import Qwen3_5ForConditionalGeneration

# backbone 目录里需要一并拷贝的配套文件
COPY_FILES = [
    "config.json",
    "generation_config.json",  # 若存在
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",  # 若存在
]


def copy_support_files(backbone_dir: str, out_dir: str):
    """拷贝 tokenizer / image processor / config 等配套文件（跳过权重文件）。"""
    copied, skipped = [], []
    for name in COPY_FILES:
        src = os.path.join(backbone_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied.append(name)
        else:
            skipped.append(name)
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/sft_full_4gpu/best.pt")
    ap.add_argument("--output-dir", default="outputs/sft_full_4gpu/safetensors_model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 1. 加载 best.pt
    print(f"加载 checkpoint: {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = ck["backbone"]
    state_dict = ck["state_dict"]
    print(f"  step={ck.get('step')} mode={ck.get('mode')} backbone={backbone}")
    print(f"  state_dict keys={len(state_dict)}")

    # 2. 重建模型结构 + strict load
    print(f"从 backbone 重建模型结构: {backbone}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        backbone, torch_dtype=dtype, local_files_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"  load_state_dict(strict=True): missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, f"权重不匹配: missing={missing} unexpected={unexpected}"
    model.eval()

    # 3. 导出 safetensors
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"导出 safetensors 到: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    print("  model.safetensors 已写入")

    # 4. 拷贝配套文件
    copied, skipped = copy_support_files(backbone, args.output_dir)
    print(f"  已拷贝配套文件: {copied}")
    if skipped:
        print(f"  backbone 中不存在（跳过）: {skipped}")

    # 5. 写转换元信息
    meta = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_backbone": backbone,
        "step": ck.get("step"),
        "mode": ck.get("mode"),
        "dtype": args.dtype,
    }
    with open(os.path.join(args.output_dir, "conversion_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 转换完成: {args.output_dir}")
    print(f"   文件数: {len(os.listdir(args.output_dir))}")


if __name__ == "__main__":
    main()
