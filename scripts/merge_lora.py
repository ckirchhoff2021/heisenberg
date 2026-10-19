#!/usr/bin/env python3
"""Merge SFT LoRA adapter into base Qwen3.5-0.8B and save as a standalone model.

Output structure mirrors /tmp/merged_8x08b_ckpt1218 (model.safetensors + configs),
so it can be loaded directly by vLLM `serve` or `grpo_eval_vllm.py --mode local`.
"""
import argparse
import shutil
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

# Qwen3.5-VL 实际模型类（与 grpo_eval.py 保持一致）
try:
    from transformers import Qwen3_5ForConditionalGeneration as ModelCls
except ImportError:
    ModelCls = AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="checkpoints/Qwen3.5-0.8B")
    ap.add_argument("--adapter", default="outputs/sft_08b_4gpu_v1")
    ap.add_argument("--output", default="/tmp/merged_sft_08b_4gpu_v1")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model from {args.base_model} ...")
    model = ModelCls.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging adapter into base weights ...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model to {args.output} ...")
    model.save_pretrained(args.output, safe_serialization=True)

    # 处理器/分词器/chat template 从 adapter 目录复制（训练时已保存对齐的版本）
    src_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
        "generation_config.json",
    ]
    for fn in src_files:
        s = os.path.join(args.adapter, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.output, fn))
            print(f"  copied {fn}")
        else:
            print(f"  [skip] {fn} not in adapter dir")

    print(f"\nDone. Merged model at {args.output}")
    print(f"Files: {sorted(os.listdir(args.output))}")


if __name__ == "__main__":
    main()
