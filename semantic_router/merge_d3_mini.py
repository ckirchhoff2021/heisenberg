#!/usr/bin/env python3
"""Merge a d3-mini LoRA adapter back into its backbone.

The output is a standalone checkpoint with the same structure as d3-mini.
The fine-tuned ``readout.safetensors`` from the adapter directory is copied
next to the merged backbone weights.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a fine-tuned d3-mini LoRA adapter into its backbone."
    )
    parser.add_argument(
        "--d3-dir",
        default=os.environ.get("D3_DIR", "checkpoints/d3-mini"),
        help="base d3-mini checkpoint/runtime directory",
    )
    parser.add_argument(
        "--adapter-dir",
        default=os.environ.get(
            "ADAPTER_DIR", str(REPO_ROOT / "outputs/d3_mini_finetuned")
        ),
        help="LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "outputs/jev-4B-aiops"),
        help="directory for the merged standalone checkpoint",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("MERGE_DEVICE", "cuda:0"),
        help="torch device used while loading and merging",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--skip-forward-check",
        action="store_true",
        help="skip reloading the saved checkpoint for a forward smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    print(f"[1/5] Loading base backbone: {args.d3_dir} "
          f"(device={args.device})", flush=True)
    backbone = Qwen3_5Model.from_pretrained(
        args.d3_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    )

    print("[2/5] Loading LoRA adapter and merging", flush=True)
    language_model = PeftModel.from_pretrained(
        backbone.language_model, args.adapter_dir
    )
    backbone.language_model = language_model.merge_and_unload()

    param_count = sum(p.numel() for p in backbone.parameters())
    print(f"  merged backbone params = {param_count / 1e9:.3f}B", flush=True)

    print(f"[3/5] Saving merged checkpoint -> {args.output_dir}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    backbone.save_pretrained(
        args.output_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=True,
    )

    readout_src = Path(args.adapter_dir) / "readout.safetensors"
    if readout_src.exists():
        print("[4/5] Copying fine-tuned readout", flush=True)
        shutil.copy2(readout_src, Path(args.output_dir) / "readout.safetensors")
    else:
        print("[4/5] No fine-tuned readout.safetensors found; skipping",
              flush=True)

    if args.skip_forward_check:
        print("[5/5] Forward check skipped", flush=True)
        print("DONE", flush=True)
        return

    print("[5/5] Reloading output and running forward check", flush=True)
    del backbone, language_model
    torch.cuda.empty_cache()
    reloaded = Qwen3_5Model.from_pretrained(
        args.output_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    token_ids = torch.tensor([[32, 33, 34, 35]], device=args.device)
    embeds = reloaded.get_input_embeddings()(token_ids)
    result = reloaded.language_model(inputs_embeds=embeds)
    last_hidden = result.last_hidden_state[:, -1].float()
    print(f"  forward OK, last hidden shape={tuple(last_hidden.shape)}",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
