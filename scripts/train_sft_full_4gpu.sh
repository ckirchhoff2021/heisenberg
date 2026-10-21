#!/bin/bash
# A 组对照：全参数 SFT（生成式 hard CE，无教师/无采样/无蒸馏）
# 目的：单测「解冻 vision tower」能否打破之前 LoRA 冻结 SFT 的坍缩
# GPU 0,1,2,3
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0,1,2,3
# /tmp/fla_block 提供空 fla 包，强制 transformers 走 torch 参考 GDN kernel（fla chunk kernel 在 A40 上显存 OOM）
export PYTHONPATH="/tmp/fla_block:$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BACKBONE="checkpoints/Qwen3.5-0.8B"
TRAIN_FILE="datas/aiops/files/jsons/argos-trains.json"
DEV_FILE="datas/aiops/files/jsons/argos-valids.json"
OUTPUT_DIR="outputs/sft_full_4gpu"

# 全参数 SFT：有效 batch = 1 × 4卡 × 8 accum = 32 图/step，500 步 ≈ 4.9 epoch
.venv/bin/torchrun --standalone --nproc_per_node=4 train/opsd.py \
  --mode sft \
  --backbone "$BACKBONE" \
  --train-file "$TRAIN_FILE" \
  --dev-file "$DEV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --grad-accum 8 \
  --learning-rate 1e-5 \
  --eval-every 100 \
  --eval-samples 200 \
  --max-new-tokens 64 \
  --grad-checkpointing
