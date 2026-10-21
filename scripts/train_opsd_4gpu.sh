#!/bin/bash
# B 组：OPSD 全参数 + 特权蒸馏（reverse KL 过程监督）
# 教师 = 同架构 base 冻结；特权信息 = 黄金标签注入教师；学生 on-policy 采样
# GPU 4,5,6,7
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
# /tmp/fla_block 提供空 fla 包，强制 transformers 走 torch 参考 GDN kernel（fla chunk kernel 在 A40 上显存 OOM）
export PYTHONPATH="/tmp/fla_block:$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

BACKBONE="checkpoints/Qwen3.5-0.8B"
TRAIN_FILE="datas/aiops/files/jsons/argos-trains.json"
DEV_FILE="datas/aiops/files/jsons/argos-valids.json"
OUTPUT_DIR="outputs/opsd_4gpu"

# 纯 reverse KL（sft-weight 0）；有效 batch = 1 × 4卡 × 8 accum = 32 图/step
.venv/bin/torchrun --standalone --nproc_per_node=4 train/opsd.py \
  --mode opsd \
  --backbone "$BACKBONE" \
  --train-file "$TRAIN_FILE" \
  --dev-file "$DEV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --grad-accum 8 \
  --learning-rate 1e-5 \
  --kl-temperature 1.0 \
  --sample-temperature 1.0 \
  --sft-weight 0.0 \
  --eval-every 100 \
  --eval-samples 200 \
  --max-new-tokens 64 \
  --grad-checkpointing
