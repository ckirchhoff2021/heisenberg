#!/bin/bash
# JEV 多模态判别模型 — 4 卡 DDP 训练启动脚本
# 用法: bash scripts/train_jev_4gpu.sh
# 说明: 本机只有 0-7 号卡（无 8 号卡），全部空闲；用 4,5,6,7 四张卡。
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
# /tmp/fla_block 提供空 fla 包，强制 transformers 走 torch 参考 GDN kernel（fla chunk kernel 在 A40 上显存 OOM）
export PYTHONPATH="/tmp/fla_block:$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --------------------------
# 数据 / 模型配置
# --------------------------
BACKBONE="checkpoints/Qwen3.5-0.8B"
TRAIN_FILE="datas/aiops/files/jsons/argos-trains.json"
DEV_FILE="datas/aiops/files/jsons/argos-valids.json"
OUTPUT_DIR="outputs/jev_4gpu"

# A40 46GB 显存下大 batch（56 路径）会 OOM：microbatch 降到 2；
# 全局有效 batch = microbatch(2) × world(4) × grad_accum(8) = 64 题/step
.venv/bin/torchrun --standalone --nproc_per_node=4 train/jev.py \
  --backbone "$BACKBONE" \
  --train-file "$TRAIN_FILE" \
  --dev-file "$DEV_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --microbatch 2 \
  --grad-accum 8 \
  --eval-every 100
