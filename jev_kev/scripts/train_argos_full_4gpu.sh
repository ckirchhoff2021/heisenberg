#!/bin/bash
# jev_kev 多模态 argos 曲线分类 — 全量微调（无 LoRA）4 卡 DDP 训练启动脚本
# 对比 jev（置换等变头，全量微调）：backbone lr 1e-5 / head lr 1e-4
# 用法: bash jev_kev/scripts/train_argos_full_4gpu.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 离线加载：backbone 用绝对路径，禁 HF hub 访问
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 干净的分布式 rendezvous
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE

BACKBONE="checkpoints/Qwen3.5-0.8B"
TRAIN_FILE="datas/aiops/files/jsons/argos-trains.json"
DEV_FILE="datas/aiops/files/jsons/argos-valids.json"
FILES_ROOT="datas/aiops/files"
PROMPT_FILE="datas/aiops/files/prompts/argos-jev-prompt.md"
OUTPUT_DIR="outputs/jev_kev_argos_full"

# 全量微调：无 LoRA（lora 0），backbone 含 vision tower 全参训练
# lr 对齐 jev：backbone 1e-5 / head 1e-4
# 全量显存更大，batch=1 + accum=8（全局有效 batch = 1×4×8 = 32 record/step）
.venv/bin/torchrun --standalone --nproc_per_node=4 jev_kev/source/train_argos.py \
  --backbone "$BACKBONE" \
  --train-file "$TRAIN_FILE" \
  --dev-file "$DEV_FILE" \
  --files-root "$FILES_ROOT" \
  --prompt-file "$PROMPT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --batch 1 \
  --accum 8 \
  --lr 1e-5 \
  --head-lr 1e-4 \
  --lora 0 \
  --max-state 3072 \
  --max-pixels 262144 \
  --eval-every 100
