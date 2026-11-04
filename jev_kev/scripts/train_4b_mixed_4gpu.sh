#!/bin/bash
# jev_kev 多模态混合数据集（jev-mixed.json）— 4B backbone LoRA 4 卡 DDP 训练
# backbone: Qwen3.5-4B（多模态，4573M 参数）
# 数据: datas/aiops/noise/jev-mixed.json（4 任务：legacy trend/recover + noise argos/kepler）
# 用法: bash jev_kev/scripts/train_4b_mixed_4gpu.sh
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

BACKBONE="checkpoints/Qwen3.5-4B"
MIXED_FILE="datas/aiops/noise/jev-mixed.json"
OUTPUT_DIR="outputs/jev_4b_mixed"

# LoRA（rank 16）只训 language_model 的 LoRA adapter + PointerHead（33.8M 可训练），
# 冻结 vision tower。lr 对齐 0.8B LoRA 经验值 1e-4（AdamW scale-invariant）。
# state 最长 legacy_argos ~2766 token，max-state 3072 足够。
# 全局有效 batch = batch(1) × world(4) × accum(8) = 32 record/step
.venv/bin/torchrun --standalone --nproc_per_node=4 jev_kev/source/train_argos.py \
  --backbone "$BACKBONE" \
  --mixed-file "$MIXED_FILE" \
  --files-root "datas/aiops/files" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --batch 1 \
  --accum 8 \
  --lr 1e-4 \
  --head-lr 1e-4 \
  --lora 16 \
  --max-state 3072 \
  --max-pixels 262144 \
  --eval-every 100
