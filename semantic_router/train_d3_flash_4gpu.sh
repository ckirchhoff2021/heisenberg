#!/bin/bash
# d3-flash 微调训练启动脚本（4 卡 DDP）
# 用法: bash semantic_router/train_d3_flash_4gpu.sh
#
# 训练对象: d3-flash 的 backbone(LoRA r=16, 冻结 vision) + readout 头(255×4096)
# 数据: datas/aiops/noise/jev-mixed.json (train 11755 / eval 2600)
# 前向: d3 的 noncausal_full_attention (PromptEncoder)
# 输出: outputs/d3_flash_finetuned/ (LoRA adapter + readout.safetensors + meta)
#
# 与 d3-mini 训练同协议（复用 train_d3_mini.py），仅换 d3-dir 与 output-dir。
# d3-flash 为 9B：主干冻结、LoRA+readout 训练，单卡 batch=1 显存约 25GB。

set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE

D3_DIR="${D3_DIR:-checkpoints/d3-flash}"
MIXED_FILE="datas/aiops/noise/jev-mixed.json"
OUTPUT_DIR="outputs/d3_flash_finetuned"

mkdir -p "$OUTPUT_DIR"

# 全局有效 batch = batch(1) × world(4) × accum(8) = 32 record/step
.venv/bin/torchrun --standalone --nproc_per_node=4 semantic_router/train_d3_mini.py \
  --d3-dir "$D3_DIR" \
  --mixed-file "$MIXED_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 500 \
  --batch 1 \
  --accum 8 \
  --lr 1e-4 \
  --readout-lr 1e-4 \
  --lora 16 \
  --freeze-vision 1 \
  --eval-every 100 2>&1 | tee "$OUTPUT_DIR/train.log"
