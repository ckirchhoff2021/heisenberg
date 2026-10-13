#!/bin/bash
# GRPO Training Launch Script
# Modify the parameters below as needed

# --------------------------
# Training Configuration
# --------------------------
# Model settings
MODEL_CHECKPOINT="checkpoints/Qwen3.5-0.8B"
# MODEL_CHECKPOINT="checkpoints/Qwen3.5-4B"  # Uncomment for 4B model
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Data settings
TRAIN_DATA_PATH="datas/aiops/files/jsons/argos-trains.json"
FILES_ROOT="datas/aiops/files"          # 图片根目录，按训练机实际路径修改
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt.md"

# Output settings
OUTPUT_DIR="outputs/grpo_vllm"

# Training hyperparameters
BATCH_SIZE=4
EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
LEARNING_RATE=5e-6
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=1024

# Precision settings (H20/A100 use BF16 by default; set FP16="--fp16" to switch)
FP16=""

# Other settings
USE_VLLM="--use-vllm"
VLLM_MODE="server"
VLLM_SERVER_BASE_URL="http://localhost:8010"
VLLM_GROUP_PORT=51216
REPORT_TO="tensorboard"

# Runtime settings (local machine)
# 训练进程在 GPU1，vLLM server 在 GPU2（端口 8010，VLLM_SERVER_DEV_MODE=1）
PYTHON=".venv/bin/python"
export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --------------------------
# Run Training
# --------------------------
echo "Starting GRPO training with the following configuration:"
echo "Model: $MODEL_CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Python: $PYTHON"
echo "GPU: $CUDA_VISIBLE_DEVICES"

$PYTHON train/grpo.py \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --files-root "$FILES_ROOT" \
  --prompt-file "$PROMPT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-batch-size "$BATCH_SIZE" \
  --num-train-epochs "$EPOCHS" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-generations "$NUM_GENERATIONS" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  $FP16 \
  $USE_VLLM \
  --vllm-mode "$VLLM_MODE" \
  --vllm-server-base-url "$VLLM_SERVER_BASE_URL" \
  --vllm-group-port "$VLLM_GROUP_PORT" \
  --report-to "$REPORT_TO"
