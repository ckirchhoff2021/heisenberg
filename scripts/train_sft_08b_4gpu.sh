#!/bin/bash
# SFT Training Launch Script — Qwen3.5-0.8B — 4卡数据并行（全监督）
# 与 train_sft_08b.sh（单卡）的区别：
#   - 4 卡 torchrun DDP（纯数据并行，无 vLLM）
#   - 纯 cross-entropy 监督，直接逼近黄金标签，无采样、无 reward hacking 风险
#   - 数据级调整见 dataset/sft.py：有 output 借 <reason>（标签以 JSON 字段为准），
#     无 output 用换行占位 reason + 拼接 curve/recover

# --------------------------
# Training Configuration
# --------------------------
MODEL_CHECKPOINT="checkpoints/Qwen3.5-0.8B"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

TRAIN_DATA_PATH="datas/aiops/files/jsons/argos-trains.json"
FILES_ROOT="datas/aiops/files"
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt.md"

OUTPUT_DIR="outputs/sft_08b_4gpu_v1"

# 4 卡数据并行：per_device_batch_size 是「每卡」的 batch
# effective batch = 4 × 4 × grad_accum(4) = 64 samples/step
BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
LEARNING_RATE=5e-5
# Qwen3.5-VL 图片 token ~3900-4100，max_length 必须 >= prompt+completion 总长
MAX_LENGTH=8192

REPORT_TO="tensorboard"

# --------------------------
# Runtime settings — 4 卡 torchrun DDP
# --------------------------
NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 干净的分布式 rendezvous
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE

# --------------------------
# Run Training (torchrun DDP)
# --------------------------
echo "Starting SFT training (0.8B, 4-GPU DDP, full supervision) ..."
echo "Model: $MODEL_CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Per-device batch: $BATCH_SIZE | grad accum: $GRAD_ACCUM"
echo "Effective samples/step: $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"
echo "epochs: $EPOCHS | lr: $LEARNING_RATE | max_length: $MAX_LENGTH"

.venv/bin/torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  train/sft.py \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --files-root "$FILES_ROOT" \
  --prompt-file "$PROMPT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --num-train-epochs "$EPOCHS" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --max-length "$MAX_LENGTH" \
  --report-to "$REPORT_TO"
