#!/bin/bash
# SFT Training Launch Script — Qwen3.5-0.8B — 4卡数据并行（无 reason 版）
# 与 train_sft_08b_4gpu.sh（v1，含 reason）的关键区别：
#   - completion 只输出 <trend>/<recover> 标签，去掉长文本 <reason>
#   - prompt 用 argos-dp-prompt-noreason.md（输出格式只要求标签，few-shot 也删了 reason）
# 原因：v1 里 reason 占 completion loss 的 84%，模型只学复述 reason、学不到
#   「图片→标签」映射，导致验证集/训练集都坍缩到固定输出 offset+no_recover。

# --------------------------
# Training Configuration
# --------------------------
MODEL_CHECKPOINT="checkpoints/Qwen3.5-0.8B"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

TRAIN_DATA_PATH="datas/aiops/files/jsons/argos-trains.json"
FILES_ROOT="datas/aiops/files"
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt-noreason.md"

OUTPUT_DIR="outputs/sft_08b_4gpu_v2_noreason"

# 4 卡数据并行：per_device_batch_size 是「每卡」的 batch
# effective batch = 4 × 4 × grad_accum(4) = 64 samples/step
BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
LEARNING_RATE=5e-5
# Qwen3.5-VL 图片 token ~3900-4100，max_length 必须 >= prompt+completion 总长
# 无 reason 后 completion 极短（~10 token），8192 仍安全（prompt 本身 ~4100）
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
echo "Starting SFT training (0.8B, 4-GPU DDP, no-reason full supervision) ..."
echo "Model: $MODEL_CHECKPOINT"
echo "Prompt: $PROMPT_FILE"
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
