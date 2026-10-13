#!/bin/bash
# OPD + GRPO 混合训练启动脚本
# Modify the parameters below as needed

# --------------------------
# Training Configuration
# --------------------------
# Model settings
MODEL_CHECKPOINT="checkpoints/Qwen3.5-0.8B"
TEACHER_CHECKPOINT="checkpoints/Qwen3.5-7B"  # 冻结的强教师，需与学生同词表
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Data settings
TRAIN_DATA_PATH="datas/aiops/files/jsons/seed-grpo.json"
FILES_ROOT="datas/aiops/files"          # 图片根目录，按训练机实际路径修改
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt.md"

# Output settings
OUTPUT_DIR="outputs/grpo_opd"

# Training hyperparameters
BATCH_SIZE=4
EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
LEARNING_RATE=5e-6
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=4096

# OPD混合训练参数
OPD_LOSS_WEIGHT=1.0  # 蒸馏项权重 lambda，0表示退化为纯GRPO训练
OPD_KL_DIRECTION="forward"  # forward: KL(teacher||student)；reverse: KL(student||teacher)

# Precision settings (H20/A100 use BF16 by default; uncomment FP16 to switch)
FP16=""

# Other settings
USE_VLLM=""
REPORT_TO="tensorboard"

# --------------------------
# Run Training
# --------------------------
echo "Starting OPD+GRPO mixed training with the following configuration:"
echo "Student model: $MODEL_CHECKPOINT"
echo "Teacher model: $TEACHER_CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "OPD loss weight: $OPD_LOSS_WEIGHT ($OPD_KL_DIRECTION KL)"

python train/grpo_opd_mixed.py \
  --model-checkpoint "$MODEL_CHECKPOINT" \
  --teacher-checkpoint "$TEACHER_CHECKPOINT" \
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
  --opd-loss-weight "$OPD_LOSS_WEIGHT" \
  --opd-kl-direction "$OPD_KL_DIRECTION" \
  $FP16 \
  $USE_VLLM \
  --report-to "$REPORT_TO"
