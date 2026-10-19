#!/bin/bash
# GRPO Training Launch Script — Qwen3.5-0.8B — 8卡数据并行 (reward v2 + colocate vLLM)
# 与 train_grpo.sh 的区别：
#   - 8 卡 torchrun DDP（数据并行），每卡内嵌一个 colocate vLLM engine
#   - vllm_mode=colocate + tensor_parallel_size=1（纯数据并行，无 server 单点）
#   - reward v2：错误=0 + 纯 inverse-frequency 类平衡（详见 train/grpo.py 注释）
#   - beta=0.1 防坍缩锚定
#   - 独立输出目录 outputs/grpo_vllm_8x08b_v2

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

OUTPUT_DIR="outputs/grpo_vllm_8x08b_v2"

# 8 卡数据并行：per_device_batch_size 是「每卡」的 batch
# effective batch = 8 × 4 × num_generations(4) = 128 completions/step
BATCH_SIZE=4
EPOCHS=3
LOGGING_STEPS=5
SAVE_STEPS=100
LEARNING_RATE=5e-6
NUM_GENERATIONS=4
MAX_COMPLETION_LENGTH=1024

# reward v2：beta 加强锚定
BETA=0.1

# 精度
FP16=""

# colocate vLLM 配置
USE_VLLM="--use-vllm"
VLLM_MODE="colocate"
VLLM_TENSOR_PARALLEL_SIZE=1          # 1 = 纯数据并行，每卡一个 engine
VLLM_GPU_MEMORY_UTILIZATION=0.4      # 0.8B 模型小，训练 LoRA + vLLM 共卡，0.4 足够
REPORT_TO="tensorboard"

# --------------------------
# Runtime settings — 8 卡 torchrun DDP
# --------------------------
PYTHON=".venv/bin/python"
NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# colocate vLLM 需要干净的分布式 rendezvous（TRL 内部 ensure_master_addr_port 会处理，
# 但显式设置避免与残留环境变量冲突）
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE

# --------------------------
# Run Training (torchrun DDP)
# --------------------------
echo "Starting GRPO training (0.8B, 8-GPU DDP, reward v2) ..."
echo "Model: $MODEL_CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Beta: $BETA | TP size: $VLLM_TENSOR_PARALLEL_SIZE | GPU mem util: $VLLM_GPU_MEMORY_UTILIZATION"
echo "Per-device batch: $BATCH_SIZE | Num generations: $NUM_GENERATIONS"
echo "Effective completions/step: $((NPROC_PER_NODE * BATCH_SIZE * NUM_GENERATIONS))"

.venv/bin/torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train/grpo.py \
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
  --beta "$BETA" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  $FP16 \
  $USE_VLLM \
  --vllm-mode "$VLLM_MODE" \
  --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --report-to "$REPORT_TO"
