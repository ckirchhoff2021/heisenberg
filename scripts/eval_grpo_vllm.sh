#!/bin/bash
# GRPO模型评测启动脚本（VLLM加速版）
# 比原生HuggingFace推理快5~10倍，支持本地引擎和远程服务两种模式

# --------------------------
# 基础配置
# --------------------------
# 推理模式选择：local=本地VLLM引擎(推荐) / remote=远程VLLM API服务
MODE="local"

# 公共配置
VALID_DATA_PATH="datas/aiops/files/jsons/argos-valids.json"
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt.md"
FILES_ROOT="datas/aiops/files"
OUTPUT_FILE="./eval_results_vllm.json"
MAX_NEW_TOKENS=1024

# --------------------------
# 本地模式配置（MODE=local时生效）
# --------------------------
MODEL_PATH="outputs/grpo/checkpoint-100"
BATCH_SIZE=32  # 0.8B模型H20显卡可开32~64，4B模型可开16~32

# --------------------------
# 远程模式配置（MODE=remote时生效）
# --------------------------
API_BASE="http://localhost:8000/v1"
API_KEY="EMPTY"
MODEL_NAME="qwen3.5-0.8b-curve"

# --------------------------
# 运行评测
# --------------------------
echo "Starting VLLM accelerated evaluation in $MODE mode..."

# 检查依赖
if ! python -c "import vllm, openai, sklearn, tqdm" > /dev/null 2>&1; then
    echo "Installing required dependencies..."
    pip install vllm openai scikit-learn tqdm
fi

if [ "$MODE" = "local" ]; then
    echo "Model path: $MODEL_PATH"
    echo "Batch size: $BATCH_SIZE"
    python evaluate/grpo_eval_vllm.py \
      --mode local \
      --model-path "$MODEL_PATH" \
      --batch-size "$BATCH_SIZE" \
      --valid-data-path "$VALID_DATA_PATH" \
      --prompt-file "$PROMPT_FILE" \
      --files-root "$FILES_ROOT" \
      --output-file "$OUTPUT_FILE" \
      --max-new-tokens "$MAX_NEW_TOKENS"
else
    echo "Remote API: $API_BASE"
    echo "Model name: $MODEL_NAME"
    python evaluate/grpo_eval_vllm.py \
      --mode remote \
      --api-base "$API_BASE" \
      --api-key "$API_KEY" \
      --model-name "$MODEL_NAME" \
      --valid-data-path "$VALID_DATA_PATH" \
      --prompt-file "$PROMPT_FILE" \
      --files-root "$FILES_ROOT" \
      --output-file "$OUTPUT_FILE" \
      --max-new-tokens "$MAX_NEW_TOKENS"
fi
