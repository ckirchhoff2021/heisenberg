#!/bin/bash
# GRPO模型评测启动脚本

# --------------------------
# 配置参数
# --------------------------
# 必填：训练好的模型路径（支持LoRA适配器路径或全量模型路径）
MODEL_PATH="outputs/grpo/checkpoint-100"

# 可选配置
VALID_DATA_PATH="datas/aiops/files/jsons/argos-valids.json"
PROMPT_FILE="datas/aiops/files/prompts/argos-dp-prompt.md"
FILES_ROOT="datas/aiops/files"
# 指标输出文件，不填则仅打印到控制台
OUTPUT_FILE="./eval_results.json"
MAX_NEW_TOKENS=1024
DEVICE="cuda"

# --------------------------
# 运行评测
# --------------------------
echo "Starting model evaluation..."
echo "Model path: $MODEL_PATH"
echo "Validation data: $VALID_DATA_PATH"
echo "Output metrics to: $OUTPUT_FILE"

# 检查依赖
if ! python -c "import sklearn, tqdm" > /dev/null 2>&1; then
    echo "Installing required dependencies..."
    pip install scikit-learn tqdm
fi

python evaluate/grpo_eval.py \
  --model-path "$MODEL_PATH" \
  --valid-data-path "$VALID_DATA_PATH" \
  --prompt-file "$PROMPT_FILE" \
  --files-root "$FILES_ROOT" \
  --output-file "$OUTPUT_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE"
