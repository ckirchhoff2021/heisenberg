#!/bin/bash
# 构建 vllm 0.27 + trl 训练镜像（含 Qwen3.5 GatedDeltaNet kernel）
# 用法: bash scripts/build_vllm027_trl_image.sh [TAG]
#   默认 TAG=v0.0.2；可传参覆盖，如 bash scripts/build_vllm027_trl_image.sh v0.0.3
set -euo pipefail

IMAGE_NAME="vllm027-trl-train"
IMAGE_TAG="${1:-v0.0.2}"

cd "$(dirname "$0")/.."

echo "=== 构建镜像 ${IMAGE_NAME}:${IMAGE_TAG} ==="
docker build \
  -f docker/Dockerfile.vllm027-trl \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

echo ""
echo "=== 验证镜像内的关键包版本 + Qwen3.5 kernel ==="
docker run --rm --entrypoint /bin/bash "${IMAGE_NAME}:${IMAGE_TAG}" -c "
echo '--- torch ---';    python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 | tail -1
echo '--- vllm ---';     python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1
echo '--- trl ---';      python3 -c 'import trl; print(trl.__version__)' 2>&1 | tail -1
echo '--- transformers ---'; python3 -c 'import transformers; print(transformers.__version__)' 2>&1 | tail -1
echo '--- causal-conv1d ---'; python3 -c 'import causal_conv1d; print(causal_conv1d.__version__)' 2>&1 | tail -1
echo '--- fla ---';      python3 -c 'import fla; print(\"fla OK\")' 2>&1 | tail -1
" 2>&1 | grep -vE "UserWarning|Warning|ResourceTracker|_stop|__del__|_recursion|multiprocess|ignored|Failed to load|libcudart|cuda\.cudart|cuda\.nvrtc"

echo ""
echo "=== 镜像构建完成: ${IMAGE_NAME}:${IMAGE_TAG} ==="
