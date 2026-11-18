#!/usr/bin/env bash
# d3-flash 部署脚本（独立 stack，不与 jev-4B-aiops 冲突）
# 用法: bash semantic_router/deploy_d3_flash.sh
#
# 关键点：
#   - unset PYTHONPATH：避免外部环境注入的 PYTHONPATH 污染
#   - VLLM_SR_STACK_NAME=d3flash：独立 stack，容器名带前缀，端口用 port-offset 隔离
#   - --device-ids 5：绑定 GPU5（空闲卡）
#   - 监听端口 8898（config 里写死），management/metrics 用 port-offset 1000 避开 8080/9190
set -eu
unset PYTHONPATH

export VLLM_SR_STACK_NAME="d3flash"
export VLLM_SR_PORT_OFFSET="1000"

# 按需替换为自己仓库的 vLLM Semantic Router 镜像和引擎配置
IMAGE="${VLLM_SR_IMAGE:-vllm-semantic-router:0.5.0-dev}"
CONFIG="${VLLM_SR_CONFIG:-checkpoints/semantic-router/sr-engine-flash.yaml}"

vllm-sr serve \
  --config "$CONFIG" \
  --engine \
  --platform cuda \
  --device-ids 5 \
  --max-num-batched-tokens 8192 \
  --image "$IMAGE" \
  --image-pull-policy "${VLLM_SR_PULL_POLICY:-IfNotPresent}" \
  --replace-active-config \
  --verbose
