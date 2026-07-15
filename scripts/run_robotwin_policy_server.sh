#!/bin/bash
# Start LingBot-VLA WebSocket server for RoboTwin eval.
# Usage: bash scripts/run_robotwin_policy_server.sh [gpu_id] [port]

set -euo pipefail

GPU_ID=${1:-0}
PORT=${2:-8006}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export QWEN25_PATH="${QWEN25_PATH:-${ROOT}/weights/Qwen2.5-VL-3B-Instruct}"
export TMPDIR="${TMPDIR:-/mnt/data14/yyg/tmp}"

MODEL_PATH="${MODEL_PATH:-${ROOT}/lingbot-vla-4b-posttrain-robotwin/lingbot-vla-4b-posttrain-robotwin}"
NORM_PATH="${NORM_PATH:-${ROOT}/assets/norm_stats/robotwin_50.json}"

cd "${ROOT}"
echo "GPU=${GPU_ID} PORT=${PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "QWEN25_PATH=${QWEN25_PATH}"

python -m deploy.lingbot_vla_policy \
  --model_path "${MODEL_PATH}" \
  --norm_path "${NORM_PATH}" \
  --use_length 50 \
  --port "${PORT}" \
  --use_compile
