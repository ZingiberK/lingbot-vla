#!/bin/bash
# Start 8 LingBot policy servers (GPU g -> port 8006+g), keep running for fast eval.
# Usage: bash scripts/start_lingbot_servers_8gpu.sh [log_dir]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${1:-${ROOT}/logs/lingbot_servers_$(date +%Y%m%d_%H%M%S)}"
CONDA_BASE="${CONDA_BASE:-/mnt/data14/ccy/pip_packs/miniconda3}"
BASE_PORT="${BASE_PORT:-8006}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/server_pids.txt"
: > "${PID_FILE}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lingbotvla
export QWEN25_PATH="${QWEN25_PATH:-${ROOT}/weights/Qwen2.5-VL-3B-Instruct}"
export TMPDIR="${TMPDIR:-/mnt/data14/yyg/tmp}"

for ((g=0; g<NUM_GPUS; g++)); do
  port=$((BASE_PORT + g))
  if ss -tln 2>/dev/null | grep -q ":${port} "; then
    echo "port ${port} already in use, skip gpu${g}"
    continue
  fi
  export CUDA_VISIBLE_DEVICES="${g}"
  nohup python -m deploy.lingbot_vla_policy \
    --model_path "${MODEL_PATH:-${ROOT}/lingbot-vla-4b-posttrain-robotwin/lingbot-vla-4b-posttrain-robotwin}" \
    --norm_path "${NORM_PATH:-${ROOT}/assets/norm_stats/robotwin_50.json}" \
    --use_length 50 \
    --port "${port}" \
    --use_compile \
    >> "${LOG_DIR}/server_gpu${g}.log" 2>&1 &
  echo $! >> "${PID_FILE}"
  echo "started gpu${g} port=${port} pid=$!"
done

echo "LOG_DIR=${LOG_DIR}"
echo "Waiting for servers..."
for ((g=0; g<NUM_GPUS; g++)); do
  port=$((BASE_PORT + g))
  for i in $(seq 1 120); do
    ss -tln 2>/dev/null | grep -q ":${port} " && break
    sleep 5
  done
done
echo "Servers ready."
