#!/bin/bash
# Phase 2 — Motus RL closed loop (Flow-SDE PPO + supervised WM).
#
# Per round r:
#   1) start Motus RL server (motus env) with the current deploy ckpt
#   2) RoboTwin RL rollout worker (robotwin env) -> episodes/traces/futures
#   3) stop server
#   4) motus_rl.ppo_update (motus env) -> round<r>_action.pt (action+und)
#   5) motus_rl.merge_round_ckpt -> next-round deploy ckpt
#
# Video (WAN) branch is trained by the supervised FM pass inside ppo_update; the
# action expert is updated by PPO (isolated gradients). See Motus/MOTUS_PLAN.md.
#
# Fill in the paths / conda envs below before running.

set -euo pipefail

# ---- paths ----
MOTUS_ROOT="${MOTUS_ROOT:-/mnt/data14/yyg/Motus}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/data14/yyg/RoboTwin}"
LINGBOT_VLA_ROOT="${LINGBOT_VLA_ROOT:-/mnt/data14/yyg/lingbot-vla}"
export MOTUS_INFER_ROOT="${MOTUS_INFER_ROOT:-${MOTUS_ROOT}/inference/robotwin/Motus}"

WAN_PATH="${WAN_PATH:-/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B}"
VLM_PATH="${VLM_PATH:-/share/home/bhz/pretrained_models/Qwen3-VL-2B-Instruct}"
BASE_CKPT="${BASE_CKPT:-${MOTUS_ROOT}/deploy_ckpts/motus_finetune}"   # Phase-1 deploy ckpt
RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_rl}"
CKPT_DIR="${CKPT_DIR:-${MOTUS_ROOT}/deploy_ckpts}"

# ---- conda envs ----
CONDA_SH="${CONDA_SH:-/mnt/data14/ccy/pip_packs/miniconda3/etc/profile.d/conda.sh}"
MOTUS_ENV="${MOTUS_ENV:-motus}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"

# ---- run config ----
TASKS="${TASKS:-stack_blocks_three}"
NUM_ROUNDS="${NUM_ROUNDS:-3}"
NUM_SEEDS="${NUM_SEEDS:-16}"
GROUP_SIZE="${GROUP_SIZE:-4}"
PORT="${PORT:-8400}"
ETA="${ETA:-0.5}"
GPU="${GPU:-0}"

source "${CONDA_SH}"
mkdir -p "${CKPT_DIR}"

CUR_CKPT="${BASE_CKPT}"

for ((r=0; r<NUM_ROUNDS; r++)); do
  echo "==================== ROUND ${r} (ckpt=${CUR_CKPT}) ===================="

  # 1) start server (motus env)
  conda activate "${MOTUS_ENV}"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m deploy.motus_rl_server \
      --motus_ckpt "${CUR_CKPT}" --wan "${WAN_PATH}" --vlm "${VLM_PATH}" --port "${PORT}" \
      >"${RL_ROOT}/server_round${r}.log" 2>&1 &
  SERVER_PID=$!
  echo "server pid=${SERVER_PID}, waiting for :${PORT} ..."
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then break; fi
    sleep 5
  done

  # 2) rollout (robotwin env)
  conda activate "${ROBOTWIN_ENV}"
  cd "${ROBOTWIN_ROOT}"
  for task in ${TASKS}; do
    CUDA_VISIBLE_DEVICES="${GPU}" python script/motus_rl_rollout_worker.py \
        --task_name "${task}" --host 127.0.0.1 --port "${PORT}" \
        --rl_root "${RL_ROOT}" --round "${r}" \
        --num_seeds "${NUM_SEEDS}" --group_size "${GROUP_SIZE}" --eta "${ETA}"
  done

  # 3) stop server
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true

  # 4) PPO update (motus env)
  conda activate "${MOTUS_ENV}"
  cd "${MOTUS_ROOT}"
  OUT_PPO="${CKPT_DIR}/round${r}_action.pt"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m motus_rl.ppo_update \
      --tasks ${TASKS} --rl_root "${RL_ROOT}" --round "${r}" \
      --motus_ckpt "${CUR_CKPT}" --wan "${WAN_PATH}" --vlm "${VLM_PATH}" \
      --out_ckpt "${OUT_PPO}"

  # 5) merge -> next-round deploy ckpt
  NEXT_CKPT="${CKPT_DIR}/motus_rl_round$((r+1))"
  python -m motus_rl.merge_round_ckpt --base "${CUR_CKPT}" --ppo_payload "${OUT_PPO}" --out "${NEXT_CKPT}"
  CUR_CKPT="${NEXT_CKPT}"
done

echo "Motus RL done. Final ckpt: ${CUR_CKPT}"
