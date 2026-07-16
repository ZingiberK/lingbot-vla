#!/usr/bin/env bash
# Phase 1 data collection — pure LingBot-VLA rollouts -> Motus finetune npz.
#
# For every SUCCESSFUL episode, RoboTwin/script/rl_rollout_worker.py saves per chunk:
#   <RL_ROOT>/<task>/round0/traj/<task>_seed<seed>_ep<ep>.npz
#     cam_high/left/right    : [n_chunk, H, W, 3] uint8   (chunk first frame, 3 views)
#     future_high/left/right : [n_chunk, T, H, W, 3] uint8 (future frames, 3 views)
#     state                  : [n_chunk, 14] float32
#     target                 : [n_chunk, 16, 14] float32
#     instruction            : str
#
# Full recollect (50 tasks, ~100 success/task, 8 GPUs):
#   RL_ROOT=/mnt/data14/yyg/wrm_rl_runs/motus_v1 TARGET_SUCCESS=100 NUM_SEEDS=180 \
#   nohup bash scripts/collect_motus_v1.sh > logs/collect_motus_v1.log 2>&1 &
#
# Smoke: SMOKE=1 GPUS=0 TARGET_SUCCESS=2 NUM_SEEDS=6 bash scripts/collect_motus_v1.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                       # lingbot-vla
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/data14/yyg/RoboTwin}"
CONDA_BASE="${CONDA_BASE:-/mnt/data14/ccy/pip_packs/miniconda3}"
VLA_ENV="${VLA_ENV:-lingbotvla}"
ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin}"

FULL_TASKS="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size \
click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block \
handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad \
move_playingcard_away move_stapler_pad open_laptop open_microwave \
pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right \
place_bread_basket place_bread_skillet place_burger_fries place_can_basket \
place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup \
place_fan place_mouse_pad place_object_basket place_object_scale \
place_object_stand place_phone_stand place_shoe press_stapler \
put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object \
shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two \
stack_bowls_three stack_bowls_two stamp_seal turn_switch"

TASKS=(${TASKS:-${FULL_TASKS}})
RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_v1}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/lingbot-vla-4b-posttrain-robotwin/lingbot-vla-4b-posttrain-robotwin}"
NORM_PATH="${NORM_PATH:-${ROOT}/assets/norm_stats/robotwin_50.json}"
QWEN25_PATH="${QWEN25_PATH:-${ROOT}/weights/Qwen2.5-VL-3B-Instruct}"
export QWEN25_PATH
export LINGBOT_VLA_ROOT="${ROOT}"
export TMPDIR="${TMPDIR:-/mnt/data14/yyg/tmp}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"
PORT="${PORT:-8300}"
ROUND=0
TARGET_SUCCESS="${TARGET_SUCCESS:-100}"
NUM_SEEDS="${NUM_SEEDS:-180}"
USE_LENGTH="${USE_LENGTH:-50}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  TASKS=(stack_blocks_three handover_block)
  echo "[SMOKE] tasks=${TASKS[*]} gpus=${GPUS} target=${TARGET_SUCCESS} seeds=${NUM_SEEDS}"
fi

mkdir -p "${RL_ROOT}" "${ROOT}/logs"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 1) start one pure-VLA server per GPU ----
conda activate "${VLA_ENV}"
cd "${ROOT}"
SERVER_PIDS=(); SERVER_PORTS=()
for g in "${GPU_ARR[@]}"; do
  p=$((PORT + g))
  if ss -tln 2>/dev/null | grep -q ":${p} "; then log "port ${p} in use, reuse"; SERVER_PORTS+=("${p}"); continue; fi
  CUDA_VISIBLE_DEVICES="${g}" nohup python -m deploy.lingbot_vla_policy \
      --model_path "${MODEL_PATH}" --norm_path "${NORM_PATH}" --use_length "${USE_LENGTH}" \
      --port "${p}" >"${ROOT}/logs/vla_server_gpu${g}.log" 2>&1 &
  SERVER_PIDS+=($!); SERVER_PORTS+=("${p}")
done
log "waiting for VLA servers ${SERVER_PORTS[*]} ..."
for p in "${SERVER_PORTS[@]}"; do
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${p}/healthz" >/dev/null 2>&1; then break; fi
    sleep 5
  done
done

cleanup() { for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT

# ---- 2) per task: scan seeds (rank0), then sharded collection workers ----
conda activate "${ROBOTWIN_ENV}"
cd "${ROBOTWIN_ROOT}"
for task in "${TASKS[@]}"; do
  # Skip tasks already at target (resumable across restarts).
  # mkdir first: `find` on a missing path exits 1 and would kill the run under
  # `set -e` (that is what stopped the previous launch after beat_block_hammer).
  traj_dir="${RL_ROOT}/${task}/round${ROUND}/traj"
  mkdir -p "${traj_dir}"
  have=$(find "${traj_dir}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  if [[ "${TARGET_SUCCESS}" -gt 0 && "${have}" -ge "${TARGET_SUCCESS}" ]]; then
    log "SKIP ${task}: already ${have}/${TARGET_SUCCESS} npz"; continue
  fi

  # Reuse a cached feasible-seed list if present (own prior scan OR pre-seeded
  # from an earlier collection); expert-feasibility is stable, so this skips the
  # single-GPU ~25min/task scan. Only scan fresh when no cache exists.
  seeds_cache="${RL_ROOT}/${task}/train_seeds.json"
  if [[ -s "${seeds_cache}" ]]; then
    log "SEEDS_REUSE ${task}: $(python -c "import json;print(len(json.load(open('${seeds_cache}'))))" 2>/dev/null || echo '?') cached"
  else
    log "SCAN ${task}"
    # Pin the scan to a single GPU; unpinned, curobo's warp init touches every
    # visible device and a single bad/[N/A] GPU aborts the whole process.
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python script/rl_rollout_worker.py \
        --task_name "${task}" --rl_root "${RL_ROOT}" \
        --round "${ROUND}" --num_seeds "${NUM_SEEDS}" --scan_only
  fi

  log "COLLECT ${task} (shards=${NGPU}, target=${TARGET_SUCCESS})"
  WPIDS=()
  for i in "${!GPU_ARR[@]}"; do
    g="${GPU_ARR[$i]}"; p="${SERVER_PORTS[$i]}"
    CUDA_VISIBLE_DEVICES="${g}" python script/rl_rollout_worker.py \
        --task_name "${task}" --host 127.0.0.1 --port "${p}" --use_length "${USE_LENGTH}" \
        --rl_root "${RL_ROOT}" --round "${ROUND}" --num_seeds "${NUM_SEEDS}" \
        --shard_id "${i}" --num_shards "${NGPU}" --target_success "${TARGET_SUCCESS}" &
    WPIDS+=($!)
  done
  for pid in "${WPIDS[@]}"; do wait "${pid}" || true; done
  n=$(find "${traj_dir}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  log "DONE ${task}: ${n} traj npz"
done

log "ALL DONE. root=${RL_ROOT}"
