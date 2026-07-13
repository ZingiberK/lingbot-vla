#!/usr/bin/env bash
# Tier 1 eval: finetuned Motus SDEdit on HELD-OUT eval seeds [100000,200000).
#
# Seed protocol mirrors RoboTwin eval_policy.py / LingBotVLA/eval.sh:
#   --seed 0 -> st_seed=100000, scan upward for expert-feasible layouts, test_num=100.
# Our --eval_seeds does the same (no manual seed list). Default NUM_SEEDS=100.
#
# Quick (2 tasks, 10 eval seeds):
#   SMOKE=1 MOTUS_T0=0.3 VLA_GPU=0 MOTUS_GPU=1 bash scripts/run_motus_ft_eval.sh
#
# Formal full (47 tasks, 100 eval episodes/task, t0=0.3):
#   MOTUS_T0=0.3 RL_ROOT=/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval_full/t0_0.3 \
#     nohup bash scripts/run_motus_ft_eval.sh > logs/motus_ft_eval_full_t03.nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/data14/yyg/RoboTwin}"
CONDA_BASE="${CONDA_BASE:-/mnt/data14/ccy/pip_packs/miniconda3}"
MOTUS_INFER_ROOT="${MOTUS_INFER_ROOT:-/mnt/data14/yyg/Motus/inference/robotwin/Motus}"

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
# 50 tasks — same set as run_ppo_full_v1 / LingBotVLA eval (comparison_clean.csv)

TASKS=(${TASKS:-${FULL_TASKS}})
FT_CKPT="${FT_CKPT:-/mnt/data14/yyg/Motus/runs/motus_ft_action_v1/action_expert_final.pt}"
MOTUS_T0="${MOTUS_T0:-0.3}"
RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval_full/t0_${MOTUS_T0}}"

INIT_CKPT="${INIT_CKPT:-/mnt/data14/yyg/Motus/runs/wrm_und_zero_v1/wrm_und_step040000.pt}"
DELTA_STATS="${DELTA_STATS:-${ROOT}/residual_data/robotwin_full_clean_rand_und_zero/delta_norm_stats.json}"
WAN_DIR="${WAN_DIR:-/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/lingbot-vla-4b-posttrain-robotwin/lingbot-vla-4b-posttrain-robotwin}"
NORM_PATH="${NORM_PATH:-${ROOT}/assets/norm_stats/robotwin_50.json}"
MOTUS_CKPT="${MOTUS_CKPT:-/mnt/data14/liuxiao/pretrained_models/Motus_robotwin2}"
MOTUS_VLM="${MOTUS_VLM:-/mnt/data14/liuxiao/pretrained_models/Qwen3-VL-2B-Instruct}"

VLA_GPU="${VLA_GPU:-0}"
MOTUS_GPU="${MOTUS_GPU:-1}"
VLA_PORT="${VLA_PORT:-8300}"
MOTUS_PORT="${MOTUS_PORT:-8400}"
ROUND=0
NUM_SEEDS="${NUM_SEEDS:-100}"   # matches eval_policy.py test_num=100
GROUP_SIZE=1
RL_STEPS=10
ETA=0.5

if [[ "${SMOKE:-0}" == "1" ]]; then
  TASKS=(stack_blocks_three handover_block)
  NUM_SEEDS=10
  RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval/t0_${MOTUS_T0}}"
  echo "[SMOKE] tasks=${TASKS[*]} eval_seeds=${NUM_SEEDS} t0=${MOTUS_T0} ft=${FT_CKPT}"
fi

[[ -f "${FT_CKPT}" ]] || { echo "ERROR: FT_CKPT not found: ${FT_CKPT}"; exit 1; }

mkdir -p "${RL_ROOT}" "${ROOT}/logs"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
MAIN_LOG="${RL_ROOT}/orchestrator.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"; }

wait_ws_ready() {
  local port="$1" pid="$2" name="$3"
  local py="${CONDA_BASE}/envs/RoboTwin/bin/python"
  for i in $(seq 1 240); do
    ss -tln 2>/dev/null | grep -q ":${port} " && \
    "${py}" -c "
import sys; sys.path.insert(0, '${ROOT}')
import websockets.sync.client
from deploy.msgpack_numpy import unpackb
c = websockets.sync.client.connect('ws://127.0.0.1:${port}', compression=None, max_size=None, open_timeout=5)
unpackb(c.recv()); c.close()
" 2>/dev/null && { log "${name} server ready on :${port} (${i}x5s)"; return 0; }
    sleep 5
    kill -0 "${pid:-0}" 2>/dev/null || { log "ERROR: ${name} server :${port} exited early"; return 1; }
  done
  log "ERROR: ${name} server not ready on :${port}"; return 1
}

is_rollout_done() {
  local task="$1" need n
  need=$(( NUM_SEEDS * GROUP_SIZE ))
  local ep_dir="${RL_ROOT}/${task}/round${ROUND}/episodes"
  [[ -d "${ep_dir}" ]] || return 1
  n=$(find "${ep_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  [[ "${n}" -ge "${need}" ]]
}

log "MOTUS FT EVAL | tasks=${#TASKS[@]} test_num=${NUM_SEEDS} t0=${MOTUS_T0} ft=${FT_CKPT} | root=${RL_ROOT}"
log "seed protocol: eval_policy-style (st_seed=100000, scan+rollout inline per task, NO pre-scan pass)"

# ---- VLA server ----
conda activate lingbotvla
export QWEN25_PATH="${QWEN25_PATH:-${ROOT}/weights/Qwen2.5-VL-3B-Instruct}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
log "starting VLA server gpu=${VLA_GPU} port=${VLA_PORT}"
CUDA_VISIBLE_DEVICES="${VLA_GPU}" nohup python -m deploy.lingbot_wrm_rl_server \
  --model_path "${MODEL_PATH}" --norm_path "${NORM_PATH}" --use_length 50 \
  --wrm_ckpt "${INIT_CKPT}" --delta_stats "${DELTA_STATS}" --wan_dir "${WAN_DIR}" \
  --video_mode denoise --rl_steps "${RL_STEPS}" --eta "${ETA}" \
  --rl_root "${RL_ROOT}" --rl_round "${ROUND}" --port "${VLA_PORT}" \
  >> "${RL_ROOT}/vla_server_gpu${VLA_GPU}.log" 2>&1 &
VLA_PID=$!

# ---- Motus FT server ----
conda activate motus
export PYTHONUNBUFFERED=1
export MOTUS_INFER_ROOT="${MOTUS_INFER_ROOT}"
cd "${ROOT}"
log "starting Motus FT server gpu=${MOTUS_GPU} port=${MOTUS_PORT} t0=${MOTUS_T0}"
CUDA_VISIBLE_DEVICES="${MOTUS_GPU}" nohup python -m deploy.motus_sdedit_server \
  --motus_ckpt "${MOTUS_CKPT}" --wan "${WAN_DIR}" --vlm "${MOTUS_VLM}" \
  --port "${MOTUS_PORT}" --default_t0 "${MOTUS_T0}" --steps 10 \
  --ft_action_ckpt "${FT_CKPT}" \
  >> "${RL_ROOT}/motus_server_gpu${MOTUS_GPU}.log" 2>&1 &
MOTUS_PID=$!

cleanup() { for pid in "${VLA_PID:-0}" "${MOTUS_PID:-0}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT

wait_ws_ready "${VLA_PORT}" "${VLA_PID}" "VLA" || { cleanup; exit 1; }
wait_ws_ready "${MOTUS_PORT}" "${MOTUS_PID}" "Motus" || { cleanup; exit 1; }

# ---- refine rollouts (eval_policy-style: worker scans from 100000 + rolls inline) ----
conda activate RoboTwin
export LINGBOT_VLA_ROOT="${ROOT}"
cd "${ROBOTWIN_ROOT}"
for task in "${TASKS[@]}"; do
  if is_rollout_done "${task}"; then log "SKIP done task=${task}"; continue; fi
  mkdir -p "${RL_ROOT}/${task}/round${ROUND}"
  log "eval task=${task} t0=${MOTUS_T0} (test_num=${NUM_SEEDS}, eval band)"
  CUDA_VISIBLE_DEVICES="${VLA_GPU}" python script/rl_rollout_worker.py \
    --task_name "${task}" --host 127.0.0.1 --port "${VLA_PORT}" --use_length 50 \
    --rl_root "${RL_ROOT}" --round "${ROUND}" \
    --num_seeds "${NUM_SEEDS}" --group_size "${GROUP_SIZE}" --eval_seeds \
    --refine_motus --motus_host 127.0.0.1 --motus_port "${MOTUS_PORT}" \
    --motus_t0 "${MOTUS_T0}" --motus_chunk 16 --motus_ds 3 \
    >> "${RL_ROOT}/${task}/round${ROUND}/worker_stdout.log" 2>&1 || log "WARN task=${task}"
  ndone=$(find "${RL_ROOT}/${task}/round${ROUND}/episodes" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l || true)
  nsucc=$( { grep -l '"success": true' "${RL_ROOT}/${task}/round${ROUND}/episodes"/*.json 2>/dev/null || true; } | wc -l)
  log "task=${task} DONE episodes=${ndone} success=${nsucc}"
done

cleanup; trap - EXIT

log "==== MOTUS FT EVAL SUMMARY (eval_seeds, t0=${MOTUS_T0}) ===="
GRAND_EP=0; GRAND_SUCC=0
for task in "${TASKS[@]}"; do
  ep_dir="${RL_ROOT}/${task}/round${ROUND}/episodes"
  [[ -d "${ep_dir}" ]] || continue
  n=$(find "${ep_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l || true)
  s=$( { grep -l '"success": true' "${ep_dir}"/*.json 2>/dev/null || true; } | wc -l)
  GRAND_EP=$(( GRAND_EP + n )); GRAND_SUCC=$(( GRAND_SUCC + s ))
  [[ "${n}" -gt 0 ]] && log "  ${task}: ${s}/${n} = $(awk "BEGIN{printf \"%.3f\", ${s}/${n}}")"
done
[[ "${GRAND_EP}" -gt 0 ]] && log "OVERALL: ${GRAND_SUCC}/${GRAND_EP} = $(awk "BEGIN{printf \"%.3f\", ${GRAND_SUCC}/${GRAND_EP}}")"
log "ALL DONE. root=${RL_ROOT}"
