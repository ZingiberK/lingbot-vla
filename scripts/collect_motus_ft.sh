#!/usr/bin/env bash
# Motus finetune data collection (Tier 1, see Motus/WRM_RL_PLAN_MOTUS_REFINE.md §10).
#
# Pure VLA-zero rollouts; for every SUCCESSFUL episode, save Motus-usable tuples per chunk:
#   motus_ft/<task>_seed<seed>_ep<ep>.npz :
#     cam_high/left/right : [n_chunk, H, W, 3] uint8   (3-view raw frames)
#     state               : [n_chunk, 14]     float32  (raw qpos, Motus joint order)
#     target              : [n_chunk, 16, 14] float32  (a_vla raw qpos, stride-3 -> Motus grid)
#     instruction         : scalar str
#
# Used to BC-finetune Motus's action expert (video/und frozen) toward VLA-stride3, then
# re-test SDEdit low-t0 refine: does closing the manifold gap revive the blend? (Tier 1 §10)
#
# All vla_zero (delta=0), NO SDE trace / NO futures / NO learner. Single round. 8-GPU sharded.
#
#   nohup bash scripts/collect_motus_ft.sh > logs/motus_ft_collect.nohup.log 2>&1 &
# Smoke (2 tasks, tiny):
#   SMOKE=1 GPU=0 bash scripts/collect_motus_ft.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"                       # lingbot-vla
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/data14/yyg/RoboTwin}"
CONDA_BASE="${CONDA_BASE:-/mnt/data14/ccy/pip_packs/miniconda3}"

FULL_TASKS="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size \
click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block \
handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad \
move_playingcard_away move_stapler_pad open_microwave \
pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right \
place_bread_basket place_bread_skillet place_burger_fries place_can_basket \
place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup \
place_fan place_mouse_pad place_object_basket \
place_object_stand place_phone_stand place_shoe press_stapler \
put_bottles_dustbin rotate_qrcode scan_object \
shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two \
stack_bowls_three stack_bowls_two stamp_seal turn_switch"
# NOTE: open_laptop / place_object_scale / put_object_cabinet dropped (RoboTwin env
# arm_tag AttributeError -> 0 episodes). 47 tasks.

TASKS=(${TASKS:-${FULL_TASKS}})
RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_ft}"
INIT_CKPT="${INIT_CKPT:-/mnt/data14/yyg/Motus/runs/wrm_und_zero_v1/wrm_und_step040000.pt}"
DELTA_STATS="${DELTA_STATS:-${ROOT}/residual_data/robotwin_full_clean_rand_und_zero/delta_norm_stats.json}"
WAN_DIR="${WAN_DIR:-/mnt/data14/liuxiao/pretrained_models/Wan2.2-TI2V-5B}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/lingbot-vla-4b-posttrain-robotwin/lingbot-vla-4b-posttrain-robotwin}"
NORM_PATH="${NORM_PATH:-${ROOT}/assets/norm_stats/robotwin_50.json}"

GPU="${GPU:-0}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"
PORT="${PORT:-8300}"

ROUND=0
# 50 distinct layouts/task, 1 rollout each (VLA ~88% SR -> ~44 success traj/task).
# ~10 chunks/traj -> ~440 (frame,stride3-action) samples/task; ample for action-expert BC.
NUM_SEEDS="${NUM_SEEDS:-50}"
GROUP_SIZE="${GROUP_SIZE:-1}"
ETA="${ETA:-0.5}"          # unused for vla_zero (delta forced to 0), server still needs it
RL_STEPS="${RL_STEPS:-10}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  TASKS=(stack_blocks_three handover_block)
  GPUS="${GPU}"; IFS=',' read -r -a GPU_ARR <<< "${GPUS}"; NGPU="${#GPU_ARR[@]}"
  NUM_SEEDS=3; GROUP_SIZE=1
  echo "[SMOKE] tasks=${TASKS[*]} gpus=${GPUS} num_seeds=${NUM_SEEDS} group_size=${GROUP_SIZE}"
fi

mkdir -p "${RL_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
MAIN_LOG="${RL_ROOT}/orchestrator.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"; }

is_rollout_done() {
  local task="$1" need n
  need=$(( NUM_SEEDS * GROUP_SIZE ))
  local ep_dir="${RL_ROOT}/${task}/round${ROUND}/episodes"
  [[ -d "${ep_dir}" ]] || return 1
  n=$(find "${ep_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  [[ "${n}" -ge "${need}" ]]
}

wait_ws_ready() {
  local port="$1" pid="$2"
  local py="${CONDA_BASE}/envs/RoboTwin/bin/python"
  for i in $(seq 1 180); do
    ss -tln 2>/dev/null | grep -q ":${port} " && \
    "${py}" -c "
import sys; sys.path.insert(0, '${ROOT}')
import websockets.sync.client
from deploy.msgpack_numpy import unpackb
c = websockets.sync.client.connect('ws://127.0.0.1:${port}', compression=None, max_size=None, open_timeout=5)
unpackb(c.recv()); c.close()
" 2>/dev/null && { log "server ready on :${port} (${i}x5s)"; return 0; }
    sleep 5
    kill -0 "${pid:-0}" 2>/dev/null || { log "ERROR: server :${port} exited early (see ${RL_ROOT}/server_gpu*.log)"; return 1; }
  done
  log "ERROR: server not ready on :${port}"; return 1
}

log "MOTUS FT COLLECT | tasks=${#TASKS[@]} gpus=${GPUS}(N=${NGPU}) seeds=${NUM_SEEDS} G=${GROUP_SIZE} -> ${NUM_SEEDS}*${GROUP_SIZE}=$(( NUM_SEEDS*GROUP_SIZE ))/task | root=${RL_ROOT}"

# ---- 0) scan feasible train seeds once per task, NGPU parallel ----
conda activate RoboTwin
export LINGBOT_VLA_ROOT="${ROOT}"
cd "${ROBOTWIN_ROOT}"
SCAN_PIDS=(); slot=0
for task in "${TASKS[@]}"; do
  mkdir -p "${RL_ROOT}/${task}/round${ROUND}"
  [[ -f "${RL_ROOT}/${task}/train_seeds.json" ]] && { log "seeds cached task=${task}"; continue; }
  gpu="${GPU_ARR[$(( slot % NGPU ))]}"
  log "scan_only task=${task} gpu=${gpu} num_seeds=${NUM_SEEDS}"
  CUDA_VISIBLE_DEVICES="${gpu}" python script/rl_rollout_worker.py \
    --task_name "${task}" --rl_root "${RL_ROOT}" --round "${ROUND}" \
    --num_seeds "${NUM_SEEDS}" --group_size "${GROUP_SIZE}" --scan_only \
    >> "${RL_ROOT}/${task}/round${ROUND}/scan_stdout.log" 2>&1 &
  SCAN_PIDS+=($!); slot=$(( slot + 1 ))
  if (( ${#SCAN_PIDS[@]} >= NGPU )); then
    for pid in "${SCAN_PIDS[@]}"; do wait "${pid}" || log "WARN scan pid=${pid} nonzero"; done
    SCAN_PIDS=()
  fi
done
for pid in "${SCAN_PIDS[@]}"; do wait "${pid}" || log "WARN scan pid=${pid} nonzero"; done

# ---- 1) rollout server(s) (lingbotvla): one per GPU ----
conda activate lingbotvla
export QWEN25_PATH="${QWEN25_PATH:-${ROOT}/weights/Qwen2.5-VL-3B-Instruct}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
SERVER_PIDS=(); SERVER_PORTS=()
for (( g=0; g<NGPU; g++ )); do
  p=$(( PORT + g )); gpu="${GPU_ARR[$g]}"
  log "starting server gpu=${gpu} port=${p} (ckpt=${INIT_CKPT})"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m deploy.lingbot_wrm_rl_server \
    --model_path "${MODEL_PATH}" --norm_path "${NORM_PATH}" --use_length 50 \
    --wrm_ckpt "${INIT_CKPT}" --delta_stats "${DELTA_STATS}" --wan_dir "${WAN_DIR}" \
    --video_mode denoise --rl_steps "${RL_STEPS}" --eta "${ETA}" \
    --rl_root "${RL_ROOT}" --rl_round "${ROUND}" --port "${p}" \
    >> "${RL_ROOT}/server_gpu${gpu}.log" 2>&1 &
  SERVER_PIDS+=($!); SERVER_PORTS+=("${p}")
done
for (( g=0; g<NGPU; g++ )); do
  wait_ws_ready "${SERVER_PORTS[$g]}" "${SERVER_PIDS[$g]}" || {
    for pid in "${SERVER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; exit 1; }
done

# ---- 2) RoboTwin rollout worker(s): NGPU shards in parallel per task (motus_ft_collect) ----
conda activate RoboTwin
export LINGBOT_VLA_ROOT="${ROOT}"
cd "${ROBOTWIN_ROOT}"
for task in "${TASKS[@]}"; do
  if is_rollout_done "${task}"; then log "SKIP done task=${task}"; continue; fi
  mkdir -p "${RL_ROOT}/${task}/round${ROUND}"
  log "collect task=${task} shards=${NGPU} (motus_ft_collect)"
  WPIDS=()
  for (( g=0; g<NGPU; g++ )); do
    p=$(( PORT + g )); gpu="${GPU_ARR[$g]}"
    CUDA_VISIBLE_DEVICES="${gpu}" python script/rl_rollout_worker.py \
      --task_name "${task}" --port "${p}" --use_length 50 \
      --rl_root "${RL_ROOT}" --round "${ROUND}" \
      --num_seeds "${NUM_SEEDS}" --group_size "${GROUP_SIZE}" \
      --shard_id "${g}" --num_shards "${NGPU}" --save_futures false \
      --motus_ft_collect \
      >> "${RL_ROOT}/${task}/round${ROUND}/worker_shard${g}_stdout.log" 2>&1 &
    WPIDS+=($!)
  done
  FAIL=0
  for pid in "${WPIDS[@]}"; do wait "${pid}" || FAIL=1; done
  [[ "${FAIL}" == "1" ]] && log "WARN task=${task} had a worker shard return nonzero"
  ndone=$(find "${RL_ROOT}/${task}/round${ROUND}/episodes" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
  nft=$(find "${RL_ROOT}/${task}/round${ROUND}/motus_ft" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
  log "task=${task} DONE episodes=${ndone} motus_ft=${nft}"
done

# ---- 3) stop server(s) ----
for pid in "${SERVER_PIDS[@]}"; do
  log "stopping server pid=${pid}"; kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true
done

# ---- 4) final tally ----
TOT_EP=$(find "${RL_ROOT}" -path '*/round0/episodes/*.json' 2>/dev/null | wc -l)
TOT_FT=$(find "${RL_ROOT}" -path '*/round0/motus_ft/*.npz' 2>/dev/null | wc -l)
log "ALL DONE. total_episodes=${TOT_EP} total_motus_ft_traj=${TOT_FT} root=${RL_ROOT}"
