#!/usr/bin/env bash
# Tier 1 FULL eval, parallel over 4 GPU pairs (mirrors run_robotwin_wrm_fast_8gpu's 8-GPU spread).
#
# Motus refine needs TWO ~40GB servers (VLA + Motus) that won't co-fit on one 80GB GPU,
# so we pair GPUs: (0,1)(2,3)(4,5)(6,7) -> 4 concurrent pipelines, each = 1 VLA + 1 Motus
# server + a RoboTwin worker, processing a disjoint shard of tasks. Each task = 100 eval
# episodes via eval_policy-style inline scan from st_seed=100000 (no pre-scan, no manual seeds).
#
# This is just a task-sharding wrapper around run_motus_ft_eval.sh (one instance per pair).
#
#   MOTUS_T0=0.3 nohup bash scripts/run_motus_ft_eval_8gpu.sh > logs/motus_ft_eval_8gpu_t03.nohup.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MOTUS_T0="${MOTUS_T0:-0.3}"
FT_CKPT="${FT_CKPT:-/mnt/data14/yyg/Motus/runs/motus_ft_action_v1/action_expert_final.pt}"
RL_ROOT="${RL_ROOT:-/mnt/data14/yyg/wrm_rl_runs/motus_ft_eval_full/t0_${MOTUS_T0}}"
NUM_SEEDS="${NUM_SEEDS:-100}"

FULL_TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
  handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
  move_playingcard_away move_stapler_pad open_laptop open_microwave
  pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
  place_bread_basket place_bread_skillet place_burger_fries place_can_basket
  place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
  place_fan place_mouse_pad place_object_basket place_object_scale
  place_object_stand place_phone_stand place_shoe press_stapler
  put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
  shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two
  stack_bowls_three stack_bowls_two stamp_seal turn_switch
)
if [[ -n "${TASKS:-}" ]]; then read -r -a FULL_TASKS <<< "${TASKS}"; fi

# 4 GPU pairs -> 4 concurrent shards. VLA_GPU MOTUS_GPU VLA_PORT MOTUS_PORT per pair.
PAIRS=("0 1 8300 8400" "2 3 8302 8402" "4 5 8304 8404" "6 7 8306 8406")
NPAIR=${#PAIRS[@]}

mkdir -p "${RL_ROOT}" "${ROOT}/logs"
MAIN_LOG="${RL_ROOT}/orchestrator_8gpu.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MAIN_LOG}"; }

log "MOTUS FT EVAL 8GPU | tasks=${#FULL_TASKS[@]} pairs=${NPAIR} t0=${MOTUS_T0} test_num=${NUM_SEEDS} ft=${FT_CKPT}"

# round-robin assign tasks to pairs
declare -a SHARD_TASKS
for ((i=0; i<NPAIR; i++)); do SHARD_TASKS[$i]=""; done
for ((t=0; t<${#FULL_TASKS[@]}; t++)); do
  p=$(( t % NPAIR ))
  SHARD_TASKS[$p]+="${FULL_TASKS[$t]} "
done

PIDS=()
for ((i=0; i<NPAIR; i++)); do
  read -r vg mg vp mp <<< "${PAIRS[$i]}"
  sub="${SHARD_TASKS[$i]}"
  [[ -z "${sub// }" ]] && continue
  log "pair${i}: VLA_GPU=${vg} MOTUS_GPU=${mg} ports=${vp}/${mp} | tasks=[ ${sub}]"
  TASKS="${sub}" MOTUS_T0="${MOTUS_T0}" FT_CKPT="${FT_CKPT}" RL_ROOT="${RL_ROOT}" \
    NUM_SEEDS="${NUM_SEEDS}" VLA_GPU="${vg}" MOTUS_GPU="${mg}" VLA_PORT="${vp}" MOTUS_PORT="${mp}" \
    nohup bash "${ROOT}/scripts/run_motus_ft_eval.sh" \
      > "${ROOT}/logs/motus_ft_eval_pair${i}_t${MOTUS_T0}.nohup.log" 2>&1 &
  PIDS+=($!)
  sleep 10   # stagger server starts to avoid a thundering-herd on model load I/O
done

log "launched ${#PIDS[@]} pairs (pids: ${PIDS[*]}); waiting ..."
FAIL=0
for pid in "${PIDS[@]}"; do wait "${pid}" || FAIL=1; done
[[ "${FAIL}" == "1" ]] && log "WARN: a pair returned nonzero"

# ---- aggregate SR over all tasks ----
log "==== MOTUS FT EVAL 8GPU SUMMARY (t0=${MOTUS_T0}, test_num=${NUM_SEEDS}) ===="
GRAND_EP=0; GRAND_SUCC=0
for task in "${FULL_TASKS[@]}"; do
  ep_dir="${RL_ROOT}/${task}/round0/episodes"
  [[ -d "${ep_dir}" ]] || continue
  n=$(find "${ep_dir}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l || true)
  s=$( { grep -l '"success": true' "${ep_dir}"/*.json 2>/dev/null || true; } | wc -l)
  GRAND_EP=$(( GRAND_EP + n )); GRAND_SUCC=$(( GRAND_SUCC + s ))
  [[ "${n}" -gt 0 ]] && log "  ${task}: ${s}/${n} = $(awk "BEGIN{printf \"%.3f\", ${s}/${n}}")"
done
[[ "${GRAND_EP}" -gt 0 ]] && log "OVERALL: ${GRAND_SUCC}/${GRAND_EP} = $(awk "BEGIN{printf \"%.3f\", ${GRAND_SUCC}/${GRAND_EP}}")"
log "ALL DONE. root=${RL_ROOT}"
