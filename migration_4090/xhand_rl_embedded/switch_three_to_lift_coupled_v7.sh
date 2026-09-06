#!/usr/bin/env bash
set -euo pipefail

# Append-only migration for the three v6 policies that learned table-supported
# force closure.  The v7 reward retains discovery credit but couples full
# force-closure shaping to continuous bimanual lift progress.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
switch_log="$root/logs/shaped_v7_lift_coupled_switch.log"
objects=(cracker_large pyramid cube)

declare -A devices=(
  [cracker_large]="cuda:2"
  [pyramid]="cuda:5"
  [cube]="cuda:4"
)
declare -A nominals=(
  [cracker_large]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
  [pyramid]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cube.pt"
)
declare -A nominal_hashes=(
  [cracker_large]="506d5e07da2994909bb5a4fbf85ec0cbf22b13c1300b3805e7297de5e49bf3ca"
  [pyramid]="b48e4a481d0bf6c06bc56e184fba4d1817bff91f181eab058b7d6696e203de4e"
  [cube]="5afd052d3b7ea16b5b54214b606423587ee0432306d66d59e5fc0c80ebc0ab6c"
)
declare -A checkpoints=(
  [cracker_large]="$root/runs/cracker_large/training_retry_009/model_1500.pt"
  [pyramid]="$root/runs/pyramid/training_retry_017/model_900.pt"
  [cube]="$root/runs/cube/training_retry_015/model_2600.pt"
)
declare -A checkpoint_hashes=(
  [cracker_large]="4b448d894893a05fa074022bed78b6473daaf21280483ae26abe53280a3901dc"
  [pyramid]="cf02bedbbfe6367e719b249266c40fbd8eb19680fa7868a4ceeea2dc9f4d5814"
  [cube]="4d5af8a498daf1679ddba9dfff31b9d2fa378f2d79e3869f2cfea3a36d75b5bc"
)
declare -A outputs=(
  [cracker_large]="$root/runs/cracker_large/training_retry_010"
  [pyramid]="$root/runs/pyramid/training_retry_018"
  [cube]="$root/runs/cube/training_retry_016"
)
declare -A max_iterations=(
  [cracker_large]="5500"
  [pyramid]="4900"
  [cube]="6600"
)
declare -A seeds=(
  [cracker_large]="84"
  [pyramid]="93"
  [cube]="196"
)
declare -A phases=(
  [cracker_large]="1"
  [pyramid]="1"
  [cube]="2"
)
declare -A integrations=(
  [cracker_large]="0.02"
  [pyramid]="0.02"
  [cube]="0.03"
)
declare -A arm_scales=(
  [cracker_large]="0.08"
  [pyramid]="0.08"
  [cube]="0.12"
)
declare -A hand_scales=(
  [cracker_large]="0.20"
  [pyramid]="0.20"
  [cube]="0.24"
)
declare -A penetration_weights=(
  [cracker_large]="-12.0"
  [pyramid]="-12.0"
  [cube]="-20.0"
)
declare -A noise_stds=(
  [cracker_large]="0.04"
  [pyramid]="0.04"
  [cube]="0.05"
)

mkdir -p "$root/logs"

worker_alive() {
  local object="$1"
  ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.(bridge_worker|train)/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
    END { exit(found ? 0 : 1) }
  '
}

restart_watchdog() {
  if ps -ww -eo comm=,args= | awk '
    ($1 == "bash") && $0 ~ /reconcile_target100\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    return
  fi
  echo "$(date --iso-8601=seconds) switch screen taking over as watchdog" >> "$switch_log"
  exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
}
trap restart_watchdog EXIT

# Validate every immutable input and append-only output before stopping v6.
for object in "${objects[@]}"; do
  [[ -f "${checkpoints[$object]}" ]]
  [[ -f "${nominals[$object]}" ]]
  [[ "$(sha256sum "${checkpoints[$object]}" | awk '{print $1}')" == "${checkpoint_hashes[$object]}" ]]
  [[ "$(sha256sum "${nominals[$object]}" | awk '{print $1}')" == "${nominal_hashes[$object]}" ]]
  if [[ -e "${outputs[$object]}" ]]; then
    echo "$(date --iso-8601=seconds) ERROR output exists object=$object output=${outputs[$object]}" >> "$switch_log"
    exit 1
  fi
done
echo "$(date --iso-8601=seconds) preflight passed" >> "$switch_log"

screen -S xhand_formal_v6_shaped_switch -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v5r_watchdog -X quit >/dev/null 2>&1 || true
for object in "${objects[@]}"; do
  screen -S "xhand_formal_v6_shaped_${object}" -X quit >/dev/null 2>&1 || true
done

for _ in $(seq 1 90); do
  any_alive=0
  for object in "${objects[@]}"; do
    if worker_alive "$object"; then any_alive=1; break; fi
  done
  (( any_alive == 0 )) && break
  sleep 1
done
for object in "${objects[@]}"; do
  if worker_alive "$object"; then
    echo "$(date --iso-8601=seconds) ERROR v6 worker did not stop object=$object" >> "$switch_log"
    exit 1
  fi
done

sleep 90
screen -wipe >/dev/null 2>&1 || true

for object in "${objects[@]}"; do
  screen_name="xhand_formal_v7_lift_coupled_${object}"
  log="$root/logs/$object/shaped_v7_lift_coupled_formal_screen.log"
  mkdir -p "$root/logs/$object"
  command=(
    env PYTHONPATH="$repo" "$isaac_python"
    -m migration_4090.xhand_rl_embedded.train
    --manifest "$manifest" --object "$object"
    --nominal-dataset "${nominals[$object]}" --output "${outputs[$object]}"
    --num-envs 128 --max-iterations "${max_iterations[$object]}"
    --seed "${seeds[$object]}" --device "${devices[$object]}" --headless
    --kit-portable-root "/tmp/xhand_rl_embedded_kit/v7_lift_coupled/$object"
    --penetration-reward-weight "${penetration_weights[$object]}"
    --terminal-success-weight 2500.0
    --init-noise-std "${noise_stds[$object]}" --entropy-coef 0.0001
    --resume-checkpoint "${checkpoints[$object]}"
    --residual-activation-phase "${phases[$object]}"
    --instantaneous-residual-weight 0.0
    --residual-integration "${integrations[$object]}"
    --arm-action-scale-rad "${arm_scales[$object]}"
    --hand-action-scale-rad "${hand_scales[$object]}"
    --stable-lift-reward-weight 16.0
    --force-closure-reward-weight 12.0
    --force-closure-shape-all-hold-contacts
    --reset-optimizer-on-resume
    --freeze-policy-noise
  )
  if [[ "${phases[$object]}" == "1" ]]; then command+=(--gamma 0.999); fi

  echo "$(date --iso-8601=seconds) launch object=$object output=${outputs[$object]} source=${checkpoints[$object]}" >> "$switch_log"
  env -u STY screen -L -Logfile "$log" -dmS "$screen_name" "${command[@]}"

  verified=0
  for _ in $(seq 1 120); do
    if worker_alive "$object" && [[ -f "${outputs[$object]}/training_provenance.json" ]] &&
      "$isaac_python" - "${outputs[$object]}/training_provenance.json" \
        "${checkpoints[$object]}" "${checkpoint_hashes[$object]}" \
        "${phases[$object]}" "${integrations[$object]}" \
        "${arm_scales[$object]}" "${hand_scales[$object]}" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
assert Path(p["resume_checkpoint"]).resolve() == Path(sys.argv[2]).resolve()
assert p["resume_checkpoint_sha256"] == sys.argv[3]
assert p["residual_activation_phase"] == int(sys.argv[4])
assert p["residual_integration"] == float(sys.argv[5])
assert p["arm_action_scale_rad"] == float(sys.argv[6])
assert p["hand_action_scale_rad"] == float(sys.argv[7])
assert p["stable_lift_reward_weight"] == 16.0
assert p["force_closure_reward_weight"] == 12.0
assert p["force_closure_shape_all_hold_contacts"] is True
assert p["reward_shaping_revision"] == "stable_lift_force_closure_v7_lift_coupled"
assert p["resume_optimizer_state_loaded"] is False
assert p["exact_training_rollout_capture"] is True
assert p["exact_checkpoint_timing"] == "before_ppo_update"
assert p["training_trajectory_recording"] is True
PY
    then
      verified=1
      break
    fi
    sleep 1
  done
  if (( verified == 0 )); then
    echo "$(date --iso-8601=seconds) ERROR v7 worker/provenance failed object=$object" >> "$switch_log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object screen=$screen_name" >> "$switch_log"
  sleep 20
done

echo "$(date --iso-8601=seconds) all v7 workers verified" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
