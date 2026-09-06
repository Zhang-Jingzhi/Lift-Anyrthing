#!/usr/bin/env bash
set -euo pipefail

# Migrate the remaining stalled objects to v7 while restoring the phase-0
# residual semantics and legacy nominal paired with their historical policy
# checkpoints.  sphere_small/cracker start immediately before observed hard
# successes; sphere starts from its strongest historical force-closure basin.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
switch_log="$root/logs/shaped_v7_phase0_switch.log"
objects=(sphere sphere_small cracker)

declare -A devices=(
  [sphere]="cuda:0"
  [sphere_small]="cuda:1"
  [cracker]="cuda:3"
)
declare -A nominals=(
  [sphere]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/sphere.pt"
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/sphere_small.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cracker.pt"
)
declare -A nominal_hashes=(
  [sphere]="0507a12fd832a4062060e701d135769624adf39e7cb0f48bf6ac2272ac08fdb9"
  [sphere_small]="7cec72c2d32f7792ebb69f42de4229ae98b62c89f6e34a4b020e6dbb02265c8b"
  [cracker]="e7e159bf9d75f424f35de4202b98d60c7d337b5d2979a02c3a0c2943e2cff1ad"
)
declare -A checkpoints=(
  [sphere]="$root/runs/sphere/training/model_1100.pt"
  [sphere_small]="$root/runs/sphere_small/training/model_3900.pt"
  [cracker]="$root/runs/cracker/training/model_2800.pt"
)
declare -A checkpoint_hashes=(
  [sphere]="a0d593a4a04f3a6861ca39d62eecd133c50baac195b2b91f2dc839439fd24522"
  [sphere_small]="e44568d975786495d02d60079d5d69d60ab9d539f433beda036b9c7f3975e264"
  [cracker]="92f0eaecc462a444d6b648ca233a3e2ec095c04d2a7627a814de7a309d928429"
)
declare -A outputs=(
  [sphere]="$root/runs/sphere/training_retry_010"
  [sphere_small]="$root/runs/sphere_small/training_retry_014"
  [cracker]="$root/runs/cracker/training_retry_011"
)
declare -A max_iterations=(
  [sphere]="5100"
  [sphere_small]="7900"
  [cracker]="6800"
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

screen -S xhand_formal_v7_lift_coupled_switch -X quit >/dev/null 2>&1 || true
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
    echo "$(date --iso-8601=seconds) ERROR old worker did not stop object=$object" >> "$switch_log"
    exit 1
  fi
done

sleep 90
screen -wipe >/dev/null 2>&1 || true

for object in "${objects[@]}"; do
  screen_name="xhand_formal_v7_phase0_${object}"
  log="$root/logs/$object/shaped_v7_phase0_formal_screen.log"
  mkdir -p "$root/logs/$object"
  echo "$(date --iso-8601=seconds) launch object=$object output=${outputs[$object]} source=${checkpoints[$object]}" >> "$switch_log"
  env -u STY screen -L -Logfile "$log" -dmS "$screen_name" \
    env PYTHONPATH="$repo" "$isaac_python" \
    -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" --object "$object" \
    --nominal-dataset "${nominals[$object]}" --output "${outputs[$object]}" \
    --num-envs 128 --max-iterations "${max_iterations[$object]}" \
    --seed 84 --device "${devices[$object]}" --headless \
    --kit-portable-root "/tmp/xhand_rl_embedded_kit/v7_phase0/$object" \
    --penetration-reward-weight -20.0 --terminal-success-weight 2500.0 \
    --init-noise-std 0.05 --entropy-coef 0.0001 \
    --resume-checkpoint "${checkpoints[$object]}" \
    --residual-activation-phase 0 \
    --instantaneous-residual-weight 0.0 --residual-integration 0.03 \
    --arm-action-scale-rad 0.12 --hand-action-scale-rad 0.24 \
    --stable-lift-reward-weight 16.0 \
    --force-closure-reward-weight 12.0 \
    --force-closure-shape-all-hold-contacts \
    --reset-optimizer-on-resume --freeze-policy-noise

  verified=0
  for _ in $(seq 1 120); do
    if worker_alive "$object" && [[ -f "${outputs[$object]}/training_provenance.json" ]] &&
      "$isaac_python" - "${outputs[$object]}/training_provenance.json" \
        "${checkpoints[$object]}" "${checkpoint_hashes[$object]}" \
        "${nominals[$object]}" "${nominal_hashes[$object]}" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
assert Path(p["resume_checkpoint"]).resolve() == Path(sys.argv[2]).resolve()
assert p["resume_checkpoint_sha256"] == sys.argv[3]
assert Path(p["nominal_dataset"]).resolve() == Path(sys.argv[4]).resolve()
assert p["nominal_dataset_sha256"] == sys.argv[5]
assert p["seed"] == 84
assert p["residual_activation_phase"] == 0
assert p["instantaneous_residual_weight"] == 0.0
assert p["residual_integration"] == 0.03
assert p["arm_action_scale_rad"] == 0.12
assert p["hand_action_scale_rad"] == 0.24
assert p["stable_lift_reward_weight"] == 16.0
assert p["force_closure_reward_weight"] == 12.0
assert p["force_closure_shape_all_hold_contacts"] is True
assert p["reward_shaping_revision"] == "stable_lift_force_closure_v7_lift_coupled"
assert p["resume_optimizer_state_loaded"] is False
assert p["exact_training_rollout_capture"] is True
assert p["training_trajectory_recording"] is True
PY
    then
      verified=1
      break
    fi
    sleep 1
  done
  if (( verified == 0 )); then
    echo "$(date --iso-8601=seconds) ERROR v7 phase0 worker/provenance failed object=$object" >> "$switch_log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object screen=$screen_name" >> "$switch_log"
  sleep 20
done

echo "$(date --iso-8601=seconds) all v7 phase0 workers verified" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
