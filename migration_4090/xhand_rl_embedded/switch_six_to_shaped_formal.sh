#!/usr/bin/env bash
set -euo pipefail

# Switch the six active legacy-reward formal workers to append-only PPO retries
# using the validated stable-lift and all-hold force-closure shaping revision.
# Historical checkpoints and attempts are immutable; every output below must be
# absent before this script is allowed to stop an active worker.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
watchdog_screen="xhand_formal_v5r_watchdog"
switch_log="$root/logs/shaped_v6_switch.log"

objects=(sphere sphere_small cracker_large cracker pyramid cube)

declare -A devices=(
  [sphere]="cuda:0"
  [sphere_small]="cuda:1"
  [cracker_large]="cuda:2"
  [cracker]="cuda:3"
  [pyramid]="cuda:5"
  [cube]="cuda:4"
)
declare -A nominals=(
  [sphere]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/sphere.pt"
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/sphere_small.pt"
  [cracker_large]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cracker.pt"
  [pyramid]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cube.pt"
)
declare -A nominal_hashes=(
  [sphere]="b7d7f40e108496be6f5e65319a554733f07ddc70ac6cbe483e415ddcdad6bc09"
  [sphere_small]="7cec72c2d32f7792ebb69f42de4229ae98b62c89f6e34a4b020e6dbb02265c8b"
  [cracker_large]="506d5e07da2994909bb5a4fbf85ec0cbf22b13c1300b3805e7297de5e49bf3ca"
  [cracker]="e7e159bf9d75f424f35de4202b98d60c7d337b5d2979a02c3a0c2943e2cff1ad"
  [pyramid]="b48e4a481d0bf6c06bc56e184fba4d1817bff91f181eab058b7d6696e203de4e"
  [cube]="5afd052d3b7ea16b5b54214b606423587ee0432306d66d59e5fc0c80ebc0ab6c"
)
declare -A checkpoints=(
  [sphere]="$root/runs/sphere/training_retry_008/model_800.pt"
  [sphere_small]="$root/runs/sphere_small/training/model_3900.pt"
  [cracker_large]="$root/runs/cracker_large/training_retry_008/model_1400.pt"
  [cracker]="$root/runs/cracker/training/model_2800.pt"
  [pyramid]="$root/runs/pyramid/training_retry_016/model_800.pt"
  [cube]="$root/runs/cube/training/model_2500.pt"
)
declare -A checkpoint_hashes=(
  [sphere]="6391d969d9f8b36f3e0cb210c81d1f96a0a6a7f5ae95a1d68de5c87515fa6489"
  [sphere_small]="e44568d975786495d02d60079d5d69d60ab9d539f433beda036b9c7f3975e264"
  [cracker_large]="9736ef5b57183c4b12ed703ee7ab3bd3641fe9a40fcfeab6df538c8353036025"
  [cracker]="92f0eaecc462a444d6b648ca233a3e2ec095c04d2a7627a814de7a309d928429"
  [pyramid]="afb9e6c40df7d647f7311f26a3d7065b3522993250ec3e438a98cdba73a8a10b"
  [cube]="00c327a75020a8080f9e7f34ad96229f391d28fcb29389dc79bcf2a175567d75"
)
declare -A outputs=(
  [sphere]="$root/runs/sphere/training_retry_009"
  [sphere_small]="$root/runs/sphere_small/training_retry_013"
  [cracker_large]="$root/runs/cracker_large/training_retry_009"
  [cracker]="$root/runs/cracker/training_retry_010"
  [pyramid]="$root/runs/pyramid/training_retry_017"
  [cube]="$root/runs/cube/training_retry_015"
)
declare -A max_iterations=(
  [sphere]="4800"
  [sphere_small]="7900"
  [cracker_large]="5400"
  [cracker]="6800"
  [pyramid]="4800"
  [cube]="6500"
)
declare -A seeds=(
  [sphere]="84"
  [sphere_small]="194"
  [cracker_large]="84"
  [cracker]="195"
  [pyramid]="93"
  [cube]="196"
)
declare -A phases=(
  [sphere]="1"
  [sphere_small]="2"
  [cracker_large]="1"
  [cracker]="2"
  [pyramid]="1"
  [cube]="2"
)
declare -A integrations=(
  [sphere]="0.02"
  [sphere_small]="0.03"
  [cracker_large]="0.02"
  [cracker]="0.03"
  [pyramid]="0.02"
  [cube]="0.03"
)
declare -A arm_scales=(
  [sphere]="0.08"
  [sphere_small]="0.12"
  [cracker_large]="0.08"
  [cracker]="0.12"
  [pyramid]="0.08"
  [cube]="0.12"
)
declare -A hand_scales=(
  [sphere]="0.20"
  [sphere_small]="0.24"
  [cracker_large]="0.20"
  [cracker]="0.24"
  [pyramid]="0.20"
  [cube]="0.24"
)
declare -A penetration_weights=(
  [sphere]="-12.0"
  [sphere_small]="-20.0"
  [cracker_large]="-12.0"
  [cracker]="-20.0"
  [pyramid]="-12.0"
  [cube]="-20.0"
)
declare -A noise_stds=(
  [sphere]="0.04"
  [sphere_small]="0.05"
  [cracker_large]="0.04"
  [cracker]="0.05"
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

# Complete all immutable-input and append-only-output checks before touching
# the currently running trainers.
for object in "${objects[@]}"; do
  [[ -f "${checkpoints[$object]}" ]]
  [[ -f "${nominals[$object]}" ]]
  [[ "$(sha256sum "${checkpoints[$object]}" | awk '{print $1}')" == "${checkpoint_hashes[$object]}" ]]
  [[ "$(sha256sum "${nominals[$object]}" | awk '{print $1}')" == "${nominal_hashes[$object]}" ]]
  if [[ -e "${outputs[$object]}" ]]; then
    echo "$(date --iso-8601=seconds) ERROR append-only output exists object=$object output=${outputs[$object]}" >> "$switch_log"
    exit 1
  fi
done
echo "$(date --iso-8601=seconds) preflight passed" >> "$switch_log"

screen -S "$watchdog_screen" -X quit >/dev/null 2>&1 || true
for object in "${objects[@]}"; do
  screen -S "xhand_formal_v5r_${object}_bridge" -X quit >/dev/null 2>&1 || true
  screen -S "xhand_formal_v5r_${object}_historical_exact" -X quit >/dev/null 2>&1 || true
done

for _ in $(seq 1 90); do
  any_alive=0
  for object in "${objects[@]}"; do
    if worker_alive "$object"; then
      any_alive=1
      break
    fi
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

# Kit, Vulkan, and PhysX release resources asynchronously after process exit.
sleep 90
screen -wipe >/dev/null 2>&1 || true

for object in "${objects[@]}"; do
  screen_name="xhand_formal_v6_shaped_${object}"
  log="$root/logs/$object/shaped_v6_formal_screen.log"
  mkdir -p "$root/logs/$object"
  echo "$(date --iso-8601=seconds) launch object=$object output=${outputs[$object]} source=${checkpoints[$object]}" >> "$switch_log"

  command=(
    env PYTHONPATH="$repo" "$isaac_python"
    -m migration_4090.xhand_rl_embedded.train
    --manifest "$manifest" --object "$object"
    --nominal-dataset "${nominals[$object]}" --output "${outputs[$object]}"
    --num-envs 128 --max-iterations "${max_iterations[$object]}"
    --seed "${seeds[$object]}" --device "${devices[$object]}" --headless
    --kit-portable-root "/tmp/xhand_rl_embedded_kit/v6_shaped/$object"
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
  if [[ "${phases[$object]}" == "1" ]]; then
    command+=(--gamma 0.999)
  fi
  env -u STY screen -L -Logfile "$log" -dmS "$screen_name" "${command[@]}"

  verified=0
  for _ in $(seq 1 120); do
    if worker_alive "$object" && [[ -f "${outputs[$object]}/training_provenance.json" ]] &&
      "$isaac_python" - "${outputs[$object]}/training_provenance.json" \
        "${checkpoints[$object]}" "${checkpoint_hashes[$object]}" \
        "${nominals[$object]}" "${nominal_hashes[$object]}" \
        "${phases[$object]}" "${integrations[$object]}" \
        "${arm_scales[$object]}" "${hand_scales[$object]}" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
assert Path(p["resume_checkpoint"]).resolve() == Path(sys.argv[2]).resolve()
assert p["resume_checkpoint_sha256"] == sys.argv[3]
assert Path(p["nominal_dataset"]).resolve() == Path(sys.argv[4]).resolve()
assert p["nominal_dataset_sha256"] == sys.argv[5]
assert p["residual_activation_phase"] == int(sys.argv[6])
assert p["residual_integration"] == float(sys.argv[7])
assert p["arm_action_scale_rad"] == float(sys.argv[8])
assert p["hand_action_scale_rad"] == float(sys.argv[9])
assert p["instantaneous_residual_weight"] == 0.0
assert p["stable_lift_reward_weight"] == 16.0
assert p["force_closure_reward_weight"] == 12.0
assert p["force_closure_shape_all_hold_contacts"] is True
assert p["reward_shaping_revision"] == "stable_lift_force_closure_v6"
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
    echo "$(date --iso-8601=seconds) ERROR shaped worker/provenance failed object=$object" >> "$switch_log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object screen=$screen_name" >> "$switch_log"
  sleep 20
done

echo "$(date --iso-8601=seconds) all six shaped workers verified" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
