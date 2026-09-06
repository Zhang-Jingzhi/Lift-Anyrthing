#!/usr/bin/env bash
set -euo pipefail

# Move three stalled formal lineages onto immutable checkpoints immediately
# preceding previously observed training hard-success events.  Each target is
# a new append-only retry directory with exact pre-PPO-update capture enabled;
# current retry checkpoints and every historical artifact remain untouched.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
watchdog_screen="xhand_formal_v5r_watchdog"
switch_log="$root/logs/proven_historical_seed_switch.log"
resume_existing="${XHAND_HISTORICAL_RESUME_EXISTING:-0}"

declare -a default_objects=(sphere_small cracker cube)
declare -A devices=(
  [sphere_small]="cuda:1"
  [cracker]="cuda:3"
  [cube]="cuda:4"
)
declare -A old_screens=(
  [sphere_small]="xhand_formal_v5r_sphere_small_bridge"
  [cracker]="xhand_formal_v5r_cracker_bridge"
  [cube]="xhand_formal_v5r_cube_bridge"
)
declare -A nominals=(
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/sphere_small.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cracker.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cube.pt"
)
declare -A seeds=(
  [sphere_small]="$root/runs/sphere_small/training/model_3900.pt"
  [cracker]="$root/runs/cracker/training/model_2800.pt"
  [cube]="$root/runs/cube/training/model_2500.pt"
)
declare -A seed_hashes=(
  [sphere_small]="e44568d975786495d02d60079d5d69d60ab9d539f433beda036b9c7f3975e264"
  [cracker]="92f0eaecc462a444d6b648ca233a3e2ec095c04d2a7627a814de7a309d928429"
  [cube]="00c327a75020a8080f9e7f34ad96229f391d28fcb29389dc79bcf2a175567d75"
)
declare -A hard_success_iterations=(
  [sphere_small]="3917"
  [cracker]="2814"
  [cube]="2537"
)
declare -A outputs=(
  [sphere_small]="$root/runs/sphere_small/training_retry_011"
  [cracker]="$root/runs/cracker/training_retry_009"
  [cube]="$root/runs/cube/training_retry_014"
)
declare -A max_iterations=(
  [sphere_small]="7900"
  [cracker]="6800"
  [cube]="6500"
)
declare -A training_seeds=(
  [sphere_small]="194"
  [cracker]="195"
  [cube]="196"
)

if (( $# > 0 )); then
  objects=("$@")
else
  objects=("${default_objects[@]}")
fi
for object in "${objects[@]}"; do
  if [[ -z "${devices[$object]:-}" ]]; then
    echo "unknown historical-seed object: $object" >&2
    exit 2
  fi
done

mkdir -p "$root/logs"

resume_watchdog() {
  if ps -ww -eo comm=,args= | awk '
    ($1 == "bash") && $0 ~ /reconcile_target100\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    return
  fi
  echo "$(date --iso-8601=seconds) switch screen taking over as watchdog" >> "$switch_log"
  exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
}
trap resume_watchdog EXIT

for object in "${objects[@]}"; do
  seed="${seeds[$object]}"
  actual_hash="$(sha256sum "$seed" | awk '{print $1}')"
  if [[ "$actual_hash" != "${seed_hashes[$object]}" ]]; then
    echo "$(date --iso-8601=seconds) ERROR object=$object seed_hash=$actual_hash" >> "$switch_log"
    exit 1
  fi
  if [[ -e "${outputs[$object]}" ]]; then
    if [[ "$resume_existing" != "1" || ! -d "${outputs[$object]}" || \
      -f "${outputs[$object]}/TRAINING_COMPLETE.json" ]]; then
      echo "$(date --iso-8601=seconds) ERROR output already exists object=$object output=${outputs[$object]}" \
        >> "$switch_log"
      exit 1
    fi
    "$isaac_python" - "${outputs[$object]}/training_provenance.json" \
      "${seeds[$object]}" "${training_seeds[$object]}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
seed = Path(sys.argv[2])
assert payload["resume_checkpoint_iteration"] == int(seed.stem.rsplit("_", 1)[1])
assert payload["seed"] == int(sys.argv[3])
assert payload["exact_training_rollout_capture"] is True
assert payload["exact_checkpoint_timing"] == "before_ppo_update"
assert payload["training_trajectory_recording"] is True
assert payload["residual_activation_phase"] == 2
assert payload["instantaneous_residual_weight"] == 0.0
assert payload["residual_integration"] == 0.03
PY
    echo "$(date --iso-8601=seconds) resume_existing object=$object output=${outputs[$object]}" \
      >> "$switch_log"
  fi
  "$isaac_python" - "$root/runs/$object/training/TRAINING_COMPLETE.json" \
    "${hard_success_iterations[$object]}" "$seed" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text())
success_iteration = int(sys.argv[2])
seed = Path(sys.argv[3]).resolve()
selection = marker["checkpoint_selection"]
assert selection["training_hard_success_observed"] is True
assert int(selection["training_hard_success_iteration"]) == success_iteration
seed_iteration = int(seed.stem.rsplit("_", 1)[1])
assert seed_iteration <= success_iteration
assert success_iteration - seed_iteration <= 100
PY
done

screen -S "$watchdog_screen" -X quit >/dev/null 2>&1 || true

for object in "${objects[@]}"; do
  direct_screen="xhand_formal_v5r_${object}_historical_exact"
  echo "$(date --iso-8601=seconds) stop object=$object old_screen=${old_screens[$object]}" \
    >> "$switch_log"
  screen -S "${old_screens[$object]}" -X quit >/dev/null 2>&1 || true
  screen -S "$direct_screen" -X quit >/dev/null 2>&1 || true
  for _ in $(seq 1 90); do
    if ! ps -ww -eo comm=,args= | awk -v object="$object" '
      ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
      $0 ~ /migration_4090\.xhand_rl_embedded\.(bridge_worker|train)/ &&
      $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
      END { exit(found ? 0 : 1) }
    '; then
      break
    fi
    sleep 1
  done
  if ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.(bridge_worker|train)/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    echo "$(date --iso-8601=seconds) ERROR object=$object old process did not stop" >> "$switch_log"
    exit 1
  fi

  # Stagger teardown/startup across physical GPUs to avoid simultaneous Kit
  # allocator and Vulkan initialization races.
  sleep 90
  mkdir -p "$root/logs/$object"
  echo "$(date --iso-8601=seconds) launch object=$object seed=${seeds[$object]} output=${outputs[$object]}" \
    >> "$switch_log"
  env -u STY screen -L \
    -Logfile "$root/logs/$object/historical_success_seed_formal_screen.log" \
    -dmS "$direct_screen" env PYTHONPATH="$repo" "$isaac_python" \
    -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" --object "$object" \
    --nominal-dataset "${nominals[$object]}" --output "${outputs[$object]}" \
    --num-envs 128 --max-iterations "${max_iterations[$object]}" \
    --seed "${training_seeds[$object]}" --device "${devices[$object]}" --headless \
    --kit-portable-root "/tmp/xhand_rl_embedded_kit/v5r_historical_exact/$object" \
    --penetration-reward-weight -20.0 --terminal-success-weight 2500.0 \
    --init-noise-std 0.05 --entropy-coef 0.0001 \
    --resume-checkpoint "${seeds[$object]}" --residual-activation-phase 2 \
    --instantaneous-residual-weight 0.0 --residual-integration 0.03 \
    --arm-action-scale-rad 0.12 --hand-action-scale-rad 0.24 \
    --freeze-policy-noise

  verified=0
  for _ in $(seq 1 90); do
    child_pid="$(ps -ww -eo pid=,comm=,args= | awk \
      -v object="$object" -v output="${outputs[$object]}" '
      ($2 == "python" || $2 ~ /^python[0-9.]+$/) &&
      $0 ~ /migration_4090\.xhand_rl_embedded\.train/ &&
      $0 ~ ("--object[ =]+" object "([ =]|$)") &&
      index($0, output) { print $1; exit }
    ')"
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null && \
      [[ -f "${outputs[$object]}/training_provenance.json" ]] && \
      "$isaac_python" - "${outputs[$object]}/training_provenance.json" \
        "${seeds[$object]}" "${training_seeds[$object]}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
seed = Path(sys.argv[2])
assert payload["resume_checkpoint_iteration"] == int(seed.stem.rsplit("_", 1)[1])
assert payload["seed"] == int(sys.argv[3])
assert payload["exact_training_rollout_capture"] is True
assert payload["exact_checkpoint_timing"] == "before_ppo_update"
assert payload["training_trajectory_recording"] is True
assert payload["residual_activation_phase"] == 2
assert payload["instantaneous_residual_weight"] == 0.0
assert payload["residual_integration"] == 0.03
PY
    then
      verified=1
      break
    fi
    sleep 1
  done
  if (( verified == 0 )); then
    echo "$(date --iso-8601=seconds) ERROR object=$object new process/provenance failed" >> "$switch_log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object pid=$child_pid" >> "$switch_log"
done

echo "$(date --iso-8601=seconds) all historical-seed formal workers verified" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
