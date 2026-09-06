#!/usr/bin/env bash
set -euo pipefail

# Roll the already-running formal retries onto the exact-success capture code.
# Every process resumes from its newest numbered checkpoint; no checkpoint,
# accepted sample, or completed attempt is removed or replaced.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
watchdog_screen="xhand_formal_v5r_watchdog"
log="$root/logs/exact_capture_roll_reload.log"

declare -A outputs=(
  [sphere_small]="$root/runs/sphere_small/training_retry_010"
  [cracker_large]="$root/runs/cracker_large/training_retry_008"
  [cracker]="$root/runs/cracker/training_retry_008"
  [cube]="$root/runs/cube/training_retry_013"
)
declare -A nominals=(
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"
  [cracker_large]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cube.pt"
)
declare -A devices=(
  [sphere_small]="cuda:1"
  [cracker_large]="cuda:2"
  [cracker]="cuda:3"
  [cube]="cuda:4"
)
declare -A max_iterations=(
  [sphere_small]="4349"
  [cracker_large]="4399"
  [cracker]="4399"
  [cube]="4399"
)

mkdir -p "$root/logs"

restart_watchdog() {
  if ! ps -ww -eo comm=,args= | awk '
    ($1 == "bash") && $0 ~ /reconcile_target100\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    screen -wipe >/dev/null 2>&1 || true
    screen -dmS "$watchdog_screen" bash -lc \
      "exec '$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh' >> '$root/logs/reconcile_target100.log' 2>&1"
  fi
}
trap restart_watchdog EXIT

screen -S "$watchdog_screen" -X quit >/dev/null 2>&1 || true

objects=("$@")
if (( ${#objects[@]} == 0 )); then
  objects=(sphere_small cracker_large cracker cube)
fi

for object in "${objects[@]}"; do
  if [[ -z "${outputs[$object]:-}" ]]; then
    echo "unknown reload object: $object" >&2
    exit 2
  fi
  output="${outputs[$object]}"
  screen_name="xhand_formal_v5r_${object}_bridge"
  echo "$(date --iso-8601=seconds) stop object=$object" >> "$log"
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true

  # Screen normally forwards SIGHUP to the worker.  Refuse to start a second
  # Kit instance until the old object-specific trainer has fully disappeared.
  for _ in $(seq 1 60); do
    if ! ps -ww -eo comm=,args= | awk -v object="$object" '
      ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
      $0 ~ /migration_4090\.xhand_rl_embedded\.train/ &&
      $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
      END { exit(found ? 0 : 1) }
    '; then
      break
    fi
    sleep 1
  done
  if ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.train/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    echo "$(date --iso-8601=seconds) ERROR trainer_did_not_stop object=$object" >> "$log"
    exit 1
  fi

  sleep 90
  checkpoint="$($isaac_python - "$output" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("model_*.pt"):
    try:
        rows.append((int(path.stem.rsplit("_", 1)[1]), path.resolve()))
    except ValueError:
        continue
if not rows:
    raise SystemExit("no numbered checkpoint")
print(max(rows)[1])
PY
)"
  iteration="${checkpoint%.pt}"
  iteration="${iteration##*_}"
  echo "$(date --iso-8601=seconds) resume object=$object checkpoint=$checkpoint iteration=$iteration" >> "$log"

  screen -dmS "$screen_name" bash -lc \
    "exec env PYTHONPATH='$repo' '$isaac_python' -m migration_4090.xhand_rl_embedded.train \
    --manifest '$manifest' --object '$object' --nominal-dataset '${nominals[$object]}' \
    --output '$output' --num-envs 128 --max-iterations '${max_iterations[$object]}' \
    --seed 84 --device '${devices[$object]}' --headless \
    --kit-portable-root '/tmp/xhand_rl_embedded_kit/v5r/$object/formal' \
    --penetration-reward-weight -12.0 --terminal-success-weight 2500.0 \
    --gamma 0.999 --init-noise-std 0.04 --entropy-coef 0.0001 \
    --resume-checkpoint '$checkpoint' --residual-activation-phase 1 \
    --instantaneous-residual-weight 0.0 --residual-integration 0.02 \
    --arm-action-scale-rad 0.08 --hand-action-scale-rad 0.20 --freeze-policy-noise \
    >> '$root/logs/$object/formal_training.log' 2>&1"

  verified=0
  for _ in $(seq 1 120); do
    if ps -ww -eo comm=,args= | awk -v object="$object" '
      ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
      $0 ~ /migration_4090\.xhand_rl_embedded\.train/ &&
      $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
      END { exit(found ? 0 : 1) }
    ' && [[ -f "$output/training_provenance.json" ]] && \
      "$isaac_python" - "$output/training_provenance.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload.get("exact_training_rollout_capture") is True
assert payload.get("exact_checkpoint_timing") == "before_ppo_update"
assert payload.get("training_trajectory_recording") is True
PY
    then
      verified=1
      break
    fi
    sleep 1
  done
  if (( verified == 0 )); then
    echo "$(date --iso-8601=seconds) ERROR live_capture_provenance_missing object=$object" >> "$log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object" >> "$log"
  sleep 75
done

echo "$(date --iso-8601=seconds) complete" >> "$log"
