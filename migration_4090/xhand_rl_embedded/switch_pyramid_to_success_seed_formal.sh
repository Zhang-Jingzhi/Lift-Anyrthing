#!/usr/bin/env bash
set -euo pipefail

# Replace only the currently regressed 200% pyramid curriculum process with
# an append-only formal lineage seeded by the immutable curriculum checkpoint
# that actually observed 15 hard successes.  No old attempt or checkpoint is
# removed.  The other five object workers are never touched.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
seed="$root/runs/pyramid/curriculum/model_600.pt"
expected_seed_sha256="6de8a01346daa411d4d4bc4dcf352dcfadaab70792d24d8b15d0739815dc15b6"
bridge_screen="xhand_formal_v5r_pyramid_bridge"
watchdog_screen="xhand_formal_v5r_watchdog"
switch_log="$root/logs/pyramid_success_seed_switch.log"

mkdir -p "$root/logs/pyramid"

restart_watchdog() {
  if ps -ww -eo comm=,args= | awk '
    ($1 == "bash") && $0 ~ /reconcile_target100\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    return
  fi
  # The switch already owns a persistent screen.  Replacing this shell with
  # the reconciler is more reliable than spawning a nested screen immediately
  # before the parent exits (some screen builds leave a dead socket there).
  echo "$(date --iso-8601=seconds) switch screen taking over as watchdog" >> "$switch_log"
  exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
}
trap restart_watchdog EXIT

actual_seed_sha256="$(sha256sum "$seed" | awk '{print $1}')"
if [[ "$actual_seed_sha256" != "$expected_seed_sha256" ]]; then
  echo "$(date --iso-8601=seconds) ERROR seed hash mismatch actual=$actual_seed_sha256" \
    >> "$switch_log"
  exit 1
fi
"$isaac_python" - "$root/runs/pyramid/curriculum/CURRICULUM_TRAINING_COMPLETE.json" "$seed" <<'PY'
import json
import sys
from pathlib import Path

marker = json.loads(Path(sys.argv[1]).read_text())
seed = Path(sys.argv[2]).resolve()
assert Path(marker["checkpoint"]).resolve() == seed
assert marker.get("curriculum_success_serial_count", 0) == 15
assert marker["checkpoint_selection"]["training_hard_success_observed"] is True
assert marker["checkpoint_selection"]["training_hard_success_iteration"] == 587
PY

echo "$(date --iso-8601=seconds) stop regressed pyramid curriculum" >> "$switch_log"
screen -S "$watchdog_screen" -X quit >/dev/null 2>&1 || true
screen -S "$bridge_screen" -X quit >/dev/null 2>&1 || true

for _ in $(seq 1 90); do
  if ! ps -ww -eo comm=,args= | awk '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_(curriculum\.train|embedded\.(bridge_worker|train|collect))/ &&
    $0 ~ /--object[ =]+pyramid([ =]|$)/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    break
  fi
  sleep 1
done
if ps -ww -eo comm=,args= | awk '
  ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
  $0 ~ /migration_4090\.xhand_rl_(curriculum\.train|embedded\.(bridge_worker|train|collect))/ &&
  $0 ~ /--object[ =]+pyramid([ =]|$)/ { found=1; exit }
  END { exit(found ? 0 : 1) }
'; then
  echo "$(date --iso-8601=seconds) ERROR old pyramid process did not stop" >> "$switch_log"
  exit 1
fi

# Isaac/PhysX releases GPU and Vulkan resources asynchronously after process
# exit.  Use the same validated cooldown as the production reconciler.
sleep 90
screen -wipe >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) launch formal from success seed=$seed" >> "$switch_log"
env -u STY screen -L -Logfile "$root/logs_pyramid_bridge_success_seed.log" \
  -dmS "$bridge_screen" \
  env PYTHONPATH="$repo" "$isaac_python" \
  -m migration_4090.xhand_rl_embedded.bridge_worker \
  --manifest "$manifest" --object pyramid --nominal "$nominal" \
  --root "$root" --isaac-python "$isaac_python" --device cuda:5 \
  --target 100 --curriculum-iterations 400 --formal-iterations 4000 \
  --curriculum-envs 128 --formal-envs 128 --collect-envs 128 \
  --skip-curriculum --prefer-seed-checkpoint --seed-checkpoint "$seed" \
  --formal-seed 93 --residual-activation-phase 1 \
  --kit-portable-root /tmp/xhand_rl_embedded_kit/v5r/pyramid_success_seed

verified=0
for _ in $(seq 1 60); do
  if ps -ww -eo comm=,args= | awk '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.bridge_worker/ &&
    $0 ~ /--object[ =]+pyramid([ =]|$)/ &&
    $0 ~ /--skip-curriculum/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then
    verified=1
    break
  fi
  sleep 1
done
if (( verified == 0 )); then
  echo "$(date --iso-8601=seconds) ERROR success-seed pyramid bridge failed to start" \
    >> "$switch_log"
  exit 1
fi

echo "$(date --iso-8601=seconds) verified success-seed pyramid bridge" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
