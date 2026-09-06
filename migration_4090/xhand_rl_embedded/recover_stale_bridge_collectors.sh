#!/usr/bin/env bash
set -u

# Active bridge processes were started before the collector-lock fix was
# loaded.  When one of those in-memory parents reaches collection it launches
# a child without --attempt-id while still holding the same flock, so the child
# cannot pass its own pre-AppLauncher lock.  Detect only that legacy signature,
# terminate the blocked child, and let the append-only reconcile watchdog
# restart the bridge from intact checkpoints with the corrected code.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
collection_lock="/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"
discovered_screen="xhand_formal_v5r_discovered_collect"
log="$root/logs/stale_bridge_collector_recovery.log"
discovered_log="$root/logs/discovered_hard_success_collectors/screen.log"
mkdir -p "$(dirname "$log")"
mkdir -p "$(dirname "$discovered_log")"

discovered_supervisor_active() {
  ps -ww -eo comm=,args= | awk '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.discovered_hard_success_collector/ {
      found=1; exit
    }
    END { exit(found ? 0 : 1) }
  '
}

discovered_reload_active() {
  ps -ww -eo comm=,args= | awk '
    ($1 == "bash") &&
    $0 ~ /reload_discovered_collector_after_pid\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '
}

all_six_strict_complete() {
  "$isaac_python" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
objects = ("sphere", "sphere_small", "cracker_large", "cracker", "pyramid", "cube")
for object_name in objects:
    accepted = False
    for receipt in (root / "accepted" / object_name).glob(
        "sample_*/strict_post_audit/receipt.json"
    ):
        try:
            accepted = json.loads(receipt.read_text()).get("accepted") is True
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if accepted:
            break
    if not accepted:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

recover_discovered_supervisor() {
  # A one-shot hot reload owns the intentional 90 s no-supervisor window.
  # Do not race it.  For an unplanned disappearance, provide the same full
  # Kit/driver cooldown, re-check, then start outside this parent screen by
  # removing STY.  The child is verified rather than trusting a screen socket.
  if all_six_strict_complete || discovered_supervisor_active || discovered_reload_active; then
    return
  fi
  echo "$(date --iso-8601=seconds) missing discovered supervisor; cooldown_s=90" >> "$log"
  sleep 90
  if all_six_strict_complete || discovered_supervisor_active || discovered_reload_active; then
    return
  fi
  screen -S "$discovered_screen" -X quit >/dev/null 2>&1 || true
  screen -wipe >/dev/null 2>&1 || true
  # This recovery loop already owns a persistent screen.  Keep the supervisor
  # as its background child, so it survives connection loss without creating
  # a nested screen that can die when its short-lived launcher exits.
  env PYTHONPATH="$repo" "$isaac_python" \
    -m migration_4090.xhand_rl_embedded.discovered_hard_success_collector \
    --root "$root" --repo "$repo" --isaac-python "$isaac_python" \
    --manifest "$manifest" --collection-lock "$collection_lock" \
    --num-envs 128 --max-steps 2000 --poll-seconds 30 \
    --kit-cooldown-seconds 75 --crash-backoff-seconds 180 --base-seed 2001 \
    >> "$discovered_log" 2>&1 &
  candidate_pid=$!
  for _ in $(seq 1 30); do
    if discovered_supervisor_active; then
      echo "$(date --iso-8601=seconds) recovered discovered supervisor pid=$candidate_pid" >> "$log"
      return
    fi
    sleep 1
  done
  echo "$(date --iso-8601=seconds) ERROR discovered supervisor recovery failed" >> "$log"
  kill -TERM "$candidate_pid" 2>/dev/null || true
}

while :; do
  while read -r pid ppid elapsed; do
    [[ -n "${pid:-}" && -r "/proc/$pid/cmdline" && -r "/proc/$ppid/cmdline" ]] || continue
    child_cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    parent_cmd="$(tr '\0' ' ' < "/proc/$ppid/cmdline")"
    [[ "$child_cmd" == *"migration_4090.xhand_rl_embedded.collect"* ]] || continue
    [[ "$parent_cmd" == *"migration_4090.xhand_rl_embedded.bridge_worker"* ]] || continue
    [[ "$child_cmd" != *"--attempt-id"* ]] || continue
    (( elapsed >= 180 )) || continue
    echo "$(date --iso-8601=seconds) terminate legacy deadlocked collector pid=$pid ppid=$ppid elapsed_s=$elapsed" \
      >> "$log"
    kill -TERM "$pid" 2>/dev/null || true
  done < <(ps -eo pid=,ppid=,etimes=)
  recover_discovered_supervisor
  sleep 30
done
