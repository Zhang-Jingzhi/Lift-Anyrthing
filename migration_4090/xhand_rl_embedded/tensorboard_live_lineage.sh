#!/usr/bin/env bash
set -u

# Keep the TensorBoard port focused on the currently selected append-only
# lineage.  The historical runs remain on disk and can still be opened from a
# separate TensorBoard instance, but they must not be overlaid on the live PPO
# curves shown to the operator.
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
python_bin="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
tensorboard_bin="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/tensorboard"
curated_root="$root/visualizations/tensorboard_live_curated"
port="8083"
child=""
last_spec=""

cleanup() {
  if [[ -n "$child" ]]; then
    kill "$child" >/dev/null 2>&1 || true
    wait "$child" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

while :; do
  # Re-emit only strict-prefix/gate scalars with monotonic visualization
  # steps.  Raw event files stay immutable under runs/.
  env PYTHONPATH="$repo" "$python_bin" -m migration_4090.xhand_rl_embedded.curate_tensorboard_events \
    --snapshot "$root/status_snapshot.json" --output "$curated_root" \
    >/tmp/xhand_tb_curate.log 2>&1 || true
  spec="$($python_bin - "$root/status_snapshot.json" "$curated_root" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text())
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    payload = {}
curated_root = Path(sys.argv[2]).resolve()
parts = []
for name, worker in sorted(payload.get("workers", {}).items()):
    # Only include the new mesh-aware lineage.  During its Isaac startup the
    # event file may not exist yet; it will appear on the next refresh.
    if worker.get("nominal_lineage") not in {"meshaware_v3", "meshaware_v4", "meshaware_v4b"}:
        continue
    # A stopped worker can still have a newer curriculum event file than its
    # last formal retry.  Showing that historical curve on the live port was
    # the main source of the "formal PPO is progressing" misread.  Omit stale
    # stopped cards; the 8766 page still reports their last prefix and retry
    # provenance explicitly.
    if worker.get("stage") == "stopped_waiting_reconcile" and not worker.get("process_live"):
        continue
    directory = curated_root / name
    if not directory.is_dir() or not list(directory.glob("events.out.tfevents.*")):
        continue
    parts.append(f"{name}:{directory}")
print(",".join(parts))
PY
  )"
  if [[ -z "$spec" ]]; then
    # Do not leave a dead TensorBoard child serving the last retry after all
    # current PPO workers have stopped.  That stale curve was easy to read as
    # live training on port 8083, especially during a GPU/driver failure.
    if [[ -n "$last_spec" ]]; then
      cleanup
      last_spec=""
      printf '%s no_live_lineage\n' "$(date --iso-8601=seconds)" >> "$root/logs/tensorboard_live_lineage.log"
    fi
    sleep 20
    continue
  fi
  if [[ "$spec" != "$last_spec" ]]; then
    cleanup
    "$tensorboard_bin" --logdir_spec "$spec" --host 127.0.0.1 --port "$port" \
      --window_title "XHand RL v5r · current PPO lineage · strict prefix" \
      --reload_interval 30 --max_reload_threads 2 >/tmp/xhand_tb_live_lineage.log 2>&1 &
    child=$!
    last_spec="$spec"
  fi
  sleep 30
done
