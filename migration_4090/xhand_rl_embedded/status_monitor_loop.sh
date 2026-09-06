#!/usr/bin/env bash
set -u

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
python_bin="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
snapshot="$root/status_snapshot.json"
log="$root/logs/status_monitor_live.log"

mkdir -p "$root/logs"
while :; do
  timestamp="$(date --iso-8601=seconds)"
  tmp="${snapshot}.tmp.$$"
  if env PYTHONPATH="$repo" "$python_bin" -m migration_4090.xhand_rl_embedded.status \
      --root "$root" --window-minutes 60 > "$tmp"; then
    # Atomic replacement prevents the browser from ever observing a partial
    # JSON document while six workers are writing event files concurrently.
    mv -f "$tmp" "$snapshot"
    printf '%s snapshot_updated\n' "$timestamp" >> "$log"
  else
    rm -f "$tmp"
    printf '%s status_command_failed\n' "$timestamp" >> "$log"
  fi
  sleep 60
done
