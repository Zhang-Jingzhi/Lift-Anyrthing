#!/usr/bin/env bash
set -euo pipefail

# One-shot hot reload after the currently running collector attempt finishes.
# This preserves the active rollout, then reloads the dynamic supervisor so
# later attempts pick up newly validated diagnostics/std-scale code.

pid="${1:?collector pid is required}"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
lock="/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"
supervisor_screen="xhand_formal_v5r_discovered_collect"
reload_log="$root/logs/discovered_hard_success_collectors/reload.log"

if [[ -r "/proc/$pid/cmdline" ]]; then
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *"migration_4090.xhand_rl_embedded.collect"* ]] || exit 1
fi
while kill -0 "$pid" 2>/dev/null; do
  sleep 15
done

# The old supervisor may already have started its next child.  Close that
# screen as one process group, then provide a full driver/Kit quiescence gap.
screen -S "$supervisor_screen" -X quit >/dev/null 2>&1 || true
sleep 90
screen -wipe >/dev/null 2>&1 || true
mkdir -p "$root/logs/discovered_hard_success_collectors"

# This one-shot reloader itself normally runs inside a detached screen.  Reuse
# that persistent screen by replacing the shell with the supervisor; spawning
# a nested screen immediately before the parent exits can leave a dead socket.
echo "$(date --iso-8601=seconds) launch supervisor=$supervisor_screen" >> "$reload_log"
exec env PYTHONPATH="$repo" "$isaac_python" \
  -m migration_4090.xhand_rl_embedded.discovered_hard_success_collector \
  --root "$root" --repo "$repo" --isaac-python "$isaac_python" --manifest "$manifest" \
  --collection-lock "$lock" --num-envs 128 --max-steps 2000 --poll-seconds 30 \
  --kit-cooldown-seconds 75 --crash-backoff-seconds 180 --base-seed 2001
