#!/usr/bin/env bash
set -euo pipefail

cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
LOG=migration_4090/logs/xhand_compact_6x100_v2_smoke
mkdir -p "$LOG"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
    echo "already running: $name"
    return
  fi
  screen -L -Logfile "$LOG/${name}.log" -dmS "$name" bash -lc \
    "cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction; $*"
  echo "launched $name gpu=$gpu"
}

# Each combo gets up to three independent 12-candidate banks.  A queue moves
# to its next object only after one candidate passes three strict rollouts.
launch xh6_smoke_r3_g0 0 \
  'for s in 2026087201 2026088201 2026089201; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 0 sphere baseline 4 "$s" 12 && break; done; for s in 2026087202 2026088202 2026089202; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 0 cube baseline 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g1 1 \
  'for s in 2026087203 2026088203 2026089203; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 1 cracker baseline 4 "$s" 12 && break; done; for s in 2026087204 2026088204 2026089204; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 1 bleach baseline 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g2 2 \
  'for s in 2026087205 2026088205 2026089205; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 2 pitcher baseline 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g3 3 \
  'for s in 2026087206 2026088206 2026089206; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 3 drill baseline 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g4 4 \
  'for s in 2026087211 2026088211 2026089211; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 4 sphere bidex_v3 4 "$s" 12 && break; done; for s in 2026087212 2026088212 2026089212; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 4 cube bidex_v3 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g5 5 \
  'for s in 2026087213 2026088213 2026089213; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 5 cracker bidex_v3 4 "$s" 12 && break; done; for s in 2026087214 2026088214 2026089214; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 5 bleach bidex_v3 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g6 6 \
  'for s in 2026087215 2026088215 2026089215; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 6 pitcher bidex_v3 4 "$s" 12 && break; done'
launch xh6_smoke_r3_g7 7 \
  'for s in 2026087216 2026088216 2026089216; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 7 drill bidex_v3 4 "$s" 12 && break; done'

screen -ls
