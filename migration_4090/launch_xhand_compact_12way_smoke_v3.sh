#!/usr/bin/env bash
set -euo pipefail

cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ROOT=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/results/xhand_compact_6x100_v3_smoke
LOG=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/logs/xhand_compact_6x100_v3_smoke
mkdir -p "$ROOT" "$LOG"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
    echo "already running: $name"
    return
  fi
  screen -L -Logfile "$LOG/${name}.log" -dmS "$name" bash -lc \
    "cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction; export XHAND_COMPACT_SMOKE_ROOT='$ROOT'; $*"
  echo "launched $name gpu=$gpu"
}

# Fresh seeds and a fresh output root isolate the final candidate logic from all
# pre-fix 20260812xx/42xx/72xx/82xx/92xx experiments.
launch xh6_v3_g0 0 \
  'for s in 2026101001 2026101101 2026101201; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 0 sphere baseline 4 "$s" 12 && break; done; for s in 2026101002 2026101102 2026101202; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 0 cube baseline 4 "$s" 12 && break; done'
launch xh6_v3_g1 1 \
  'for s in 2026101003 2026101103 2026101203; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 1 cracker baseline 4 "$s" 12 && break; done; for s in 2026101004 2026101104 2026101204; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 1 bleach baseline 4 "$s" 12 && break; done'
launch xh6_v3_g2 2 \
  'for s in 2026101005 2026101105 2026101205; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 2 pitcher baseline 4 "$s" 12 && break; done'
launch xh6_v3_g3 3 \
  'for s in 2026101006 2026101106 2026101206; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 3 drill baseline 4 "$s" 12 && break; done'
launch xh6_v3_g4 4 \
  'for s in 2026101011 2026101111 2026101211; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 4 sphere bidex_v3 4 "$s" 12 && break; done; for s in 2026101012 2026101112 2026101212; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 4 cube bidex_v3 4 "$s" 12 && break; done'
launch xh6_v3_g5 5 \
  'for s in 2026101013 2026101113 2026101213; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 5 cracker bidex_v3 4 "$s" 12 && break; done; for s in 2026101014 2026101114 2026101214; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 5 bleach bidex_v3 4 "$s" 12 && break; done'
launch xh6_v3_g6 6 \
  'for s in 2026101015 2026101115 2026101215; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 6 pitcher bidex_v3 4 "$s" 12 && break; done'
launch xh6_v3_g7 7 \
  'for s in 2026101016 2026101116 2026101216; do bash migration_4090/run_xhand_compact_smoke_combo_v2.sh 7 drill bidex_v3 4 "$s" 12 && break; done'

screen -ls
