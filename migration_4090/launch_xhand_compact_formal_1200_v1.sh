#!/usr/bin/env bash
set -euo pipefail

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
LOG_ROOT="$REPO/migration_4090/logs/xhand_compact_formal_1200_v1"
cd "$REPO"
mkdir -p "$LOG_ROOT"

launch() {
  local name=$1 gpu=$2 command=$3
  if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
    echo "already running: $name"
    return
  fi
  screen -L -Logfile "$LOG_ROOT/${name}.log" -dmS "$name" bash -lc \
    "cd '$REPO'; $command"
  echo "launched $name gpu=$gpu"
}

launch xhf_g0 0 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 0 sphere baseline 202608110100 && bash migration_4090/run_xhand_compact_formal_combo_v1.sh 0 cube baseline 202608110200"
launch xhf_g1 1 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 1 cracker baseline 202608110300 && bash migration_4090/run_xhand_compact_formal_combo_v1.sh 1 bleach baseline 202608110400"
launch xhf_g2 2 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 2 pitcher baseline 202608110500"
launch xhf_g3 3 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 3 drill baseline 202608110600"
launch xhf_g4 4 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 4 sphere bidex_v3 202608111100 && bash migration_4090/run_xhand_compact_formal_combo_v1.sh 4 cube bidex_v3 202608111200"
launch xhf_g5 5 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 5 cracker bidex_v3 202608111300 && bash migration_4090/run_xhand_compact_formal_combo_v1.sh 5 bleach bidex_v3 202608111400"
launch xhf_g6 6 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 6 pitcher bidex_v3 202608111500"
launch xhf_g7 7 \
  "bash migration_4090/run_xhand_compact_formal_combo_v1.sh 7 drill bidex_v3 202608111600"

screen -ls
