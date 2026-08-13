#!/usr/bin/env bash
set -u -o pipefail

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
ROOT="$REPO/graph_exp/bimanual_data/xhand_compact_formal_1200_v1"
LOG="$REPO/migration_4090/logs/xhand_compact_formal_1200_v1/supervisor.log"
cd "$REPO"
mkdir -p "$(dirname "$LOG")"

while true; do
  COMPLETE=$(find "$ROOT" -type f -path '*/final_100/bimanual_dataset.pt' 2>/dev/null | wc -l)
  ACCEPTED=$(find "$ROOT" -type f -path '*/accepted/sample_*.json' 2>/dev/null | wc -l)
  echo "$(date '+%F %T') supervisor complete=$COMPLETE/12 accepted=$ACCEPTED/1200" | tee -a "$LOG"
  if (( COMPLETE == 12 && ACCEPTED == 1200 )); then
    echo "$(date '+%F %T') FORMAL_1200_COMPLETE" | tee -a "$LOG"
    exit 0
  fi
  bash migration_4090/launch_xhand_compact_formal_1200_v1.sh >> "$LOG" 2>&1 || true
  sleep 300
done
