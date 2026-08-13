#!/usr/bin/env bash
set -u -o pipefail

cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
LOG_ROOT=migration_4090/logs

while :; do
  active=0
  complete=0
  echo "WATCH $(date '+%F %T')"
  for gpu in 0 1 2 3 4 5 6 7; do
    name="xhand_ext_worker_gpu${gpu}"
    log="$LOG_ROOT/xhand_external_candidate_worker_gpu${gpu}.log"
    if grep -q '^WORKER_DONE ' "$log" 2>/dev/null; then
      complete=$((complete + 1))
      continue
    fi
    if screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
      active=$((active + 1))
      continue
    fi
    echo "RESTART gpu=$gpu"
    screen -dmS "$name" bash -lc \
      "cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction && bash migration_4090/run_external_xhand_candidate_worker.sh $gpu $gpu 8 >> '$log' 2>&1"
    active=$((active + 1))
  done
  echo "STATE active=$active complete=$complete"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  [[ "$complete" -eq 8 ]] && break
  sleep 60
done
echo "WATCHDOG_DONE $(date '+%F %T')"
