#!/usr/bin/env bash
set -u
for spec in "0 0 5" "1 5 10" "2 10 15" "3 15 20" "4 20 25" "5 25 30" "6 30 33" "7 33 36"; do
  read -r gpu start end <<< "$spec"
  screen -dmS "xhand_candidate_gpu${gpu}" bash -lc "bash migration_4090/run_local_candidate_screen_v1.sh ${gpu} ${start} ${end} 2>&1 | tee migration_4090/logs/xhand_local_ik_v1/phys/gpu${gpu}.log"
done
screen -ls
