#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
cd "$repo"

launch_queue() {
  local name=$1
  local gpu=$2
  shift 2
  screen -dmS "$name" bash -lc "set -o errexit; $*"
  echo "launched $name on physical GPU $gpu"
}

# Four baseline queues. Every object/seed writes to its own directory.
launch_queue bulk_base_g0 0 \
  "bash migration_4090/run_bulk_baseline_object.sh 0 ycb+bleach_cleanser 100 20260820; bash migration_4090/run_bulk_baseline_object.sh 0 ycb+cracker_box 45 20260821"
launch_queue bulk_base_g1 1 \
  "bash migration_4090/run_bulk_baseline_object.sh 1 ycb+pitcher_base 40 20260822; bash migration_4090/run_bulk_baseline_object.sh 1 contactdb+piggy_bank 65 20260823"
launch_queue bulk_base_g2 2 \
  "bash migration_4090/run_bulk_baseline_object.sh 2 ycb+power_drill 170 20260824"
launch_queue bulk_base_g3 3 \
  "bash migration_4090/run_bulk_baseline_object.sh 3 ycb+toy_airplane 65 20260825"

# Four BiDex-style queues. Candidate/audit/physics outputs are also separated.
launch_queue bulk_bidex_g4 4 \
  "bash migration_4090/run_bulk_bidex_object.sh 4 ycb+bleach_cleanser 260 20260830; bash migration_4090/run_bulk_bidex_object.sh 4 ycb+cracker_box 115 20260831"
launch_queue bulk_bidex_g5 5 \
  "bash migration_4090/run_bulk_bidex_object.sh 5 ycb+pitcher_base 100 20260832; bash migration_4090/run_bulk_bidex_object.sh 5 contactdb+piggy_bank 165 20260833"
launch_queue bulk_bidex_g6 6 \
  "bash migration_4090/run_bulk_bidex_object.sh 6 ycb+power_drill 435 20260834"
launch_queue bulk_bidex_g7 7 \
  "bash migration_4090/run_bulk_bidex_object.sh 7 ycb+toy_airplane 160 20260835"

screen -ls
