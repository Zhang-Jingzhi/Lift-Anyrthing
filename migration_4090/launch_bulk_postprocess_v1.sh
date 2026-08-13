#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
cd "$repo"

launch_post() {
  local name=$1
  shift
  test -z "$(screen -ls 2>/dev/null | grep "[.]${name}[[:space:]]" || true)"
  screen -dmS "$name" bash -lc "set -o errexit; $*"
  echo "launched $name"
}

launch_post bulk_post_g0 \
  "bash migration_4090/run_bulk_object_to_target.sh 0 baseline ycb+bleach_cleanser 100 20260820 bulk_base_g0 100; bash migration_4090/run_bulk_object_to_target.sh 0 baseline ycb+cracker_box 45 20260821 bulk_base_g0 100"
launch_post bulk_post_g1 \
  "bash migration_4090/run_bulk_object_to_target.sh 1 baseline ycb+pitcher_base 40 20260822 bulk_base_g1 100; bash migration_4090/run_bulk_object_to_target.sh 1 baseline contactdb+piggy_bank 65 20260823 bulk_base_g1 100"
launch_post bulk_post_g2 \
  "bash migration_4090/run_bulk_object_to_target.sh 2 baseline ycb+power_drill 170 20260824 bulk_base_g2 100"
launch_post bulk_post_g3 \
  "bash migration_4090/run_bulk_object_to_target.sh 3 baseline ycb+toy_airplane 65 20260825 bulk_base_g3 100"
launch_post bulk_post_g4 \
  "bash migration_4090/run_bulk_object_to_target.sh 4 bidex ycb+bleach_cleanser 260 20260830 bulk_bidex_g4 100; bash migration_4090/run_bulk_object_to_target.sh 4 bidex ycb+cracker_box 115 20260831 bulk_bidex_g4 100"
launch_post bulk_post_g5 \
  "bash migration_4090/run_bulk_object_to_target.sh 5 bidex ycb+pitcher_base 100 20260832 bulk_bidex_g5 100; bash migration_4090/run_bulk_object_to_target.sh 5 bidex contactdb+piggy_bank 165 20260833 bulk_bidex_g5 100"
launch_post bulk_post_g6 \
  "bash migration_4090/run_bulk_object_to_target.sh 6 bidex ycb+power_drill 435 20260834 bulk_bidex_g6 100"
launch_post bulk_post_g7 \
  "bash migration_4090/run_bulk_object_to_target.sh 7 bidex ycb+toy_airplane 160 20260835 bulk_bidex_g7 100"

screen -ls
