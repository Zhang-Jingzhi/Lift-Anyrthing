#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
log_root="$repo/migration_4090/logs/large_random_6x100_vhacd_high_v1"
gate="$repo/migration_4090/results/large_random_vhacd_high_v1_smoke/smoke_gate.json"
cd "$repo"

# Formal multi-GPU work is intentionally gated on the independently validated
# smoke result.  This launcher is not invoked automatically by the supervisor.
test -s "$gate"
mkdir -p "$log_root"

launch_queue() {
  local name=$1
  local gpu=$2
  local command=$3
  local log="$log_root/${name}.log"
  test -z "$(screen -ls 2>/dev/null | grep "[.]${name}[[:space:]]" || true)"
  screen -dmS "$name" bash -lc \
    "cd '$repo' && export CUDA_VISIBLE_DEVICES='$gpu' && { $command; } > >(tee -a '$log') 2>&1"
}

launch_queue lr_base_g0 0 \
  "migration_4090/run_large_random_base_queue_v1.sh 0 baseline ycb+bleach_cleanser && migration_4090/run_large_random_base_queue_v1.sh 0 baseline ycb+cracker_box"
launch_queue lr_base_g1 1 \
  "migration_4090/run_large_random_base_queue_v1.sh 1 baseline ycb+pitcher_base && migration_4090/run_large_random_base_queue_v1.sh 1 baseline contactdb+piggy_bank"
launch_queue lr_base_g2 2 \
  "migration_4090/run_large_random_base_queue_v1.sh 2 baseline ycb+power_drill"
launch_queue lr_base_g3 3 \
  "migration_4090/run_large_random_base_queue_v1.sh 3 baseline ycb+toy_airplane"
launch_queue lr_bidex_g4 4 \
  "migration_4090/run_large_random_base_queue_v1.sh 4 bidex_v3 ycb+bleach_cleanser && migration_4090/run_large_random_base_queue_v1.sh 4 bidex_v3 ycb+cracker_box"
launch_queue lr_bidex_g5 5 \
  "migration_4090/run_large_random_base_queue_v1.sh 5 bidex_v3 ycb+pitcher_base && migration_4090/run_large_random_base_queue_v1.sh 5 bidex_v3 contactdb+piggy_bank"
launch_queue lr_bidex_g6 6 \
  "migration_4090/run_large_random_base_queue_v1.sh 6 bidex_v3 ycb+power_drill"
launch_queue lr_bidex_g7 7 \
  "migration_4090/run_large_random_base_queue_v1.sh 7 bidex_v3 ycb+toy_airplane"

screen -ls
