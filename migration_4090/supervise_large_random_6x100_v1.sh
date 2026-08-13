#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
tro_python=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
root="$repo/graph_exp/bimanual_data/large_random_6x100_v1"
status_root="$repo/migration_4090/results/large_random_6x100_vhacd_high_v1"
log_root="$repo/migration_4090/logs/large_random_6x100_vhacd_high_v1"
# The high-v2 piggy validation campaign used the 622... smoke family.  Resume
# the formal queues from a disjoint 6226... family so no interrupted high-v1
# seed directory is reused or overwritten.
formal_generation_seed_offset=${FORMAL_GENERATION_SEED_OFFSET:-602608050}
mkdir -p "$status_root" "$log_root"
cd "$repo"

queue_complete() {
  local name=$1
  case "$name" in
    lr_base_g0) test -s "$root/baseline/ycb_bleach_cleanser/final_100/bimanual_dataset.pt" && test -s "$root/baseline/ycb_cracker_box/final_100/bimanual_dataset.pt" ;;
    lr_base_g1) test -s "$root/baseline/ycb_pitcher_base/final_100/bimanual_dataset.pt" && test -s "$root/baseline/contactdb_piggy_bank/final_100/bimanual_dataset.pt" ;;
    lr_base_g2) test -s "$root/baseline/ycb_power_drill/final_100/bimanual_dataset.pt" ;;
    lr_base_g3) test -s "$root/baseline/ycb_toy_airplane/final_100/bimanual_dataset.pt" ;;
    lr_bidex_g4) test -s "$root/bidex_v3/ycb_bleach_cleanser/final_100/bimanual_dataset.pt" && test -s "$root/bidex_v3/ycb_cracker_box/final_100/bimanual_dataset.pt" ;;
    lr_bidex_g5) test -s "$root/bidex_v3/ycb_pitcher_base/final_100/bimanual_dataset.pt" && test -s "$root/bidex_v3/contactdb_piggy_bank/final_100/bimanual_dataset.pt" ;;
    lr_bidex_g6) test -s "$root/bidex_v3/ycb_power_drill/final_100/bimanual_dataset.pt" ;;
    lr_bidex_g7) test -s "$root/bidex_v3/ycb_toy_airplane/final_100/bimanual_dataset.pt" ;;
    *) return 1 ;;
  esac
}

queue_command() {
  case "$1" in
    lr_base_g0)
      if test -s "$root/baseline/ycb_cracker_box/final_100/bimanual_dataset.pt"; then
        # Prefer the append-only repair pool when it exists.  The previous
        # one-shot pool was geometrically empty for bleach size001 and would
        # otherwise keep a resumed worker in a 0/375 loop.
        if test -s "$repo/migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_001_transfer_adaptive_v99.pt"; then
          echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=0 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_001_transfer_adaptive_v99.pt' migration_4090/run_large_random_base_queue_v1.sh 0 baseline ycb+bleach_cleanser"
        else
          echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=1 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_000_transfer_adaptive_v5.pt' migration_4090/run_large_random_base_queue_v1.sh 0 baseline ycb+bleach_cleanser"
        fi
      else
          echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=0 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_cracker_size000_verified_local_v7.pt' migration_4090/run_large_random_base_queue_v1.sh 0 baseline ycb+cracker_box"
      fi
      ;;
    lr_base_g1)
      if test -s "$root/baseline/contactdb_piggy_bank/final_100/bimanual_dataset.pt"; then
        echo "migration_4090/run_large_random_base_queue_v1.sh 1 baseline ycb+pitcher_base"
      else
        echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=0 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_contactdb_piggy_bank_size_000_transfer_adaptive_v17.pt' migration_4090/run_large_random_base_queue_v1.sh 1 baseline contactdb+piggy_bank"
      fi
      ;;
    lr_base_g2) echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=0 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_ycb_power_drill_size_000_transfer_adaptive_v11.pt' migration_4090/run_large_random_base_queue_v1.sh 2 baseline ycb+power_drill" ;;
    # Toy-airplane size002 had entered a no-progress loop on a reused source
    # pool.  Let the first transfer pool seed the queue, then generate fresh
    # baseline candidates on subsequent attempts if it does not produce strict
    # samples.
    lr_base_g3) echo "BASELINE_REUSE_PRECOMPUTED_SOURCE=0 BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE='$repo/migration_4090/large_random_v1_candidates/baseline_toy_size000_verified_outward_v15.pt' migration_4090/run_large_random_base_queue_v1.sh 3 baseline ycb+toy_airplane" ;;
    lr_bidex_g4)
      if test -s "$root/bidex_v3/ycb_cracker_box/final_100/bimanual_dataset.pt"; then
        echo "BIDEX_REUSE_FINAL_SOURCE=0 BIDEX_ONE_SHOT_FINAL_SOURCE='$repo/migration_4090/large_random_v1_candidates/bidex_bleach_size000_verified_parent0_local_v1.pt' migration_4090/run_large_random_base_queue_v1.sh 4 bidex_v3 ycb+bleach_cleanser"
      else
        echo "BIDEX_REUSE_FINAL_SOURCE=0 BIDEX_ONE_SHOT_FINAL_SOURCE='$repo/migration_4090/large_random_v1_candidates/bidex_cracker_size000_verified_micro_v7.pt' migration_4090/run_large_random_base_queue_v1.sh 4 bidex_v3 ycb+cracker_box"
      fi
      ;;
    lr_bidex_g5)
      if test -s "$root/bidex_v3/contactdb_piggy_bank/final_100/bimanual_dataset.pt"; then
        echo "migration_4090/run_large_random_base_queue_v1.sh 5 bidex_v3 ycb+pitcher_base"
      else
        echo "BIDEX_REUSE_FINAL_SOURCE=0 BIDEX_ONE_SHOT_FINAL_SOURCE='$repo/migration_4090/large_random_v1_candidates/bidex_piggy_size000_repeat_verified_highv2_local_v15.pt' migration_4090/run_large_random_base_queue_v1.sh 5 bidex_v3 contactdb+piggy_bank"
      fi
      ;;
    lr_bidex_g6) echo "BIDEX_REUSE_FINAL_SOURCE=0 BIDEX_ONE_SHOT_FINAL_SOURCE='$repo/migration_4090/large_random_v1_candidates/bidex_power_size000_verified_local_v8.pt' migration_4090/run_large_random_base_queue_v1.sh 6 bidex_v3 ycb+power_drill" ;;
    # Toy-airplane size001 previously entered a no-progress loop when the
    # same adaptive pool was reused after retaining zero strict samples.
    # Keep the first source as a seed, but force later attempts to generate
    # fresh BiDex candidates from source_vis; append-only adaptive pools are
    # still selected by run_large_random_base_queue_v1.sh.
    lr_bidex_g7) echo "BIDEX_REUSE_FINAL_SOURCE=0 BIDEX_ONE_SHOT_FINAL_SOURCE='$repo/migration_4090/large_random_v1_candidates/bidex_toy_size000_verified_micro_v15.pt' migration_4090/run_large_random_base_queue_v1.sh 7 bidex_v3 ycb+toy_airplane" ;;
  esac
}

declare -A gpu=(
  [lr_base_g0]=0 [lr_base_g1]=1 [lr_base_g2]=2 [lr_base_g3]=3
  [lr_bidex_g4]=4 [lr_bidex_g5]=5 [lr_bidex_g6]=6 [lr_bidex_g7]=7
)
queues=(lr_base_g0 lr_base_g1 lr_base_g2 lr_base_g3 lr_bidex_g4 lr_bidex_g5 lr_bidex_g6 lr_bidex_g7)

while true; do
  timestamp=$(date '+%Y%m%d_%H%M%S')
  latest="$status_root/progress_latest.json"
  "$tro_python" migration_4090/summarize_large_random_6x100_v1.py \
    --root "$root" --output "$latest" >/dev/null
  cp "$latest" "$status_root/progress_$timestamp.json"
  health_tmp="$status_root/health_latest.txt.tmp"
  {
    date --iso-8601=seconds
    screen -ls 2>&1 || true
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader 2>&1 || true
  } > "$health_tmp"
  mv "$health_tmp" "$status_root/health_latest.txt"
  if "$tro_python" -c "import json; assert json.load(open('$latest'))['complete']" 2>/dev/null; then
    "$tro_python" migration_4090/summarize_large_random_6x100_v1.py \
      --root "$root" --strict --output "$status_root/final_summary.json"
    echo "$(date --iso-8601=seconds) FORMAL_6X100_BOTH_METHODS_COMPLETE"
    exit 0
  fi

  for name in "${queues[@]}"; do
    if queue_complete "$name"; then
      continue
    fi
    if ! screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; then
      command=$(queue_command "$name")
      restart_log="$log_root/${name}_restart_$timestamp.log"
      echo "$(date --iso-8601=seconds) restarting $name gpu=${gpu[$name]}"
      screen -dmS "$name" bash -lc \
        "cd '$repo' && export CUDA_VISIBLE_DEVICES='${gpu[$name]}' && export FORMAL_GENERATION_SEED_OFFSET='$formal_generation_seed_offset' && { $command; } > >(tee '$restart_log') 2>&1"
    fi
  done
  sleep 60
done
