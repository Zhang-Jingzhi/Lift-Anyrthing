#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
tro_python=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
baseline="$repo/graph_exp/bimanual_data/large_random_v1_smoke/baseline/pitcher003_v1/repeat_verified/verified_dataset.pt"
gate="$repo/migration_4090/results/large_random_v1_smoke/smoke_gate.json"
status="$repo/migration_4090/results/large_random_v1_smoke/supervisor_status.txt"
cd "$repo"

test -f "$baseline"
test ! -e "$gate"

wait_for_screen() {
  local name=$1
  while screen -ls 2>/dev/null | grep -q "[.]${name}[[:space:]]"; do
    printf '%s waiting screen=%s\n' "$(date --iso-8601=seconds)" "$name" > "$status"
    sleep 20
  done
}

validate_attempt() {
  local tag=$1
  local dataset="$repo/graph_exp/bimanual_data/large_random_v1_smoke/bidex_v3/$tag/repeat_verified/verified_dataset.pt"
  if [[ ! -s "$dataset" ]]; then
    return 1
  fi
  "$tro_python" migration_4090/validate_large_random_smoke_gate.py \
    --baseline "$baseline" \
    --bidex "$dataset" \
    --output "$gate"
}

wait_for_screen lr_smoke_bidex_p003_v2
if validate_attempt pitcher003_v2; then
  printf '%s gate_pass tag=pitcher003_v2\n' "$(date --iso-8601=seconds)" > "$status"
  exit 0
fi

for attempt in 3 4 5; do
  tag="pitcher003_v${attempt}"
  seed=$((20261002 + attempt))
  log="$repo/migration_4090/logs/large_random_v1_smoke_bidex_v3_${tag}.log"
  printf '%s launching tag=%s seed=%s\n' "$(date --iso-8601=seconds)" "$tag" "$seed" > "$status"
  if BIDEX_REGION_PAIRS=24 \
     BIDEX_MAX_NORMAL_OPPOSITION_COSINE=-0.80 \
     BIDEX_MIN_RADIAL_NORMAL_ALIGNMENT=0.30 \
     BIDEX_STOP_AFTER_CANDIDATES=12 \
     BIDEX_CANDIDATE_SEED="$seed" \
     migration_4090/run_large_random_smoke_v1.sh bidex_v3 4 "$tag" \
       > >(tee "$log") 2>&1; then
    if validate_attempt "$tag"; then
      printf '%s gate_pass tag=%s\n' "$(date --iso-8601=seconds)" "$tag" > "$status"
      exit 0
    fi
  fi
  printf '%s attempt_failed tag=%s\n' "$(date --iso-8601=seconds)" "$tag" > "$status"
done

printf '%s blocked all_bidex_smoke_attempts_failed\n' "$(date --iso-8601=seconds)" > "$status"
exit 1
