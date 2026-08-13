#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU DENSITY TARGET" >&2
  exit 2
fi
gpu=$1
density=$2
target=$3
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
object=ycb+toy_airplane
slug=ycb_toy_airplane
method=bidex
root="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/$slug"
final="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/final_100_each/$slug"
log="$repo/migration_4090/logs/bulk_target_bidex_${slug}_diverse_gpu${gpu}.log"

test ! -e "$log"
cd "$repo"
exec > >(tee "$log") 2>&1

verified_count() {
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
}

verify_seed() {
  local seed=$1
  local current remaining select_count
  if [[ -f "$root/seed_$seed/repeat_verified/verified_dataset.pt" ]]; then
    echo "seed $seed already repeat-verified"
    return
  fi
  current=$(verified_count)
  remaining=$((target - current))
  if [[ "$remaining" -le 0 ]]; then
    return
  fi
  select_count=$((remaining * 3 / 2 + 12))
  if [[ "$select_count" -gt 160 ]]; then select_count=160; fi
  echo "repeat-verifying seed=$seed current=$current remaining=$remaining select=$select_count"
  bash migration_4090/repeat_verify_bulk_seed.sh \
    "$gpu" "$method" "$object" "$density" "$seed" "$select_count" 1 12
}

count=$(verified_count)
round=0
while [[ "$count" -lt "$target" && "$round" -lt 20 ]]; do
  seed=$((20266835 + round * 1000))
  echo "diverse top-up round=$round seed=$seed verified=$count target=$target"
  bash migration_4090/run_bulk_bidex_toy_diverse_object.sh \
    "$gpu" "$density" "$seed"
  verify_seed "$seed"
  count=$(verified_count)
  round=$((round + 1))
done

if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: only $count unique repeat-verified samples after $round rounds" >&2
  exit 1
fi
test ! -e "$final"
"$conda_root/envs/tro/bin/python" migration_4090/assemble_bulk_object.py \
  --input-root "$root" \
  --object-name "$object" \
  --method "$method" \
  --target "$target" \
  --output-dir "$final"
echo "COMPLETE method=$method object=$object count=$target final=$final"
