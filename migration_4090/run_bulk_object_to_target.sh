#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 GPU METHOD OBJECT DENSITY INITIAL_SEED INITIAL_SCREEN TARGET [LOG_TAG]" >&2
  exit 2
fi
gpu=$1
method=$2
object=$3
density=$4
initial_seed=$5
initial_screen=$6
target=$7
log_tag=${8:-}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
slug=${object//+/_}
root="$repo/graph_exp/bimanual_data/bulk_irregular_${method}_v1/$slug"
final="$repo/graph_exp/bimanual_data/bulk_irregular_${method}_v1/final_100_each/$slug"
if [[ -n "$log_tag" ]]; then
  log="$repo/migration_4090/logs/bulk_target_${method}_${slug}_${log_tag}_gpu${gpu}.log"
else
  log="$repo/migration_4090/logs/bulk_target_${method}_${slug}_gpu${gpu}.log"
fi

cd "$repo"
exec > >(tee "$log") 2>&1
echo "Waiting for initial screen $initial_screen"
while screen -ls 2>/dev/null | grep -q "[.]${initial_screen}[[:space:]]"; do
  sleep 30
done
echo "Initial screen finished; validating output for $method $object"
test -f "$root/seed_$initial_seed/bimanual_dataset.pt"

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
    "$gpu" "$method" "$object" "$density" "$seed" "$select_count" 1
}

verify_seed "$initial_seed"
count=$(verified_count)
round=1
while [[ "$count" -lt "$target" && "$round" -le 12 ]]; do
  seed=$((initial_seed + round * 1000))
  echo "top-up round=$round seed=$seed verified=$count target=$target"
  if [[ "$method" == "baseline" ]]; then
    offsets=(2.8125 1.40625 4.21875 0.703125 2.109375 3.515625 4.921875)
    offset_index=$(((round - 1) % ${#offsets[@]}))
    roll_offset=${offsets[$offset_index]}
    echo "baseline roll_offset_degrees=$roll_offset"
    bash migration_4090/run_bulk_baseline_object.sh \
      "$gpu" "$object" "$density" "$seed" "$roll_offset"
  elif [[ "$method" == "bidex" ]]; then
    bash migration_4090/run_bulk_bidex_object.sh \
      "$gpu" "$object" "$density" "$seed"
  else
    echo "unknown method: $method" >&2
    exit 2
  fi
  verify_seed "$seed"
  count=$(verified_count)
  round=$((round + 1))
done

if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: only $count unique repeat-verified samples after 12 top-up rounds" >&2
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
