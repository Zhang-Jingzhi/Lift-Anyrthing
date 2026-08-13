#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
gpu=6
object=ycb+power_drill
density=220
initial_seed=20995824
target=100
offsets=(2.8125 1.40625 4.21875 0.703125 2.109375 3.515625 4.921875)
root="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/ycb_power_drill"
final="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/final_100_each/ycb_power_drill"
cd "$repo"

while screen -ls 2>/dev/null | grep -q '[.]baseline_drill_diverse_full[[:space:]]'; do
  echo "waiting for baseline_drill_diverse_full"
  sleep 30
done
test -f "$root/seed_$initial_seed/bimanual_dataset.pt"

verified_count() {
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
}

strict_count() {
  "$conda_root/envs/tro/bin/python" - "$1" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(len(payload.get("samples", [])))
PY
}

verify_seed() {
  local seed=$1
  local dataset="$root/seed_$seed/bimanual_dataset.pt"
  local strict current remaining select_count
  if [[ -f "$root/seed_$seed/repeat_verified/verified_dataset.pt" ]]; then
    return
  fi
  strict=$(strict_count "$dataset")
  if [[ "$strict" -eq 0 ]]; then
    echo "seed=$seed has no strict candidate; skipping repeat verification"
    return
  fi
  current=$(verified_count)
  remaining=$((target - current))
  if [[ "$remaining" -le 0 ]]; then return; fi
  select_count=$((remaining * 3 / 2 + 12))
  if [[ "$select_count" -gt 160 ]]; then select_count=160; fi
  echo "repeat-verifying seed=$seed strict=$strict current=$current select=$select_count"
  bash migration_4090/repeat_verify_bulk_seed.sh \
    "$gpu" baseline "$object" "$density" "$seed" "$select_count" 1 4
}

verify_seed "$initial_seed"
count=$(verified_count)
round=1
while [[ "$count" -lt "$target" && "$round" -le 12 ]]; do
  seed=$((initial_seed + round * 1000))
  offset_index=$(((round - 1) % ${#offsets[@]}))
  roll_offset=${offsets[$offset_index]}
  echo "adaptive top-up round=$round seed=$seed verified=$count roll_offset=$roll_offset"
  bash migration_4090/run_bulk_baseline_object_custom.sh \
    "$gpu" "$object" "$density" "$seed" 60 20 32 4 3 "$roll_offset"
  verify_seed "$seed"
  count=$(verified_count)
  round=$((round + 1))
done

if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: baseline drill only $count repeat-verified samples" >&2
  exit 1
fi
test ! -e "$final"
"$conda_root/envs/tro/bin/python" migration_4090/assemble_bulk_object.py \
  --input-root "$root" \
  --object-name "$object" \
  --method baseline \
  --target "$target" \
  --output-dir "$final"
echo "COMPLETE method=baseline object=$object count=$target final=$final"
