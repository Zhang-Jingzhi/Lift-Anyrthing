#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
gpu=3
object=ycb+toy_airplane
density=65
target=100
root="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/ycb_toy_airplane"
final="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/final_100_each/ycb_toy_airplane"
offsets=(2.8125 1.40625 4.21875 0.703125 2.109375 3.515625 4.921875)
cd "$repo"

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

count=$(verified_count)
echo "roll-offset top-up starting unique_verified=$count target=$target"
for round in "${!offsets[@]}"; do
  if [[ "$count" -ge "$target" ]]; then break; fi
  seed=$((20996826 + round * 1000))
  offset=${offsets[$round]}
  output="$root/seed_$seed"
  echo "round=$round seed=$seed roll_offset=$offset verified=$count"
  bash migration_4090/run_bulk_baseline_object_custom.sh \
    "$gpu" "$object" "$density" "$seed" 60 20 32 12 1.15 "$offset"
  strict=$(strict_count "$output/bimanual_dataset.pt")
  if [[ "$strict" -eq 0 ]]; then
    echo "seed=$seed has no strict samples"
    continue
  fi
  remaining=$((target - count))
  select_count=$((remaining * 3 / 2 + 12))
  if [[ "$select_count" -gt 160 ]]; then select_count=160; fi
  bash migration_4090/repeat_verify_bulk_seed.sh \
    "$gpu" baseline "$object" "$density" "$seed" "$select_count" 1 12
  count=$(verified_count)
done

if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: baseline toy only $count unique repeat-verified samples" >&2
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
