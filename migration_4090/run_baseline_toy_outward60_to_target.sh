#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
object=ycb+toy_airplane
slug=ycb_toy_airplane
gpu=3
density=65
target=100
first_seed=20263826
root="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/$slug"
final="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/final_100_each/$slug"
smoke="$repo/graph_exp/bimanual_data/baseline_toy_airplane_outward60_contact12_smoke_v2/bimanual_dataset.pt"
log="$repo/migration_4090/logs/bulk_target_baseline_ycb_toy_airplane_outward60_contact12_gpu3.log"
cd "$repo"
exec > >(tee "$log") 2>&1

echo "Waiting for outward-60/contact-12 smoke"
while screen -ls 2>/dev/null | grep -q '[.]baseline_toy_outward60_contact12_smoke_v2[[:space:]]'; do
  sleep 30
done
test -f "$smoke"
smoke_count=$("$conda_root/envs/tro/bin/python" -c \
  "import torch; print(len(torch.load('$smoke',map_location='cpu',weights_only=False)['samples']))")
echo "outward-60/contact-12 smoke strict samples=$smoke_count"
if [[ "$smoke_count" -lt 1 ]]; then
  echo "FAILED: outward-60/contact-12 smoke produced no strict sample" >&2
  exit 1
fi

verified_count() {
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
}

run_and_verify() {
  local seed=$1
  local current remaining select_count
  bash migration_4090/run_bulk_baseline_object_custom.sh \
    "$gpu" "$object" "$density" "$seed" 60 20 32 12
  current=$(verified_count)
  remaining=$((target - current))
  if [[ "$remaining" -le 0 ]]; then return; fi
  select_count=$((remaining * 3 / 2 + 12))
  if [[ "$select_count" -gt 160 ]]; then select_count=160; fi
  bash migration_4090/repeat_verify_bulk_seed.sh \
    "$gpu" baseline "$object" "$density" "$seed" "$select_count" 1 12
}

round=0
count=$(verified_count)
while [[ "$count" -lt "$target" && "$round" -le 12 ]]; do
  seed=$((first_seed + round * 1000))
  echo "outward-60 round=$round seed=$seed verified=$count target=$target"
  run_and_verify "$seed"
  count=$(verified_count)
  round=$((round + 1))
done
if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: only $count unique verified airplane samples" >&2
  exit 1
fi
test ! -e "$final"
"$conda_root/envs/tro/bin/python" migration_4090/assemble_bulk_object.py \
  --input-root "$root" --object-name "$object" --method baseline \
  --target "$target" --output-dir "$final"
echo "COMPLETE baseline toy_airplane count=$target final=$final"
