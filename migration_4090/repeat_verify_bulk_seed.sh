#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 GPU METHOD OBJECT DENSITY SEED MAX_SELECT MIN_CONTACT_LINKS [GEOMETRY_CONTACT_MM]" >&2
  exit 2
fi
gpu=$1
method=$2
object=$3
density=$4
seed=$5
max_select=$6
min_links=$7
geometry_contact_mm=${8:-2}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
slug=${object//+/_}
seed_root="$repo/graph_exp/bimanual_data/bulk_irregular_${method}_v1/$slug/seed_$seed"
source="$seed_root/bimanual_dataset.pt"
selected="$seed_root/repeat_selection.pt"
output="$seed_root/repeat_verified"
log="$repo/migration_4090/logs/bulk_repeat_${method}_${slug}_seed${seed}_gpu${gpu}.log"

test -f "$source"
test ! -e "$selected"
test ! -e "$output"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/envs/tro/bin/python" migration_4090/select_bulk_repeat_candidates.py \
  --source-dataset "$source" \
  --output "$selected" \
  --max-samples "$max_select"
count=$("$conda_root/envs/tro/bin/python" -c \
  "import torch; print(len(torch.load('$selected', map_location='cpu', weights_only=False)['samples']))")
if [[ "$count" -eq 0 ]]; then
  echo "No strict samples to repeat-verify for $method $object seed $seed"
  exit 0
fi
"$conda_root/envs/tro/bin/python" scripts/repeat_verify_xlarge_batch.py \
  --repo "$repo" \
  --source-dataset "$selected" \
  --output-dir "$output" \
  --isaac-python "$ISAAC_PYTHON" \
  --additional-repeats 2 \
  --batch-size 16 \
  --geometry-batch-size 4 \
  --gpu 0 \
  --object-name "$object" \
  --object-density "$density" \
  --finger-effort-limit 0.7 \
  --min-contact-links "$min_links" \
  --geometry-contact-mm "$geometry_contact_mm"
