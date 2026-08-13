#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
source="$repo/graph_exp/bimanual_data/4090_irregular_bleach_baseline_source4_candidates06_strict_minlinks1_v20/sample1_dataset_v21.pt"
output="$repo/graph_exp/bimanual_data/4090_irregular_bleach_baseline_source4_candidates06_strict_minlinks1_v20/repeat_verified_v21"
log="$repo/migration_4090/logs/baseline_bleach_repeat_v21.log"

test ! -e "$output"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python scripts/repeat_verify_xlarge_batch.py \
  --source-dataset "$source" \
  --output-dir "$output" \
  --isaac-python "$isaac_python" \
  --object-name ycb+bleach_cleanser \
  --object-density 100 \
  --finger-effort-limit 0.7 \
  --min-contact-links 1 \
  --additional-repeats 2 \
  --batch-size 2 \
  --geometry-batch-size 1 \
  --gpu 0
