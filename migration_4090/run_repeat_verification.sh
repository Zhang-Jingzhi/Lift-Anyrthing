#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
log="$repo/migration_4090/logs/repeat_verification_gpu0.log"
source_dir="$repo/graph_exp/bimanual_data/4090_smoke_cylinder_xlarge_source14_targeted_retry5"
output="$source_dir/repeat_verified_gpu0"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run -n tro python scripts/repeat_verify_xlarge_batch.py \
  --source-dataset "$source_dir/repeat_source_sample.pt" \
  --quality-json "$source_dir/repeat_source_quality.json" \
  --output-dir "$output" \
  --isaac-python "$isaac_python" \
  --additional-repeats 2 \
  --batch-size 2 \
  --geometry-batch-size 1 \
  --gpu 0
