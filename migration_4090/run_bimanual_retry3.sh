#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
log="$repo/migration_4090/logs/bimanual_generation_smoke_retry3.log"
output="$repo/graph_exp/bimanual_data/4090_smoke_cylinder_xlarge_seed20260804_retry3"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run -n tro python generate_bimanual_pilot.py \
  --source-vis data/bimanual/source_vis.pt \
  --objects contactdb+cylinder_xlarge \
  --pairs-per-object 2 \
  --roll-count 4 \
  --isaac-batch-size 4 \
  --realized-batch-size 2 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --independent-directions \
  --right-max-outward-mm 120 \
  --right-outward-step-mm 5 \
  --seed 20260804 \
  --gpu 0 \
  --output-dir "$output" \
  --force
