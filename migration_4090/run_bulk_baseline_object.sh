#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU OBJECT_NAME DENSITY SEED [ROLL_OFFSET_DEGREES]" >&2
  exit 2
fi
gpu=$1
object=$2
density=$3
seed=$4
roll_offset_degrees=${5:-0}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
slug=${object//+/_}
output="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/$slug/seed_$seed"
log="$repo/migration_4090/logs/bulk_baseline_${slug}_seed${seed}_gpu${gpu}.log"

test ! -e "$output"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python generate_bimanual_pilot.py \
  --source-vis data/bimanual/source_vis.pt \
  --objects "$object" \
  --pairs-per-object 20 \
  --roll-count 32 \
  --roll-offset-degrees "$roll_offset_degrees" \
  --isaac-batch-size 16 \
  --realized-batch-size 4 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --tabletop-left-roll-degrees 0 \
  --tabletop-root-height-mm 60 \
  --lateral-max-root-z-mm 90 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
  --symmetric-source \
  --right-max-outward-mm 20 \
  --right-outward-step-mm 10 \
  --radial-fine-step-mm 2 \
  --radial-min-offset-mm -20 \
  --penetration-mm 2 \
  --contact-mm 2 \
  --min-contact-links 1 \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit 0.7 \
  --contact-offset 0.002 \
  --object-density "$density" \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --seed "$seed" \
  --gpu 0 \
  --output-dir "$output" \
  --force
