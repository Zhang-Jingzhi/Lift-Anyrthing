#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "usage: $0 GPU OBJECT DENSITY SEED MAX_OUTWARD_MM PAIRS ROLLS CONTACT_MM [MAX_ROOT_HEIGHT_FRACTION] [ROLL_OFFSET_DEGREES]" >&2
  exit 2
fi
gpu=$1
object=$2
density=$3
seed=$4
max_outward=$5
pairs=$6
rolls=$7
contact_mm=$8
max_root_height_fraction=${9:-1.15}
roll_offset_degrees=${10:-0}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
slug=${object//+/_}
output="$repo/graph_exp/bimanual_data/bulk_irregular_baseline_v1/$slug/seed_$seed"
log="$repo/migration_4090/logs/bulk_baseline_${slug}_seed${seed}_outward${max_outward}_contact${contact_mm}_gpu${gpu}.log"

test ! -e "$output"
test ! -e "$log"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python generate_bimanual_pilot.py \
  --source-vis data/bimanual/source_vis.pt \
  --objects "$object" \
  --pairs-per-object "$pairs" \
  --roll-count "$rolls" \
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
  --max-root-height-fraction "$max_root_height_fraction" \
  --lateral-max-root-z-mm 90 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
  --symmetric-source \
  --right-max-outward-mm "$max_outward" \
  --right-outward-step-mm 10 \
  --radial-fine-step-mm 2 \
  --radial-min-offset-mm -20 \
  --penetration-mm 2 \
  --contact-mm "$contact_mm" \
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
