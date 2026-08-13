#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

for left_roll in 0 90 180 270; do
  output="graph_exp/bimanual_data/baseline_power_drill_roll${left_roll}_smoke_v1"
  log="migration_4090/logs/baseline_power_drill_roll${left_roll}_smoke_v1.log"
  test ! -e "$output"
  test ! -e "$log"
  "$conda_root/bin/conda" run --no-capture-output -n tro \
    python generate_bimanual_pilot.py \
    --source-vis data/bimanual/source_vis.pt \
    --objects ycb+power_drill \
    --pairs-per-object 4 \
    --roll-count 8 \
    --isaac-batch-size 16 \
    --realized-batch-size 2 \
    --left-robot-name allegro_left \
    --right-robot-name allegro_right \
    --gravity 9.8 \
    --gravity-settle-step 500 \
    --support-during-closure \
    --no-fixture-during-closure \
    --opposition-mode tabletop \
    --tabletop-left-roll-degrees "$left_roll" \
    --tabletop-root-height-mm 60 \
    --max-root-height-fraction 3 \
    --lateral-max-root-z-mm 100 \
    --lateral-max-height-diff-mm 45 \
    --independent-directions \
    --symmetric-source \
    --right-max-outward-mm 60 \
    --right-outward-step-mm 10 \
    --radial-fine-step-mm 2 \
    --radial-min-offset-mm -20 \
    --penetration-mm 2 \
    --contact-mm 4 \
    --min-contact-links 1 \
    --robot-friction 1 \
    --object-friction 1 \
    --finger-effort-limit 0.7 \
    --contact-offset 0.002 \
    --object-density 170 \
    --max-gravity-displacement 0.01 \
    --max-direction-displacement 0.015 \
    --seed "$((20268824 + left_roll))" \
    --gpu 0 \
    --output-dir "$output" \
    --force >"$log" 2>&1
done
