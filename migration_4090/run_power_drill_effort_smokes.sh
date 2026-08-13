#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo"

for effort in 0.15 0.25 0.35 0.45 0.55; do
  tag=${effort/./p}
  output="graph_exp/bimanual_data/baseline_power_drill_effort_${tag}_smoke_v1"
  log="migration_4090/logs/baseline_power_drill_effort_${tag}_smoke_v1.log"
  test ! -e "$output"
  test ! -e "$log"
  "$conda_root/bin/conda" run --no-capture-output -n tro \
    python generate_bimanual_pilot.py \
    --source-vis migration_4090/bulk_candidates/baseline_power_drill_candidate27_v1.pt \
    --objects ycb+power_drill \
    --pairs-per-object 1 \
    --roll-count 1 \
    --isaac-batch-size 1 \
    --realized-batch-size 1 \
    --left-robot-name allegro_left \
    --right-robot-name allegro_right \
    --gravity 9.8 \
    --gravity-settle-step 500 \
    --support-during-closure \
    --no-fixture-during-closure \
    --opposition-mode tabletop \
    --max-root-height-fraction 3 \
    --lateral-max-root-z-mm 100 \
    --lateral-max-height-diff-mm 45 \
    --independent-directions \
    --penetration-mm 2 \
    --contact-mm 4 \
    --min-contact-links 1 \
    --robot-friction 1 \
    --object-friction 1 \
    --finger-effort-limit "$effort" \
    --contact-offset 0.002 \
    --object-density 170 \
    --max-gravity-displacement 0.01 \
    --max-direction-displacement 0.015 \
    --seed 20270824 \
    --gpu 0 \
    --output-dir "$output" \
    --force >"$log" 2>&1
done
