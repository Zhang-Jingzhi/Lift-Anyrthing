#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
output="$repo/graph_exp/bimanual_data/4090_irregular_piggy_bank_sample_v3_sources0_4_cached"
log="$repo/migration_4090/logs/irregular_piggy_sample_v3_sources0_4_cached.log"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run -n tro python generate_bimanual_pilot.py \
  --source-vis migration_4090/derived/source_vis_piggy_sources0_4.pt \
  --objects contactdb+piggy_bank \
  --pairs-per-object 5 \
  --roll-count 16 \
  --isaac-batch-size 24 \
  --realized-batch-size 2 \
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
  --right-max-outward-mm 120 \
  --right-outward-step-mm 10 \
  --penetration-mm 2 \
  --contact-mm 2 \
  --min-contact-links 3 \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit 0.7 \
  --contact-offset 0.002 \
  --object-density 50 \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --seed 20260805 \
  --gpu 0 \
  --output-dir "$output" \
  --force
