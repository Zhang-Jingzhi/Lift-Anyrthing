#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
root="$repo/graph_exp/bimanual_data/4090_irregular_bleach_bidexgrasp_standoff_strict_density260_minlinks2_v9/contact_capture_v12"
log="$repo/migration_4090/logs/bidexgrasp_bleach_contact_capture_v12.log"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo"
exec > >(tee "$log") 2>&1
"$isaac_python" validation/bimanual_isaac_main.py \
  --object-name ycb+bleach_cleanser \
  --left-q-file "$root/left_q.pt" \
  --right-q-file "$root/right_q.pt" \
  --output-file "$root/isaac_result.pt" \
  --gpu 0 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --active-hands both \
  --lift-height 0.05 \
  --lift-step 100 \
  --min-lift-height 0.03 \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit 0.7 \
  --contact-offset 0.002 \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --object-density 260 \
  --support-during-closure \
  --independent-directions \
  --capture-contacts
