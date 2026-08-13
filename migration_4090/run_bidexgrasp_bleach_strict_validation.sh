#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
output=${OUTPUT_DIR:-$repo/graph_exp/bimanual_data/4090_irregular_bleach_bidexgrasp_strict_v1}
log=${LOG_PATH:-$repo/migration_4090/logs/bidexgrasp_bleach_strict_validation_v1.log}
object_density=${OBJECT_DENSITY:-50}
finger_effort_limit=${FINGER_EFFORT_LIMIT:-0.7}
minimum_contact_links=${MINIMUM_CONTACT_LINKS:-3}
source_vis=${SOURCE_VIS:-migration_4090/derived/source_vis_bleach_bidexgrasp_candidates_v1.pt}

export PATH="$conda_root/envs/isaac/bin:$PATH"
# Use physical GPU 1 while the baseline continues on physical GPU 0.  Isaac
# sees this isolated device as logical GPU 0.
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU:-1}
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python generate_bimanual_pilot.py \
  --source-vis "$source_vis" \
  --objects ycb+bleach_cleanser \
  --pairs-per-object 2 \
  --roll-count 1 \
  --isaac-batch-size 2 \
  --realized-batch-size 2 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --lateral-max-root-z-mm 90 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
  --penetration-mm 2 \
  --contact-mm 2 \
  --min-contact-links "$minimum_contact_links" \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit "$finger_effort_limit" \
  --contact-offset 0.002 \
  --object-density "$object_density" \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --seed 20260805 \
  --gpu 0 \
  --output-dir "$output" \
  --force
