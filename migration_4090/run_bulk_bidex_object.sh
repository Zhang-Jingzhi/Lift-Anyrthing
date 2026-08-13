#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PHYSICAL_GPU OBJECT_NAME DENSITY SEED" >&2
  exit 2
fi
gpu=$1
object=$2
density=$3
seed=$4
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
slug=${object//+/_}
candidate="$repo/migration_4090/bulk_candidates/bidex_v1/$slug/seed_${seed}.pt"
audit="$repo/migration_4090/results/bulk_bidex_v1/$slug/seed_${seed}_candidate_audit.json"
output="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/$slug/seed_$seed"
log="$repo/migration_4090/logs/bulk_bidex_${slug}_seed${seed}_gpu${gpu}.log"

test ! -e "$candidate"
test ! -e "$audit"
test ! -e "$output"
mkdir -p "$(dirname "$candidate")" "$(dirname "$audit")"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python migration_4090/generate_bidexgrasp_candidates.py \
  --source-vis data/bimanual/source_vis.pt \
  --object "$object" \
  --surface-points 2048 \
  --anchors 120 \
  --region-points 160 \
  --gws-contacts 5 \
  --region-pairs 120 \
  --maximum-source-seeds 3 \
  --hand-variants 7 \
  --preserve-standoff-diversity \
  --minimum-standoff-mm 35 \
  --maximum-standoff-mm 95 \
  --standoff-step-mm 10 \
  --minimum-pair-clearance-mm 18 \
  --maximum-pair-clearance-mm 35 \
  --minimum-pair-separation-fraction 0.65 \
  --minimum-vertical-separation-mm 0 \
  --maximum-vertical-separation-mm 15 \
  --contact-mm 2 \
  --penetration-mm 2 \
  --minimum-contact-links 2 \
  --support-clearance-mm 1 \
  --friction 1 \
  --seed "$seed" \
  --output "$candidate" \
  --audit-json "$audit"
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python generate_bimanual_pilot.py \
  --source-vis "$candidate" \
  --objects "$object" \
  --pairs-per-object 120 \
  --roll-count 1 \
  --isaac-batch-size 16 \
  --realized-batch-size 4 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --lateral-max-root-z-mm 100 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
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
