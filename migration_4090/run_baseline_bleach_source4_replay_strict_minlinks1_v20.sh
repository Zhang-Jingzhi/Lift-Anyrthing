#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
export PHYSICAL_GPU=0
export OBJECT_DENSITY=100
export MINIMUM_CONTACT_LINKS=1
export SOURCE_VIS=migration_4090/derived/source_vis_bleach_baseline_source4_candidates06_v20.pt
export OUTPUT_DIR="$repo/graph_exp/bimanual_data/4090_irregular_bleach_baseline_source4_candidates06_strict_minlinks1_v20"
export LOG_PATH="$repo/migration_4090/logs/irregular_bleach_baseline_source4_candidates06_strict_minlinks1_v20.log"
exec "$repo/migration_4090/run_bidexgrasp_bleach_strict_validation.sh"
