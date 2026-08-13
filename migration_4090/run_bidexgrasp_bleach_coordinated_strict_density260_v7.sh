#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
export PHYSICAL_GPU=1
export OBJECT_DENSITY=260
export SOURCE_VIS=migration_4090/derived/source_vis_bleach_bidexgrasp_candidates_coordinated_v5.pt
export OUTPUT_DIR="$repo/graph_exp/bimanual_data/4090_irregular_bleach_bidexgrasp_coordinated_strict_density260_v7"
export LOG_PATH="$repo/migration_4090/logs/bidexgrasp_bleach_coordinated_strict_density260_v7.log"
exec "$repo/migration_4090/run_bidexgrasp_bleach_strict_validation.sh"
