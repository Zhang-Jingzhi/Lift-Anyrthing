#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
export SOURCE_VIS=migration_4090/derived/source_vis_bleach_predict_source15_v16.pt
export OBJECT_DENSITY=100
export OUTPUT_DIR="$repo/graph_exp/bimanual_data/4090_irregular_bleach_baseline_predict_source15_density100_v16"
export LOG_PATH="$repo/migration_4090/logs/irregular_bleach_baseline_predict_source15_density100_v16.log"
exec "$repo/migration_4090/run_irregular_bleach_baseline_source0_v6.sh"
