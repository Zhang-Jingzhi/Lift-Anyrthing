#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
export SOURCE_VIS=migration_4090/derived/source_vis_bleach_source1.pt
export OUTPUT_DIR="$repo/graph_exp/bimanual_data/4090_irregular_bleach_baseline_bidirectional_source1_v9"
export LOG_PATH="$repo/migration_4090/logs/irregular_bleach_baseline_bidirectional_source1_v9.log"
exec "$repo/migration_4090/run_irregular_bleach_baseline_source0_v6.sh"
