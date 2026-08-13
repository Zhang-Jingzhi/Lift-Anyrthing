#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 baseline|bidex_v3 SIZE_SEED [RUN_TAG]" >&2
  exit 2
fi
method=$1
size_seed=$2
run_tag=${3:-random_preview}
family_count=4
size_index=$((size_seed % family_count))
if (( size_index < 0 )); then
  size_index=$((size_index + family_count))
fi
printf -v size_suffix '%03d' "$size_index"

export OBJECT_OVERRIDE="ycb+pitcher_base_large_random_iso_v2_${size_suffix}"
if [[ "$method" == baseline ]]; then
  # Baseline must not consume the BiDex precomputed-candidate pool embedded in
  # the visualization source used for the BiDex replay preview.
  export SOURCE_VIS_OVERRIDE="migration_4090/derived/source_vis_pitcher_base_large_random_iso_v2_baseline.pt"
else
  export SOURCE_VIS_OVERRIDE="migration_4090/derived/source_vis_pitcher_base_large_random_iso_v2.pt"
fi
export MINIMUM_HORIZONTAL_SPAN_MM=450
export MINIMUM_HEIGHT_MM=585
export MAX_GRAVITY_DISPLACEMENT=0.0135

exec "$(dirname "$0")/run_pitcher_2x_method_sample.sh" \
  "$method" "${run_tag}_size${size_suffix}_seed${size_seed}"
