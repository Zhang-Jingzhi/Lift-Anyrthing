#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU baseline|bidex_v3 BASE_OBJECT_NAME" >&2
  exit 2
fi
gpu=$1
method=$2
base_object=$3
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
tro_python=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
catalog="$repo/migration_4090/results/large_random_6x8_v1/catalog.json"
schedule="$repo/migration_4090/results/large_random_6x8_v1/shared_schedule_6x100.json"
base_slug=${base_object//+/_}
method_root="$repo/graph_exp/bimanual_data/large_random_6x100_v1/$method/$base_slug"
final="$method_root/final_100"
# Keep the fixed 600-row cross-method schedule, while ensuring a resumed formal
# campaign starts from generation seeds that were not used by earlier queues.
# The 202608050 family was used by the first high-v1 queue.  Resume after the
# headless GPU-isolation fix from a disjoint seed family so interrupted seed
# directories remain untouched and cannot be mistaken for the new run.
formal_seed_offset=${FORMAL_GENERATION_SEED_OFFSET:-302608050}
baseline_resume_source=${BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE:-}
bidex_resume_source=${BIDEX_ONE_SHOT_FINAL_SOURCE:-}
cd "$repo"

test -f "$catalog"
test -f "$schedule"
if [[ -n "$baseline_resume_source" ]]; then
  test -s "$baseline_resume_source"
fi
if [[ -n "$bidex_resume_source" ]]; then
  test -s "$bidex_resume_source"
fi
if [[ -s "$final/bimanual_dataset.pt" ]]; then
  echo "already complete: $final"
  exit 0
fi

while IFS=$'\t' read -r size_index object density target base_seed; do
  size_root="$method_root/size_$(printf '%03d' "$size_index")"
  generation_base_seed=$((base_seed + formal_seed_offset))
  transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_v1.pt"
  enhanced_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_independent_yaw_z_v2.pt"
  repeat_margin_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_repeat_margin_v2.pt"
  verified_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_verified_v3.pt"
  micro_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_micro_v4.pt"
  adaptive_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v5.pt"
  adaptive_transfer_source_v6="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v6.pt"
  adaptive_transfer_source_v7="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v7.pt"
  adaptive_transfer_source_v8="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v8.pt"
  adaptive_transfer_source_v9="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v9.pt"
  adaptive_transfer_source_v10="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v10.pt"
  adaptive_transfer_source_v11="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v11.pt"
  adaptive_transfer_source_v12="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v12.pt"
  adaptive_transfer_source_v13="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v13.pt"
  if [[ -s "$enhanced_transfer_source" ]]; then
    transfer_source="$enhanced_transfer_source"
  fi
  if [[ -s "$repeat_margin_transfer_source" ]]; then
    transfer_source="$repeat_margin_transfer_source"
  fi
  if [[ -s "$verified_transfer_source" ]]; then
    transfer_source="$verified_transfer_source"
  fi
  if [[ -s "$micro_transfer_source" ]]; then
    transfer_source="$micro_transfer_source"
  fi
  if [[ -s "$adaptive_transfer_source" ]]; then
    transfer_source="$adaptive_transfer_source"
  fi
  if [[ -s "$adaptive_transfer_source_v6" ]]; then
    transfer_source="$adaptive_transfer_source_v6"
  fi
  if [[ -s "$adaptive_transfer_source_v7" ]]; then
    transfer_source="$adaptive_transfer_source_v7"
  fi
  if [[ -s "$adaptive_transfer_source_v8" ]]; then
    transfer_source="$adaptive_transfer_source_v8"
  fi
  if [[ -s "$adaptive_transfer_source_v9" ]]; then
    transfer_source="$adaptive_transfer_source_v9"
  fi
  if [[ -s "$adaptive_transfer_source_v10" ]]; then
    transfer_source="$adaptive_transfer_source_v10"
  fi
  if [[ -s "$adaptive_transfer_source_v11" ]]; then
    transfer_source="$adaptive_transfer_source_v11"
  fi
  if [[ -s "$adaptive_transfer_source_v12" ]]; then
    transfer_source="$adaptive_transfer_source_v12"
  fi
  if [[ -s "$adaptive_transfer_source_v13" ]]; then
    transfer_source="$adaptive_transfer_source_v13"
  fi
  # Adaptive repair pools are added while the formal queue is live.  Select
  # the highest available version automatically so a newly diagnosed local
  # neighborhood can be picked up by the supervisor without another launcher
  # edit or any overwrite of earlier pools.
  # Keep accepting newly created repair pools without requiring another
  # launcher edit.  Versions are append-only; the highest existing pool wins.
  for adaptive_version in {14..999}; do
    newer_adaptive_transfer_source="$repo/migration_4090/large_random_v1_candidates/${method}_${base_slug}_size_$(printf '%03d' "$size_index")_transfer_adaptive_v${adaptive_version}.pt"
    if [[ -s "$newer_adaptive_transfer_source" ]]; then
      transfer_source="$newer_adaptive_transfer_source"
    fi
  done
  if [[ -n "$baseline_resume_source" && "$method" == baseline ]]; then
    BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE="$baseline_resume_source" \
      migration_4090/run_large_random_asset_to_target_v1.sh \
      "$gpu" "$method" "$object" "$density" "$target" \
      "$generation_base_seed" "$size_root"
    baseline_resume_source=
  elif [[ -n "$bidex_resume_source" && "$method" == bidex_v3 ]]; then
    BIDEX_ONE_SHOT_FINAL_SOURCE="$bidex_resume_source" \
      migration_4090/run_large_random_asset_to_target_v1.sh \
      "$gpu" "$method" "$object" "$density" "$target" \
      "$generation_base_seed" "$size_root"
    bidex_resume_source=
  elif [[ -s "$transfer_source" && "$method" == baseline ]]; then
    BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE="$transfer_source" \
      migration_4090/run_large_random_asset_to_target_v1.sh \
      "$gpu" "$method" "$object" "$density" "$target" \
      "$generation_base_seed" "$size_root"
  elif [[ -s "$transfer_source" && "$method" == bidex_v3 ]]; then
    BIDEX_ONE_SHOT_FINAL_SOURCE="$transfer_source" \
      migration_4090/run_large_random_asset_to_target_v1.sh \
      "$gpu" "$method" "$object" "$density" "$target" \
      "$generation_base_seed" "$size_root"
  else
    migration_4090/run_large_random_asset_to_target_v1.sh \
      "$gpu" "$method" "$object" "$density" "$target" \
      "$generation_base_seed" "$size_root"
  fi
done < <(
  "$tro_python" - "$catalog" "$schedule" "$base_object" <<'PY'
import json, sys
from collections import Counter
catalog = json.load(open(sys.argv[1]))
schedule = json.load(open(sys.argv[2]))
base = sys.argv[3]
objects = {
    int(row['object_name'].rsplit('_', 1)[1]): row
    for row in catalog['objects']
    if row['base_object_name'] == base
}
rows = [row for row in schedule['entries'] if row['base_object_name'] == base]
counts = Counter(int(row['size_index']) for row in rows)
seeds = {}
for row in rows:
    seeds.setdefault(int(row['size_index']), []).append(int(row['sample_seed']))
if len(rows) != 100 or sorted(objects) != list(range(8)):
    raise RuntimeError(f'invalid catalog/schedule for {base}')
for size_index in range(8):
    row = objects[size_index]
    print(size_index, row['object_name'], row['density_kg_m3'], counts[size_index], min(seeds[size_index]), sep='\t')
PY
)

"$tro_python" migration_4090/assemble_large_random_base_v1.py \
  --input-root "$method_root" \
  --schedule "$schedule" \
  --base-object-name "$base_object" \
  --method "$method" \
  --output-dir "$final"
