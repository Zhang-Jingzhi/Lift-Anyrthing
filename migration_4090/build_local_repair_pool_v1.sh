#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 8 ]]; then
  echo "usage: $0 NAME SOURCE_DATASET SOURCE_VIS SOURCE_OBJECT TARGET_OBJECT ROOT_SCALE SOURCE_MESH OUTPUT" >&2
  exit 2
fi
name=$1
source_dataset=$2
source_vis=$3
source_object=$4
target_object=$5
root_scale=$6
source_mesh=$7
output=$8
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
py=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
cd "$repo"
work="migration_4090/large_random_v1_candidates/repair_${name}_v1"
mkdir -p "$work"
inputs=()
for i in $(seq 0 12); do
  part="$work/part_${i}.pt"
  audit="$work/part_${i}.json"
  "$py" migration_4090/generate_bidex_local_neighborhood.py \
    --source-dataset "$source_dataset" --source-vis "$source_vis" \
    --object "$source_object" --target-object "$target_object" \
    --root-scale-ratio "$root_scale" --source-mesh "$source_mesh" \
    --preserve-root-surface-offset --sample-index "$i" \
    --candidate-method "${name}_local_repair_v1" \
    --yaw-degrees -3 0 3 --z-mm -2 0 2 \
    --left-radial-mm -2 0 2 --right-radial-mm -2 0 2 \
    --output "$part" --audit-json "$audit"
  inputs+=("$part")
done
tmp="$work/combined.pt"
"$py" migration_4090/combine_precomputed_candidate_sources.py --inputs "${inputs[@]}" --output "$tmp"
"$py" - "$tmp" "$output" <<'PY'
import sys, torch
src, out = sys.argv[1:]
x = torch.load(src, map_location='cpu', weights_only=False)[0]
c = x['precomputed_bimanual_candidates']
seen = set(); left=[]; right=[]; meta=[]
for i,(l,r) in enumerate(zip(c['left_q'], c['right_q'])):
    key=(tuple(torch.round(l*100000).to(torch.int64).tolist()), tuple(torch.round(r*100000).to(torch.int64).tolist()))
    if key in seen: continue
    seen.add(key); left.append(l); right.append(r); meta.append(c.get('metadata',[])[i])
x['precomputed_bimanual_candidates']={'left_q':torch.stack(left),'right_q':torch.stack(right),'metadata':meta}
torch.save([x], out)
assert torch.isfinite(x['precomputed_bimanual_candidates']['left_q']).all()
print('dedup_saved',out,'count',len(left),'unique',len(seen))
PY
