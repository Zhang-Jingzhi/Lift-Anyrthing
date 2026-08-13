#!/usr/bin/env bash
set -euo pipefail
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
PY=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
outdir=migration_4090/large_random_v1_candidates/repair_bleach_v100
mkdir -p "$outdir"
inputs=()
for i in $(seq 0 12); do
  out="$outdir/part_${i}.pt"
  audit="$outdir/part_${i}.json"
  "$PY" migration_4090/generate_bidex_local_neighborhood.py \
    --source-dataset graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_bleach_cleanser/size_000/assembled_target_13/bimanual_dataset.pt \
    --source-vis migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_001_transfer_v1.pt \
    --object ycb+bleach_cleanser_formal_large_random_v1_000 \
    --target-object ycb+bleach_cleanser_formal_large_random_v1_001 \
    --root-scale-ratio 1.176 \
    --source-mesh data/data_urdf/object/ycb/bleach_cleanser_formal_large_random_v1_000/coacd_allinone.obj \
    --preserve-root-surface-offset \
    --sample-index "$i" \
    --candidate-method baseline_bleach_size001_repair_v100 \
    --yaw-degrees -3 0 3 \
    --z-mm -2 0 2 \
    --left-radial-mm -2 0 2 \
    --right-radial-mm -2 0 2 \
    --output "$out" --audit-json "$audit"
  inputs+=("$out")
done
"$PY" migration_4090/combine_precomputed_candidate_sources.py \
  --inputs "${inputs[@]}" \
  --output migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_001_transfer_adaptive_v100.pt
"$PY" - <<'PY'
import torch
p = 'migration_4090/large_random_v1_candidates/baseline_ycb_bleach_cleanser_size_001_transfer_adaptive_v100.pt'
x = torch.load(p, map_location='cpu', weights_only=False)[0]['precomputed_bimanual_candidates']
assert x['left_q'].shape == (1053, 22)
assert torch.isfinite(x['left_q']).all() and torch.isfinite(x['right_q']).all()
print('validated', tuple(x['left_q'].shape), 'unique_rows', len({tuple(q.tolist()) for q in x['left_q']}))
PY
