#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 && $# -ne 10 ]]; then
  echo "usage: $0 METHOD TARGET_SLUG SOURCE_DATASET SOURCE_OBJECT TARGET_OBJECT ROOT_SCALE SOURCE_MESH OUTPUT AUDIT [SAMPLE_INDEX]" >&2
  exit 2
fi
method=$1
target_slug=$2
source_dataset=$3
source_object=$4
target_object=$5
root_scale=$6
source_mesh=$7
output=$8
audit=$9
sample_index=${10:-0}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
py=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
cd "$repo"
source_promoted="${output%.pt}_final_pose_source.pt"
for path in "$source_promoted" "$output" "$audit"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite existing path: $path" >&2
    exit 1
  fi
done

"$py" - "$source_dataset" "$source_promoted" "$sample_index" <<'PY'
import sys, torch
src, out, sample_index = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = torch.load(src, map_location="cpu", weights_only=False)
samples = d.get("samples", [])
if not (0 <= sample_index < len(samples)):
    raise RuntimeError(f"sample index {sample_index} out of range for {len(samples)} samples")
s = dict(samples[sample_index])
m = s.get("metrics", {})
if m.get("strict_success") is not True:
    raise RuntimeError("source sample is not strict_success")
for key in ("left_q", "right_q"):
    if key not in s:
        raise RuntimeError(f"missing final pose key {key}")
s["left_q_seed"] = s["left_q"].detach().cpu().clone()
s["right_q_seed"] = s["right_q"].detach().cpu().clone()
s["source_final_pose_promoted_to_seed"] = True
torch.save({
    "version": "tro_grasp_verified_final_pose_source_v1",
    "manifest": dict(d.get("manifest", {}), promoted_from=str(src)),
    "samples": [s],
}, out)
print("promoted_final_pose", out, s["object_name"])
PY

"$py" migration_4090/generate_bidex_local_neighborhood.py \
  --source-dataset "$source_promoted" \
  --source-vis migration_4090/derived/large_random_6x8_v1/source_vis_large_random_6x8_v1.pt \
  --object "$source_object" --target-object "$target_object" \
  --root-scale-ratio "$root_scale" --source-mesh "$source_mesh" \
  --preserve-root-surface-offset --sample-index 0 \
  --candidate-method "${method}_${target_slug}_verified_final_pose_transfer_v1" \
  --yaw-degrees -8 -6 -4 -2 0 2 4 6 8 \
  --z-mm -8 -4 0 4 8 \
  --left-radial-mm -8 -4 0 4 8 \
  --right-radial-mm -8 -4 0 4 8 \
  --output "$output" --audit-json "$audit"

test -s "$output"; test -s "$audit"; test -s "$source_promoted"
echo "final-pose transfer pool ready: $output"
