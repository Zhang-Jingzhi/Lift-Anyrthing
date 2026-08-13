#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 baseline|bidex_v3 [RUN_TAG]" >&2
  exit 2
fi
method=$1
run_tag=${2:-sample_v1}
if [[ "$method" != baseline && "$method" != bidex_v3 ]]; then
  echo "unknown method: $method" >&2
  exit 2
fi

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
tro_python="$conda_root/envs/tro/bin/python"
isaac_python="$conda_root/envs/isaac/bin/python"
object=${OBJECT_OVERRIDE:-ycb+pitcher_base_2x_ref_v3}
source_vis=${SOURCE_VIS_OVERRIDE:-migration_4090/derived/source_vis_pitcher_base_2x_ref_v3.pt}
minimum_horizontal_span_mm=${MINIMUM_HORIZONTAL_SPAN_MM:-450}
minimum_height_mm=${MINIMUM_HEIGHT_MM:-590}
max_gravity_displacement=${MAX_GRAVITY_DISPLACEMENT:-0.0125}
root="$repo/graph_exp/bimanual_data/pitcher_2x_${method}_$run_tag"
selection="$repo/migration_4090/results/pitcher_2x_${method}_selection_$run_tag.pt"
verified="$root/repeat_verified"
render_root="$repo/migration_4090/renders/pitcher_2x_${method}_$run_tag"
candidate="$repo/migration_4090/bidex_v3_candidates/pitcher_2x_$run_tag.pt"
audit="$repo/migration_4090/results/pitcher_2x_bidex_v3_candidate_$run_tag.json"

paths=("$root" "$selection" "$render_root")
if [[ "$method" == bidex_v3 ]]; then
  paths+=("$candidate" "$audit")
fi
for path in "${paths[@]}"; do
  test ! -e "$path"
done
mkdir -p "$(dirname "$candidate")" "$(dirname "$audit")" "$render_root"

export PATH="$conda_root/envs/isaac/bin:$PATH"
physical_gpu=${PHYSICAL_GPU:-7}
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

if [[ "$method" == bidex_v3 ]]; then
  "$tro_python" migration_4090/generate_bidexgrasp_candidates.py \
    --source-vis "$source_vis" \
    --object "$object" \
    --minimum-object-horizontal-span-mm "$minimum_horizontal_span_mm" \
    --minimum-object-height-mm "$minimum_height_mm" \
    --surface-points 3072 \
    --anchors 160 \
    --region-points 192 \
    --gws-contacts 8 \
    --region-pairs 6 \
    --region-scale 0.22 \
    --enhanced-bidex-v3 \
    --v2-refinement-levels 2 \
    --v2-joint-steps-degrees 4 \
    --v2-aperture-scales -0.75 -0.5 -0.25 0.5 1.0 1.5 2.0 2.5 \
    --v3-parent-candidate-cap 12 \
    --v3-finger-perturbations 16 \
    --v3-finger-perturb-degrees 8 \
    --v3-side-variants 6 \
    --v3-quality-keep-fraction 0.6 \
    --v3-final-candidates 36 \
    --maximum-normal-opposition-cosine -0.15 \
    --minimum-radial-normal-alignment 0.15 \
    --minimum-height-fraction 0.22 \
    --maximum-height-fraction 0.68 \
    --minimum-pair-separation-fraction 0.70 \
    --minimum-vertical-separation-mm 0 \
    --maximum-vertical-separation-mm 40 \
    --minimum-standoff-mm 35 \
    --maximum-standoff-mm 155 \
    --standoff-step-mm 20 \
    --left-rolls 0 45 90 135 \
    --right-rolls 0 45 90 135 \
    --maximum-source-seeds 2 \
    --hand-variants 4 \
    --preserve-standoff-diversity \
    --minimum-pair-clearance-mm 50 \
    --maximum-pair-clearance-mm 550 \
    --target-pair-clearance-mm 300 \
    --contact-mm 2 \
    --penetration-mm 2 \
    --minimum-contact-links 2 \
    --minimum-contact-points 4 \
    --minimum-contact-digits 1 \
    --target-contact-links 3 \
    --target-contact-points 14 \
    --target-contact-digits 2 \
    --target-contact-spread-mm 20 \
    --support-clearance-mm 1 \
    --friction 1 \
    --seed 20260853 \
    --output "$candidate" \
    --audit-json "$audit"
  source_vis="$candidate"
  pairs_per_object=36
  roll_count=1
  density=2
  effort=1.2
  friction=1.2
  contact_offset=0.003
else
  pairs_per_object=4
  roll_count=8
  density=0.5
  effort=2
  friction=2
  contact_offset=0.005
fi

generator_args=(
  --source-vis "$source_vis"
  --objects "$object"
  --pairs-per-object "$pairs_per_object"
  --roll-count "$roll_count"
  --isaac-batch-size 16
  --realized-batch-size 4
  --left-robot-name allegro_left
  --right-robot-name allegro_right
  --gravity 9.8
  --gravity-settle-step 500
  --min-lift-height 0.02
  --support-during-closure
  --no-fixture-during-closure
  --opposition-mode tabletop
  --tabletop-left-roll-degrees 0
  --tabletop-root-height-mm 60
  --max-root-height-fraction 0.80
  --min-object-horizontal-span-mm "$minimum_horizontal_span_mm"
  --lateral-max-root-z-mm 260
  --lateral-max-height-diff-mm 55
  --independent-directions
  --penetration-mm 2
  --contact-mm 2
  --min-contact-links 1
  --robot-friction "$friction"
  --object-friction "$friction"
  --finger-effort-limit "$effort"
  --contact-offset "$contact_offset"
  --object-density "$density"
  --max-gravity-displacement "$max_gravity_displacement"
  --max-direction-displacement 0.015
  --seed 20260853
  --gpu 0
  --output-dir "$root"
  --force
)
if [[ "$method" == baseline ]]; then
  generator_args+=(
    --symmetric-source
    --right-max-outward-mm 260
    --right-outward-step-mm 40
    --radial-fine-step-mm 4
    --radial-min-offset-mm -20
  )
fi
"$tro_python" generate_bimanual_pilot.py "${generator_args[@]}"

"$tro_python" migration_4090/select_bulk_repeat_candidates.py \
  --source-dataset "$root/bimanual_dataset.pt" \
  --output "$selection" \
  --max-samples 1

"$tro_python" scripts/repeat_verify_xlarge_batch.py \
  --repo "$repo" \
  --source-dataset "$selection" \
  --output-dir "$verified" \
  --isaac-python "$isaac_python" \
  --additional-repeats 2 \
  --batch-size 8 \
  --geometry-batch-size 2 \
  --gpu 0 \
  --object-name "$object" \
  --object-density "$density" \
  --finger-effort-limit "$effort" \
  --lift-height 0.05 \
  --min-lift-height 0.02 \
  --robot-friction "$friction" \
  --object-friction "$friction" \
  --contact-offset "$contact_offset" \
  --max-gravity-displacement "$max_gravity_displacement" \
  --max-direction-displacement 0.015 \
  --min-contact-links 1 \
  --geometry-contact-mm 2

"$tro_python" scripts/render_verified_bimanual.py \
  --dataset "$verified/verified_dataset.pt" \
  --sample-index 0 \
  --pose-stage final \
  --show-table \
  --contact-csv "$render_root/contacts.csv" \
  --output "$render_root/final_three_views.png"

echo "PITCHER_2X_${method^^}_SAMPLE_COMPLETE"
