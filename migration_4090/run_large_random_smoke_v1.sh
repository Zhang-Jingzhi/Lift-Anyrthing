#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 baseline|bidex_v3 PHYSICAL_GPU RUN_TAG" >&2
  exit 2
fi
method=$1
physical_gpu=$2
run_tag=$3
if [[ "$method" != baseline && "$method" != bidex_v3 ]]; then
  echo "unknown method: $method" >&2
  exit 2
fi

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
tro_python="$conda_root/envs/tro/bin/python"
isaac_python="$conda_root/envs/isaac/bin/python"
source_vis="$repo/migration_4090/derived/large_random_6x8_v1/source_vis_large_random_6x8_v1.pt"
object=${SMOKE_OBJECT:-ycb+toy_airplane_formal_large_random_v1_000}
density=${SMOKE_DENSITY:-18.408891879093186}
friction=1.2
effort=1.2
contact_offset=0.001
bidex_region_pairs=${BIDEX_REGION_PAIRS:-10}
bidex_max_normal_opposition_cosine=${BIDEX_MAX_NORMAL_OPPOSITION_COSINE:--0.15}
bidex_min_radial_normal_alignment=${BIDEX_MIN_RADIAL_NORMAL_ALIGNMENT:-0.15}
bidex_stop_after_candidates=${BIDEX_STOP_AFTER_CANDIDATES:-0}
bidex_precomputed_source=${BIDEX_PRECOMPUTED_SOURCE:-}
smoke_seed=${SMOKE_SEED:-20269000}
bidex_candidate_seed=${BIDEX_CANDIDATE_SEED:-$smoke_seed}
root="$repo/graph_exp/bimanual_data/large_random_v1_smoke/$method/$run_tag"
selection="$repo/migration_4090/results/large_random_v1_smoke/${method}_${run_tag}_selection.pt"
verified="$root/repeat_verified"
render_root="$repo/migration_4090/renders/large_random_v1_smoke/${method}_$run_tag"
candidate="$repo/migration_4090/large_random_v1_candidates/${method}_${run_tag}.pt"
audit="$repo/migration_4090/results/large_random_v1_smoke/${method}_${run_tag}_candidate_audit.json"

paths=("$root" "$selection" "$render_root")
if [[ "$method" == bidex_v3 && -z "$bidex_precomputed_source" ]]; then
  paths+=("$candidate" "$audit")
fi
for path in "${paths[@]}"; do
  test ! -e "$path"
done
mkdir -p "$(dirname "$candidate")" "$(dirname "$audit")" "$render_root"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

if [[ "$method" == bidex_v3 ]]; then
  if [[ -n "$bidex_precomputed_source" ]]; then
    test -s "$bidex_precomputed_source"
    source_vis="$bidex_precomputed_source"
    echo "using precomputed BiDex smoke candidates: $source_vis"
  else
    "$tro_python" migration_4090/generate_bidexgrasp_candidates.py \
      --source-vis "$source_vis" \
      --object "$object" \
      --minimum-object-horizontal-span-mm 360 \
      --surface-points 2048 \
      --anchors 128 \
      --region-points 192 \
      --gws-contacts 8 \
      --region-pairs "$bidex_region_pairs" \
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
      --maximum-normal-opposition-cosine "$bidex_max_normal_opposition_cosine" \
      --minimum-radial-normal-alignment "$bidex_min_radial_normal_alignment" \
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
      --minimum-pair-clearance-mm 40 \
      --maximum-pair-clearance-mm 500 \
      --target-pair-clearance-mm 220 \
      --maximum-horizontal-root-cosine -0.80 \
      --maximum-root-z-mm 280 \
      --maximum-root-height-difference-mm 55 \
      --contact-mm 2 \
      --penetration-mm 2 \
      --minimum-contact-links 2 \
      --minimum-contact-points 3 \
      --minimum-contact-digits 1 \
      --target-contact-links 3 \
      --target-contact-points 14 \
      --target-contact-digits 2 \
      --target-contact-spread-mm 20 \
      --support-clearance-mm 1 \
      --friction "$friction" \
      --stop-after-candidates "$bidex_stop_after_candidates" \
      --seed "$bidex_candidate_seed" \
      --output "$candidate" \
      --audit-json "$audit"
    source_vis="$candidate"
  fi
  pairs_per_object=36
  roll_count=1
else
  pairs_per_object=8
  roll_count=16
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
  --lift-height 0.10
  --lift-step 150
  --min-lift-height 0.02
  --support-during-closure
  --no-fixture-during-closure
  --opposition-mode tabletop
  --tabletop-left-roll-degrees 0
  --tabletop-root-height-mm 60
  --max-root-height-fraction 0.80
  --min-object-horizontal-span-mm 360
  --lateral-max-root-z-mm 280
  --lateral-max-height-diff-mm 55
  --independent-directions
  --penetration-mm 2
  --contact-mm 2
  --min-contact-links 1
  --robot-friction "$friction"
  --object-friction "$friction"
  --finger-effort-limit "$effort"
  --contact-offset "$contact_offset"
  --object-vhacd
  --object-vhacd-high-v1
  --object-vhacd-resolution 1000000
  --object-vhacd-max-convex-hulls 128
  --object-vhacd-max-vertices 64
  --object-density "$density"
  --max-gravity-displacement 0.0125
  --max-direction-displacement 0.015
  --seed "$smoke_seed"
  --gpu 0
  --output-dir "$root"
  --force
)
if [[ "$method" == baseline ]]; then
  generator_args+=(
    --symmetric-source
    --right-max-outward-mm 300
    --right-outward-step-mm 30
    --radial-fine-step-mm 3
    --radial-min-offset-mm -24
  )
fi
"$tro_python" generate_bimanual_pilot.py "${generator_args[@]}"

"$tro_python" migration_4090/select_bulk_repeat_candidates.py \
  --source-dataset "$root/bimanual_dataset.pt" \
  --output "$selection" \
  --max-samples 8

repeat_args=(
  --repo "$repo"
  --source-dataset "$selection"
  --output-dir "$verified"
  --isaac-python "$isaac_python"
  --additional-repeats 2
  --batch-size 8
  --geometry-batch-size 2
  --gpu 0
  --object-name "$object"
  --object-density "$density"
  --finger-effort-limit "$effort"
  --lift-height 0.10
  --lift-step 150
  --min-lift-height 0.02
  --robot-friction "$friction"
  --object-friction "$friction"
  --contact-offset "$contact_offset"
  --object-vhacd
  --object-vhacd-high-v1
  --object-vhacd-resolution 1000000
  --object-vhacd-max-convex-hulls 128
  --object-vhacd-max-vertices 64
  --max-gravity-displacement 0.0125
  --max-direction-displacement 0.015
  --min-contact-links 1
  --geometry-contact-mm 2
)
"$tro_python" scripts/repeat_verify_xlarge_batch.py "${repeat_args[@]}"

"$tro_python" scripts/render_verified_bimanual.py \
  --dataset "$verified/verified_dataset.pt" \
  --sample-index 0 \
  --pose-stage final \
  --show-table \
  --contact-csv "$render_root/contacts.csv" \
  --output "$render_root/final_three_views.png"

echo "LARGE_RANDOM_V1_SMOKE_${method^^}_COMPLETE"
