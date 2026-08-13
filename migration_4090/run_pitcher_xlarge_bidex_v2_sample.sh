#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
tro_python="$conda_root/envs/tro/bin/python"
isaac_python="$conda_root/envs/isaac/bin/python"
object=ycb+pitcher_base_xlarge_ref_v1
run_tag=${1:-v1}
root="$repo/graph_exp/bimanual_data/pitcher_xlarge_bidex_v2_sample_$run_tag"
candidate="$repo/migration_4090/bidex_v2_candidates/pitcher_xlarge_sample_$run_tag.pt"
audit="$repo/migration_4090/results/pitcher_xlarge_bidex_v2_candidate_$run_tag.json"
selection="$repo/migration_4090/results/pitcher_xlarge_bidex_v2_selection_$run_tag.pt"
verified="$root/repeat_verified"
render_root="$repo/migration_4090/renders/pitcher_xlarge_bidex_v2_sample_$run_tag"

for path in "$root" "$candidate" "$audit" "$selection" "$render_root"; do
  test ! -e "$path"
done
mkdir -p "$(dirname "$candidate")" "$(dirname "$audit")" "$render_root"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=7
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

"$tro_python" migration_4090/generate_bidexgrasp_candidates.py \
  --source-vis migration_4090/derived/source_vis_pitcher_base_xlarge_ref_v1.pt \
  --object "$object" \
  --minimum-object-horizontal-span-mm 220 \
  --minimum-object-height-mm 290 \
  --surface-points 2048 \
  --anchors 128 \
  --region-points 192 \
  --gws-contacts 8 \
  --region-pairs 20 \
  --region-scale 0.22 \
  --enhanced-bidex-v2 \
  --v2-refinement-levels 1 \
  --v2-joint-steps-degrees \
  --v2-aperture-scales 0.5 1.0 1.5 2.0 2.5 \
  --maximum-normal-opposition-cosine -0.15 \
  --minimum-radial-normal-alignment 0.15 \
  --minimum-height-fraction 0.22 \
  --maximum-height-fraction 0.68 \
  --minimum-pair-separation-fraction 0.65 \
  --minimum-vertical-separation-mm 0 \
  --maximum-vertical-separation-mm 25 \
  --minimum-standoff-mm 35 \
  --maximum-standoff-mm 115 \
  --standoff-step-mm 20 \
  --maximum-source-seeds 1 \
  --hand-variants 1 \
  --pair-variants 1 \
  --stop-after-candidates 1 \
  --preserve-standoff-diversity \
  --minimum-pair-clearance-mm 12 \
  --maximum-pair-clearance-mm 45 \
  --target-pair-clearance-mm 25 \
  --contact-mm 2 \
  --penetration-mm 2 \
  --minimum-contact-links 2 \
  --minimum-contact-points 6 \
  --target-contact-links 3 \
  --target-contact-points 16 \
  --support-clearance-mm 1 \
  --friction 1 \
  --seed 20260851 \
  --output "$candidate" \
  --audit-json "$audit"

"$tro_python" generate_bimanual_pilot.py \
  --source-vis "$candidate" \
  --objects "$object" \
  --pairs-per-object 40 \
  --roll-count 1 \
  --isaac-batch-size 16 \
  --realized-batch-size 4 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --lateral-max-root-z-mm 125 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
  --penetration-mm 2 \
  --contact-mm 2 \
  --min-contact-links 2 \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit 0.7 \
  --contact-offset 0.002 \
  --object-density 100 \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --seed 20260851 \
  --gpu 0 \
  --output-dir "$root" \
  --force

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
  --object-density 100 \
  --finger-effort-limit 0.7 \
  --min-contact-links 2 \
  --geometry-contact-mm 2 \
  --require-local-quality

"$tro_python" scripts/render_verified_bimanual.py \
  --dataset "$verified/verified_dataset.pt" \
  --sample-index 0 \
  --pose-stage final \
  --show-table \
  --contact-csv "$render_root/contacts.csv" \
  --output "$render_root/final_three_views.png"

echo BIDEX_V2_XLARGE_SAMPLE_COMPLETE
