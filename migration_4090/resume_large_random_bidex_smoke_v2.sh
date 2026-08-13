#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
tro_python="$conda_root/envs/tro/bin/python"
isaac_python="$conda_root/envs/isaac/bin/python"
object=ycb+pitcher_base_formal_large_random_v1_003
density=5.893140826715964
candidate="$repo/migration_4090/large_random_v1_candidates/bidex_v3_pitcher003_v2.pt"
root="$repo/graph_exp/bimanual_data/large_random_v1_smoke/bidex_v3/pitcher003_v2"
selection="$repo/migration_4090/results/large_random_v1_smoke/bidex_v3_pitcher003_v2_selection.pt"
verified="$root/repeat_verified"
render_root="$repo/migration_4090/renders/large_random_v1_smoke/bidex_v3_pitcher003_v2"

test -s "$candidate"
test ! -e "$root"
test ! -e "$selection"
test ! -e "$render_root/final_three_views.png"
mkdir -p "$render_root"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=4
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

"$tro_python" generate_bimanual_pilot.py \
  --source-vis "$candidate" --objects "$object" \
  --pairs-per-object 36 --roll-count 1 \
  --isaac-batch-size 16 --realized-batch-size 4 \
  --left-robot-name allegro_left --right-robot-name allegro_right \
  --gravity 9.8 --gravity-settle-step 500 --min-lift-height 0.02 \
  --support-during-closure --no-fixture-during-closure \
  --opposition-mode tabletop --tabletop-left-roll-degrees 0 \
  --tabletop-root-height-mm 60 --max-root-height-fraction 0.80 \
  --min-object-horizontal-span-mm 360 \
  --lateral-max-root-z-mm 280 --lateral-max-height-diff-mm 55 \
  --independent-directions --penetration-mm 2 --contact-mm 2 \
  --min-contact-links 1 --robot-friction 1.2 --object-friction 1.2 \
  --finger-effort-limit 1.2 --contact-offset 0.003 \
  --object-density "$density" --max-gravity-displacement 0.0125 \
  --max-direction-displacement 0.015 --seed 20261004 --gpu 0 \
  --output-dir "$root" --force

"$tro_python" migration_4090/select_bulk_repeat_candidates.py \
  --source-dataset "$root/bimanual_dataset.pt" \
  --output "$selection" --max-samples 8

selected_count=$("$tro_python" -c \
  "import torch; print(len(torch.load('$selection', map_location='cpu', weights_only=False)['samples']))")
if (( selected_count == 0 )); then
  echo "No strict samples selected from the restored v2 candidate set" >&2
  exit 1
fi

"$tro_python" scripts/repeat_verify_xlarge_batch.py \
  --repo "$repo" --source-dataset "$selection" --output-dir "$verified" \
  --isaac-python "$isaac_python" --additional-repeats 2 \
  --batch-size 8 --geometry-batch-size 2 --gpu 0 \
  --object-name "$object" --object-density "$density" \
  --finger-effort-limit 1.2 --lift-height 0.05 --min-lift-height 0.02 \
  --robot-friction 1.2 --object-friction 1.2 --contact-offset 0.003 \
  --max-gravity-displacement 0.0125 --max-direction-displacement 0.015 \
  --min-contact-links 1 --geometry-contact-mm 2

"$tro_python" migration_4090/validate_large_random_smoke_gate.py \
  --baseline "$repo/graph_exp/bimanual_data/large_random_v1_smoke/baseline/pitcher003_v1/repeat_verified/verified_dataset.pt" \
  --bidex "$verified/verified_dataset.pt" \
  --output "$repo/migration_4090/results/large_random_v1_smoke/smoke_gate.json"

"$tro_python" scripts/render_verified_bimanual.py \
  --dataset "$verified/verified_dataset.pt" --sample-index 0 \
  --pose-stage final --show-table \
  --contact-csv "$render_root/contacts.csv" \
  --output "$render_root/final_three_views.png"

echo "LARGE_RANDOM_V1_SMOKE_BIDEX_V3_RESTORED_V2_COMPLETE"
