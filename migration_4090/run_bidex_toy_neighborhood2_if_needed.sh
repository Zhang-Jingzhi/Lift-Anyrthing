#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
gpu=2
seed=20994835
root="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/ycb_toy_airplane"
output="$root/seed_$seed"
source="$repo/migration_4090/bulk_candidates/bidex_v1/ycb_toy_airplane/local_neighborhood1296_v2.pt"
cd "$repo"

while screen -ls 2>/dev/null | grep -q '[.]bidex_toy_neighborhood[[:space:]]'; do
  echo "waiting for bidex_toy_neighborhood"
  sleep 30
done
count=$(
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
)
echo "first-neighborhood-complete unique_verified=$count"
if [[ "$count" -ge 100 ]]; then
  echo "target already met; second neighborhood not required"
  exit 0
fi

test ! -e "$output"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python generate_bimanual_pilot.py \
  --source-vis "$source" \
  --objects ycb+toy_airplane \
  --pairs-per-object 1296 \
  --roll-count 1 \
  --isaac-batch-size 16 \
  --realized-batch-size 2 \
  --left-robot-name allegro_left \
  --right-robot-name allegro_right \
  --gravity 9.8 \
  --gravity-settle-step 500 \
  --support-during-closure \
  --no-fixture-during-closure \
  --opposition-mode tabletop \
  --lateral-max-root-z-mm 100 \
  --lateral-max-height-diff-mm 45 \
  --independent-directions \
  --penetration-mm 2 \
  --contact-mm 12 \
  --min-contact-links 1 \
  --robot-friction 1 \
  --object-friction 1 \
  --finger-effort-limit 0.7 \
  --contact-offset 0.002 \
  --object-density 160 \
  --max-gravity-displacement 0.01 \
  --max-direction-displacement 0.015 \
  --seed "$seed" \
  --gpu 0 \
  --output-dir "$output" \
  --force

remaining=$((100 - count))
select_count=$((remaining * 2 + 20))
if [[ "$select_count" -gt 240 ]]; then select_count=240; fi
bash migration_4090/repeat_verify_bulk_seed.sh \
  "$gpu" bidex ycb+toy_airplane 160 "$seed" "$select_count" 1 12
final_count=$(
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
)
echo "second-neighborhood-complete unique_verified=$final_count"
