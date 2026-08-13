#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
isaac_python="$conda_root/envs/isaac/bin/python"
input="$repo/graph_exp/bimanual_data/4090_irregular_bleach_bidexgrasp_coordinated_strict_density500_v6/ycb__bleach_cleanser/isaac_both_chunks/00000_00001"
output="$repo/migration_4090/results/bidexgrasp_bleach_coordinated_density_sweep_v2"
main_log="$repo/migration_4090/logs/bidexgrasp_bleach_coordinated_density_sweep_v2.log"

test ! -e "$output"
mkdir -p "$output"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$repo"
exec > >(tee "$main_log") 2>&1
for density in 300 350 400 450; do
  for active in both left right; do
    case_dir="$output/density_${density}_${active}"
    mkdir -p "$case_dir"
    extra=(--gravity-only)
    if [[ "$active" == both ]]; then
      extra=(--independent-directions)
    fi
    echo "density=$density active=$active"
    "$isaac_python" validation/bimanual_isaac_main.py \
      --object-name ycb+bleach_cleanser \
      --left-q-file "$input/left_q.pt" \
      --right-q-file "$input/right_q.pt" \
      --output-file "$case_dir/isaac_result.pt" \
      --gpu 0 \
      --left-robot-name allegro_left \
      --right-robot-name allegro_right \
      --gravity 9.8 \
      --gravity-settle-step 500 \
      --active-hands "$active" \
      --lift-height 0.05 \
      --lift-step 100 \
      --min-lift-height 0.03 \
      --robot-friction 1 \
      --object-friction 1 \
      --finger-effort-limit 0.7 \
      --contact-offset 0.002 \
      --max-gravity-displacement 0.01 \
      --max-direction-displacement 0.015 \
      --object-density "$density" \
      --support-during-closure \
      "${extra[@]}" \
      > "$case_dir/isaac.log" 2>&1
  done
done
echo "density sweep complete: $output"
