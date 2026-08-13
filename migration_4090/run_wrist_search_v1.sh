#!/usr/bin/env bash
set -u
export JAX_PLATFORMS=cpu
export XHAND_FULLBODY_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf"
export XHAND_FULLBODY_IK_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf"
for s in 0.67 0.68 0.69 0.70 0.71; do
  out="migration_4090/results/xhand_wrist_search_v1/right_${s}"
  mkdir -p "$out"
  echo "=== generate right_wrist_side=$s ==="
  conda run --prefix /media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro \
    python migration_4090/generate_xhand_fullbody_grasps_v1.py \
    --output-dir "$out" \
    --object-name ycb+pitcher_base_formal_large_random_v1_000 \
    --smoke --smoke-count 1 --target-per-object 1 --method baseline \
    --seed 20260808 --left-wrist-side 0.69 --right-wrist-side "$s" \
    --accept-ik-candidates || exit $?
done
