#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?gpu}"
shift
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4 XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027 XHAND_ROBOT_BASE_Y=0.00042 XHAND_ROBOT_BASE_Z=0.0524
OUT=migration_4090/results/xhand_targeted_pose_search_v1/physics_screen_v2
mkdir -p "$OUT"
for item in "$@"; do
  method="${item%%:*}"; index="${item##*:}"
  data=migration_4090/results/xhand_targeted_pose_search_v1/${method}.pt
  report="$OUT/${method}_$(printf '%03d' "$index").json"
  "$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
    --dataset "$data" --sample-index "$index" --output "$report" --gpu "$GPU" \
    --modes both --density 100 --friction 2.0 --closure-mode teleport \
    --lift-steps 30 --gravity-steps 100 --disturbance-steps 20
done
