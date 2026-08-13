#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?gpu}"
METHOD="${2:?method}"
INDEX="${3:?index}"
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002
export XHAND_SIM_SUBSTEPS=4
export XHAND_PHYSX_POS_ITERS=16
export XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027
export XHAND_ROBOT_BASE_Y=0.00042
export XHAND_ROBOT_BASE_Z=0.0524
DATA=migration_4090/results/xhand_targeted_pose_search_v1/${METHOD}.pt
OUT=migration_4090/results/xhand_targeted_pose_search_v1/physics_full
mkdir -p "$OUT"
REPORT="$OUT/${METHOD}_$(printf '%03d' "$INDEX").json"
"$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset "$DATA" --sample-index "$INDEX" --output "$REPORT" --gpu "$GPU" \
  --modes both,left,right --density 100 --friction 2.0 --closure-mode teleport \
  --lift-steps 100 --gravity-steps 500 --disturbance-steps 100
