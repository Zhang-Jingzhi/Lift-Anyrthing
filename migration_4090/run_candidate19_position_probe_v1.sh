#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?gpu}" SCALE="${2:?scale}" STIFF="${3:?stiffness}" DAMP="${4:?damping}" EFFORT="${5:?effort}" TAG="${6:?tag}"
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4 XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027 XHAND_ROBOT_BASE_Y=0.00042 XHAND_ROBOT_BASE_Z=0.0524
OUT=migration_4090/results/xhand_targeted_pose_search_v2/position_probe
mkdir -p "$OUT"
"$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset migration_4090/results/xhand_targeted_pose_search_v2/baseline.pt --sample-index 19 \
  --output "$OUT/${TAG}.json" --gpu "$GPU" --modes both \
  --density 100 --friction 2.0 --hand-close-scale "$SCALE" --closure-mode position \
  --hand-stiffness "$STIFF" --hand-damping "$DAMP" --hand-effort-override "$EFFORT" \
  --lift-steps 100 --gravity-steps 500 --disturbance-steps 100
