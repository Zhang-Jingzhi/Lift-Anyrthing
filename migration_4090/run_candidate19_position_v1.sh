#!/usr/bin/env bash
set -uo pipefail
GPU="${1:?gpu}"
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4 XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027 XHAND_ROBOT_BASE_Y=0.00042 XHAND_ROBOT_BASE_Z=0.0524
OUT=migration_4090/results/xhand_targeted_pose_search_v2/final_gate
mkdir -p "$OUT"
"$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset migration_4090/results/xhand_targeted_pose_search_v2/baseline.pt --sample-index 19 \
  --output "$OUT/candidate19_position.json" --gpu "$GPU" --modes both \
  --density 100 --friction 2.0 --hand-close-scale 1.05 --closure-mode position \
  --hand-stiffness 4000 --hand-damping 300 --hand-effort-override 100 \
  --lift-steps 100 --gravity-steps 500 --disturbance-steps 100
