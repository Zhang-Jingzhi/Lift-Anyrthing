#!/usr/bin/env bash
set -u
GPU="${1:?gpu required}"
INDEX="${2:?index required}"
OUTROOT="migration_4090/results/xhand_leftwrist_ik_v1/phys"
mkdir -p "$OUTROOT" "migration_4090/logs/xhand_leftwrist_ik_v1/phys"
export XHAND_FULLBODY_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf"
export XHAND_FULLBODY_IK_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf"
export XHAND_ROBOT_BASE_X=0.0027
export XHAND_ROBOT_BASE_Y=0.00042
export XHAND_ROBOT_BASE_Z=0.0524
export LD_LIBRARY_PATH="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}"
conda run --prefix /media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac \
  python migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset migration_4090/results/xhand_leftwrist_ik_v1/candidates_left.pt \
  --sample-index "$INDEX" --output "$OUTROOT/candidate_$(printf '%03d' "$INDEX").json" \
  --gpu "$GPU" --modes both --density 1.0 --friction 1.0 --closure-mode teleport \
  --lift-steps 5 --gravity-steps 5 --disturbance-steps 5
