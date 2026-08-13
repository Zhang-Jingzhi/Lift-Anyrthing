#!/usr/bin/env bash
set -u
GPU="${1:?gpu required}"
START="${2:?start required}"
END="${3:?end required}"
DATASET="migration_4090/results/xhand_local_ik_v1/candidates.pt"
OUTROOT="migration_4090/results/xhand_local_ik_v1/phys_screen"
mkdir -p "$OUTROOT" "migration_4090/logs/xhand_local_ik_v1/phys"
export XHAND_FULLBODY_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf"
export XHAND_FULLBODY_IK_URDF="/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf"
export XHAND_ROBOT_BASE_X=0.0027
export XHAND_ROBOT_BASE_Y=0.00042
export XHAND_ROBOT_BASE_Z=0.0524
export LD_LIBRARY_PATH="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}"
for ((i=START; i<END; i++)); do
  out="$OUTROOT/candidate_$(printf '%03d' "$i").json"
  echo "GPU=$GPU candidate=$i"
  conda run --prefix /media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac \
    python migration_4090/validate_xhand_fullbody_isaac.py \
    --dataset "$DATASET" --sample-index "$i" --output "$out" --gpu "$GPU" \
    --modes both --density 1.0 --friction 1.0 --closure-mode teleport \
    --lift-steps 5 --gravity-steps 5 --disturbance-steps 5 || true
done
