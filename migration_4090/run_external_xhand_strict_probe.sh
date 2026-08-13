#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu}"
DATASET="${2:?dataset}"
INDEX="${3:?sample index}"
OUTPUT="${4:?output report}"
MODES="${5:-both,left,right}"
CLOSE_SCALE="${6:-1.0}"
LIFT_STEPS="${7:-100}"
GRAVITY_STEPS="${8:-500}"
DISTURBANCE_STEPS="${9:-100}"
CLOSURE_MODE="${10:-teleport}"
DENSITY="${11:-100}"
FRICTION="${12:-2.0}"
HAND_EFFORT_OVERRIDE="${13:-}"
LEFT_CLOSE_SCALE="${14:-}"
RIGHT_CLOSE_SCALE="${15:-}"
CLOSURE_STEPS="${16:-120}"
CLOSURE_SETTLE_STEPS="${17:-0}"
PRECLOSURE_SETTLE_STEPS="${18:-50}"
HAND_STIFFNESS="${19:-400}"
HAND_DAMPING="${20:-80}"
APPROACH_STEPS="${21:-0}"
APPROACH_FRACTION="${22:-1.0}"
SQUEEZE_STEPS="${23:-0}"
SQUEEZE_FRACTION="${24:-}"
LEFT_SQUEEZE_FRACTION="${25:-}"
RIGHT_SQUEEZE_FRACTION="${26:-}"

cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}

# Final asset selected by the user.  Candidate generation and Isaac validation
# must use this exact pair; repository-local approx_v1 and inward_v1 differ.
export XHAND_FULLBODY_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4
export XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027 XHAND_ROBOT_BASE_Y=0.00042 XHAND_ROBOT_BASE_Z=0.0

mkdir -p "$(dirname "$OUTPUT")"
effort_args=()
if [[ -n "$HAND_EFFORT_OVERRIDE" ]]; then
  effort_args+=(--hand-effort-override "$HAND_EFFORT_OVERRIDE")
fi
side_scale_args=()
if [[ -n "$LEFT_CLOSE_SCALE" ]]; then
  side_scale_args+=(--left-hand-close-scale "$LEFT_CLOSE_SCALE")
fi
if [[ -n "$RIGHT_CLOSE_SCALE" ]]; then
  side_scale_args+=(--right-hand-close-scale "$RIGHT_CLOSE_SCALE")
fi
squeeze_args=()
if [[ -n "$SQUEEZE_FRACTION" ]]; then
  squeeze_args+=(--squeeze-fraction "$SQUEEZE_FRACTION")
fi
if [[ -n "$LEFT_SQUEEZE_FRACTION" ]]; then
  squeeze_args+=(--left-squeeze-fraction "$LEFT_SQUEEZE_FRACTION")
fi
if [[ -n "$RIGHT_SQUEEZE_FRACTION" ]]; then
  squeeze_args+=(--right-squeeze-fraction "$RIGHT_SQUEEZE_FRACTION")
fi
"$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset "$DATASET" --sample-index "$INDEX" --output "$OUTPUT" \
  --gpu "$GPU" --modes "$MODES" --density "$DENSITY" --friction "$FRICTION" \
  --hand-close-scale "$CLOSE_SCALE" --closure-mode "$CLOSURE_MODE" \
  --hand-stiffness "$HAND_STIFFNESS" --hand-damping "$HAND_DAMPING" \
  --preclosure-settle-steps "$PRECLOSURE_SETTLE_STEPS" \
  --approach-steps "$APPROACH_STEPS" \
  --approach-fraction "$APPROACH_FRACTION" \
  --squeeze-steps "$SQUEEZE_STEPS" \
  --closure-steps "$CLOSURE_STEPS" \
  --closure-settle-steps "$CLOSURE_SETTLE_STEPS" \
  --lift-steps "$LIFT_STEPS" --gravity-steps "$GRAVITY_STEPS" \
  --disturbance-steps "$DISTURBANCE_STEPS" \
  "${effort_args[@]}" "${side_scale_args[@]}" "${squeeze_args[@]}"
