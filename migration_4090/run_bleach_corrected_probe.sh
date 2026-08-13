#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?gpu}" TAG="${2:?tag}" MODES="${3:-both}"
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
# The grasp sample and Pyroki targets were generated against the user's
# original external approx_v1 asset.  Using inward_v1 here introduces a
# systematic 20--60 mm EE mismatch, so keep physics and IK on this exact pair.
export XHAND_FULLBODY_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4 XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027 XHAND_ROBOT_BASE_Y=0.00042 XHAND_ROBOT_BASE_Z=0.0
OUT=migration_4090/results/xhand_external_width_1p36/external_asset_corrected_gate/${TAG}
mkdir -p "$OUT"
"$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
  --dataset migration_4090/results/xhand_external_width_1p36/baseline__ycb__bleach_cleanser_formal_large_random_v1_005.pt \
  --sample-index 0 --output "$OUT/report.json" --gpu "$GPU" --modes "$MODES" \
  --density 100 --friction 2.0 --hand-close-scale 1.0 --closure-mode teleport \
  --lift-steps 100 --gravity-steps 500 --disturbance-steps 100
