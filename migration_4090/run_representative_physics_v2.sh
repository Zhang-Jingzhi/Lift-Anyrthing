#!/usr/bin/env bash
set -uo pipefail
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:$PATH
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
export XHAND_SIM_DT=0.002
export XHAND_SIM_SUBSTEPS=4
export XHAND_PHYSX_POS_ITERS=16
export XHAND_PHYSX_VEL_ITERS=4
export XHAND_ROBOT_BASE_X=0.0027
export XHAND_ROBOT_BASE_Y=0.00042
export XHAND_ROBOT_BASE_Z=0.0524
export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib:${LD_LIBRARY_PATH:-}
OUT=migration_4090/results/xhand_external_width136_physics_probe_v2
LOG=migration_4090/logs/xhand_external_width136_physics_probe_v2.log
mkdir -p "$OUT"
for method in baseline bidex_v3; do
  if [ "$method" = baseline ]; then DATA=migration_4090/results/xhand_external_width136_batch_v1_final/baseline.pt; else DATA=migration_4090/results/xhand_external_width136_batch_v1_final/bidex_v3.pt; fi
  for idx in 0 100 200 300 400 500; do
    out="$OUT/${method}_${idx}.json"
    if [ -s "$out" ]; then echo "SKIP $out" | tee -a "$LOG"; continue; fi
    echo "START method=$method index=$idx" | tee -a "$LOG"
    "$ISAAC" migration_4090/validate_xhand_fullbody_isaac.py \
      --dataset "$DATA" --sample-index "$idx" --output "$out" --gpu 0 \
      --modes both --density 100 --friction 2.0 \
      --closure-mode teleport --lift-steps 30 --gravity-steps 100 --disturbance-steps 20 \
      2>&1 | tee -a "$LOG"
    echo "DONE method=$method index=$idx" | tee -a "$LOG"
  done
done
echo "SUMMARY" | tee -a "$LOG"
"/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python" - <<'PY' | tee -a "$LOG"
import glob,json,os
ps=sorted(glob.glob('migration_4090/results/xhand_external_width136_physics_probe_v2/*.json'))
for p in ps:
    try:
        d=json.load(open(p)); b=d.get('runs',{}).get('both',{})
        print(os.path.basename(p), 'physical_pass=',d.get('physical_pass'), 'both=',b.get('physical_pass'), 'lift_mm=',round(1000*b.get('lift_object_displacement_m',0),2), 'gravity_mm=',round(1000*b.get('gravity_displacement_m',0),2), 'dir_max_mm=',round(1000*max(b.get('six_direction_displacements_m',[0])),2))
    except Exception as e: print(p,e)
PY
