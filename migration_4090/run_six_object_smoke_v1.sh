#!/usr/bin/env bash
set -u

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
ROOT="$REPO/migration_4090/xhand_six_object_smoke_v1_v3"
mkdir -p "$ROOT/reports" "$ROOT/logs" "$ROOT/materialized"

objects=(sphere cube cracker bleach pitcher drill)
methods=(baseline bidex_v3)
job=0
for obj in "${objects[@]}"; do
  for method in "${methods[@]}"; do
    gpu=$((job % 8)); job=$((job + 1))
    bank="$ROOT/banks/${obj}_${method}.pt"
    work="$ROOT/materialized/${obj}_${method}_c0"
    dataset="$work/sample.pt"
    report="$ROOT/reports/${obj}_${method}_c0_both.json"
    log="$ROOT/logs/${obj}_${method}_c0_both.log"
    if [[ -s "$report" ]]; then
      echo "SKIP_EXISTING $obj $method $report"
      continue
    fi
    (
      set -u
      export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
      "$TRO" "$REPO/migration_4090/objectflow_pose_preview/materialize_isaac_smoke.py" \
        --bank "$bank" --index 0 --asset-dir "$work/assets" --output "$dataset" > "$log" 2>&1 || exit 10
      density=$("$TRO" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["samples"][0]["physical_parameters"]["recommended_density_kg_m3"])' "$dataset")
      [[ -n "$density" ]] || { echo "DENSITY_ERROR $obj $method"; exit 11; }
      export PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin:/usr/bin:/bin
      export LD_LIBRARY_PATH=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/lib
      export PYTHONPATH=/media/home/zhangjingzhi/.tro_grasp_tools/isaacgym-preview4/isaacgym/python
      export CUDA_VISIBLE_DEVICES=$gpu
      export XHAND_SIM_DT=0.002 XHAND_SIM_SUBSTEPS=4
      export XHAND_PHYSX_POS_ITERS=16 XHAND_PHYSX_VEL_ITERS=4
      export XHAND_ROBOT_BASE_X=0 XHAND_ROBOT_BASE_Y=0 XHAND_ROBOT_BASE_Z=0
      export XHAND_ROBOT_COLLISION_FILTER=1
      export XHAND_SUPPORT_SIZE_X=0.8 XHAND_SUPPORT_SIZE_Y=0.8
      export XHAND_SUPPORT_X=0.5 XHAND_SUPPORT_Y=0 XHAND_SUPPORT_Z=0.72
      "$ISAAC" "$REPO/migration_4090/validate_xhand_fullbody_isaac.py" \
        --dataset "$dataset" --sample-index 0 --output "$report" --gpu "$gpu" --modes both \
        --density "$density" --friction 2 --hand-close-scale 1.0 \
        --hand-stiffness 50 --hand-damping 2 --preclosure-settle-steps 100 \
        --closure-steps 600 --closure-settle-steps 300 --lift-steps 100 \
        --gravity-steps 500 --disturbance-steps 100 >> "$log" 2>&1
      echo "SMOKE_DONE object=$obj method=$method gpu=$gpu report=$report"
    ) &
  done
done
wait
echo "SMOKE_BATCH_DONE"
