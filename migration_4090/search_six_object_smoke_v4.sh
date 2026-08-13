#!/usr/bin/env bash
set -u

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
ISAAC=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/isaac/bin/python
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
ROOT="$REPO/migration_4090/xhand_six_object_smoke_v4"
mkdir -p "$ROOT/reports" "$ROOT/logs" "$ROOT/materialized" "$ROOT/winners"

objects=(sphere cube cracker bleach pitcher drill)
methods=(baseline bidex_v3)
job=0
for obj in "${objects[@]}"; do
  for method in "${methods[@]}"; do
    gpu=$((job % 8)); job=$((job + 1))
    (
      export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
      bank="$REPO/migration_4090/xhand_six_object_smoke_v1_v3/banks/${obj}_${method}.pt"
      winner="$ROOT/winners/${obj}_${method}.json"
      if [[ -s "$winner" ]]; then echo "WINNER_EXISTS $obj $method"; exit 0; fi
      for idx in $(seq 0 23); do
        report="$ROOT/reports/${obj}_${method}_c${idx}_both.json"
        work="$ROOT/materialized/${obj}_${method}_c${idx}"
        dataset="$work/sample.pt"
        log="$ROOT/logs/${obj}_${method}_c${idx}_both.log"
        if [[ ! -s "$report" ]]; then
          echo "TRY_START object=$obj method=$method index=$idx gpu=$gpu" >> "$ROOT/logs/search.log"
          "$TRO" "$REPO/migration_4090/objectflow_pose_preview/materialize_isaac_smoke.py" \
            --bank "$bank" --index "$idx" --asset-dir "$work/assets" --output "$dataset" > "$log" 2>&1 || continue
          density=$("$TRO" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["samples"][0]["physical_parameters"]["recommended_density_kg_m3"])' "$dataset")
          [[ -n "$density" ]] || continue
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
            --dataset "$dataset" --sample-index 0 --output "$report" --gpu 0 --modes both \
            --density "$density" --friction 2 --hand-close-scale 1.0 \
            --hand-stiffness 50 --hand-damping 2 --preclosure-settle-steps 100 \
            --closure-steps 600 --closure-settle-steps 300 --lift-steps 100 \
            --gravity-steps 500 --disturbance-steps 100 >> "$log" 2>&1 || true
        fi
        pass=$("$TRO" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("physical_pass",False)))' "$report" 2>/dev/null || echo 0)
        if [[ "$pass" == 1 ]]; then
          printf '{"object":"%s","method":"%s","candidate_index":%d,"gpu":%d,"dataset":"%s","report":"%s"}\n' "$obj" "$method" "$idx" "$gpu" "$dataset" "$report" > "$winner"
          echo "WINNER object=$obj method=$method index=$idx gpu=$gpu" >> "$ROOT/logs/search.log"
          break
        fi
      done
      [[ -s "$winner" ]] || echo "NO_WINNER object=$obj method=$method gpu=$gpu" >> "$ROOT/logs/search.log"
    ) &
  done
done
wait
echo "SEARCH_DONE"
