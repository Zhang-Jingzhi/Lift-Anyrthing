#!/usr/bin/env bash
set -uo pipefail
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
export JAX_PLATFORMS=cpu
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
OBJ=ycb+cracker_box_formal_large_random_v1_002
ROOT=migration_4090/results/xhand_targeted_pose_search_v1
mkdir -p "$ROOT/raw"
run_one() {
  local method="$1" x="$2" z="$3" side="$4"
  local tag="${method}_x${x//./p}_z${z//./p}_s${side//./p}" out="$ROOT/raw/${method}_x${x//./p}_z${z//./p}_s${side//./p}"
  mkdir -p "$out"
  if compgen -G "$out/${method}*.pt" > /dev/null; then return 0; fi
  echo "GEN $tag"
  "$TRO" migration_4090/generate_xhand_fullbody_grasps_v1.py \
    --output-dir "$out" --object-name "$OBJ" --smoke --smoke-count 1 \
    --target-per-object 1 --method "$method" --seed 20260806 \
    --target-y-extent 1.36 --wrist-x "$x" --wrist-z-offset "$z" \
    --left-wrist-side "$side" --right-wrist-side "$side" \
    --accept-ik-candidates > "$out/generate.log" 2>&1 || echo "GEN_FAIL $tag"
}
for method in baseline bidex_v3; do
  active=0
  for x in -0.010 0.000 0.010; do
    for z in 0.045 0.050 0.055; do
      for side in 0.685 0.690; do
        run_one "$method" "$x" "$z" "$side" &
        active=$((active+1))
        if [ "$active" -ge 6 ]; then
          wait -n || true
          active=$((active-1))
        fi
      done
    done
  done
  wait || true
done
echo "GENERATION_DONE"
"$TRO" - <<'PY'
import glob,os,torch
root='migration_4090/results/xhand_targeted_pose_search_v1/raw'
for method in ('baseline','bidex_v3'):
    rows=[]
    for p in sorted(glob.glob(root+f'/{method}_*/*{method}*.pt')):
        try: rows.extend(torch.load(p,map_location='cpu')['samples'])
        except Exception as e: print('LOAD_FAIL',p,e)
    out=f'migration_4090/results/xhand_targeted_pose_search_v1/{method}.pt'
    torch.save({'schema':'xhand_fullbody_grasp_pose_v1','samples':rows},out)
    print(method,len(rows),out)
PY
