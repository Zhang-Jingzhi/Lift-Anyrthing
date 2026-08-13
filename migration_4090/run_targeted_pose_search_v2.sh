#!/usr/bin/env bash
set -uo pipefail
cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
export JAX_PLATFORMS=cpu
export XHAND_FULLBODY_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1_fixed_ik.urdf
OBJ=ycb+cracker_box_formal_large_random_v1_002
ROOT=migration_4090/results/xhand_targeted_pose_search_v2
mkdir -p "$ROOT/raw"
run_one() {
  local x="$1" z="$2" side="$3"
  local tag="baseline_x${x//./p}_z${z//./p}_s${side//./p}" out="$ROOT/raw/baseline_x${x//./p}_z${z//./p}_s${side//./p}"
  mkdir -p "$out"
  if compgen -G "$out/baseline*.pt" > /dev/null; then return 0; fi
  echo "GEN $tag"
  "$TRO" migration_4090/generate_xhand_fullbody_grasps_v1.py \
    --output-dir "$out" --object-name "$OBJ" --smoke --smoke-count 1 \
    --target-per-object 1 --method baseline --seed 20260806 \
    --target-y-extent 1.36 --wrist-x "$x" --wrist-z-offset "$z" \
    --left-wrist-side "$side" --right-wrist-side "$side" \
    --accept-ik-candidates > "$out/generate.log" 2>&1 || echo "GEN_FAIL $tag"
}
active=0
for x in 0.008 0.010 0.012; do
  for z in 0.048 0.050 0.052; do
    for side in 0.6825 0.6850 0.6875; do
      run_one "$x" "$z" "$side" &
      active=$((active+1))
      if [ "$active" -ge 6 ]; then wait -n || true; active=$((active-1)); fi
    done
  done
done
wait || true
"$TRO" - <<'PY'
import glob,torch
ps=sorted(glob.glob('migration_4090/results/xhand_targeted_pose_search_v2/raw/baseline_*/*.pt'))
rows=[]
for p in ps: rows.extend(torch.load(p,map_location='cpu')['samples'])
out='migration_4090/results/xhand_targeted_pose_search_v2/baseline.pt'
torch.save({'schema':'xhand_fullbody_grasp_pose_v1','samples':rows},out)
print('DONE',len(rows),out)
PY
