#!/usr/bin/env bash
set -euo pipefail

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260825_pyramid200_armshift/nominal/pyramid.pt"
curriculum_complete="$root/runs/pyramid/curriculum_retry_pyramid200_001/CURRICULUM_TRAINING_COMPLETE.json"

while [[ ! -f "$curriculum_complete" ]]; do
  sleep 30
done

checkpoint="$($isaac_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$curriculum_complete")"
exec env PYTHONPATH="$repo" "$isaac_python" -m migration_4090.xhand_rl_embedded.bridge_worker \
  --manifest "$manifest" \
  --object pyramid \
  --nominal "$nominal" \
  --root "$root" \
  --isaac-python "$isaac_python" \
  --device cuda:5 \
  --target 100 \
  --curriculum-iterations 400 \
  --formal-iterations 4000 \
  --curriculum-envs 128 \
  --formal-envs 128 \
  --collect-envs 128 \
  --seed-checkpoint "$checkpoint" \
  --skip-curriculum \
  --prefer-seed-checkpoint \
  --kit-portable-root /tmp/xhand_rl_embedded_kit/v5r/pyramid200 \
  >> "$root/logs/pyramid/pyramid200_chain.log" 2>&1
