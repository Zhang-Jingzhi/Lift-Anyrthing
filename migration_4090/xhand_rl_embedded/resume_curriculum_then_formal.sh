#!/usr/bin/env bash
set -euo pipefail

object="$1"
curriculum_root="$2"
device="$3"
nominal="$4"
kit_root="$5"

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"

latest_name="$(find "$curriculum_root" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
latest_checkpoint="${curriculum_root}/${latest_name}"

if [[ ! -f "$curriculum_root/CURRICULUM_TRAINING_COMPLETE.json" ]]; then
  if [[ -z "$latest_checkpoint" ]]; then
    echo "no resumable curriculum checkpoint for $object" >&2
    exit 2
  fi
  echo "resuming curriculum for $object from $latest_checkpoint"
  env PYTHONPATH="$repo" "$isaac_python" -m migration_4090.xhand_rl_curriculum.train \
    --manifest "$manifest" --object "$object" --nominal-dataset "$nominal" \
    --output "$curriculum_root" --num-envs 128 --max-iterations 400 \
    --seed 184 --device "$device" --headless \
    --kit-portable-root "$kit_root/curriculum" \
    --init-noise-std 0.03 --entropy-coef 0.001 --freeze-policy-noise \
    --residual-activation-phase 1 --resume-checkpoint "$latest_checkpoint"
fi

checkpoint="$($isaac_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' \
  "$curriculum_root/CURRICULUM_TRAINING_COMPLETE.json")"
[[ -f "$checkpoint" ]] || { echo "curriculum checkpoint missing for $object: $checkpoint" >&2; exit 3; }

exec env PYTHONPATH="$repo" "$isaac_python" -m migration_4090.xhand_rl_embedded.bridge_worker \
  --manifest "$manifest" --object "$object" --nominal "$nominal" \
  --root "$root" --isaac-python "$isaac_python" --device "$device" \
  --target 100 --curriculum-iterations 400 --formal-iterations 4000 \
  --curriculum-envs 128 --formal-envs 128 --collect-envs 128 \
  --skip-curriculum --seed-checkpoint "$checkpoint" \
  --kit-portable-root "$kit_root/formal"
