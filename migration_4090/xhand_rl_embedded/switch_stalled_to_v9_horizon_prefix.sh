#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
log="$root/logs/shaped_v9_horizon_prefix_switch.log"

sphere_small_seed="$root/runs/sphere_small/curriculum_retry_004/model_349.pt"
cracker_large_seed="$root/runs/cracker_large/training_retry_011/model_1600.pt"

for path in "$python" "$manifest" "$sphere_small_seed" "$cracker_large_seed"; do
  if [[ ! -f "$path" ]]; then
    echo "$(date --iso-8601=seconds) missing required file: $path" >> "$log"
    exit 2
  fi
done

echo "$(date --iso-8601=seconds) v9 preflight passed" >> "$log"

for screen_name in \
  xhand_formal_v8_strict_sphere_small \
  xhand_formal_v8_strict_cracker_large; do
  screen -S "$screen_name" -X quit >/dev/null 2>&1 || true
done

echo "$(date --iso-8601=seconds) stopped stalled v8 workers; cooldown_s=90" >> "$log"
sleep 90

launch_bridge() {
  local object="$1"
  local device="$2"
  local seed="$3"
  local formal_seed="$4"
  local nominal="$5"
  local screen_name="xhand_formal_v9_prefix_${object}"
  local object_log="$root/logs/$object/shaped_v9_horizon_prefix_bridge_screen.log"

  screen -L -Logfile "$object_log" -dmS "$screen_name" \
    env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
    "$python" -m migration_4090.xhand_rl_embedded.bridge_worker \
      --manifest "$manifest" \
      --object "$object" \
      --nominal "$nominal" \
      --root "$root" \
      --isaac-python "$python" \
      --device "$device" \
      --target 100 \
      --curriculum-iterations 400 \
      --formal-iterations 4000 \
      --curriculum-envs 128 \
      --formal-envs 128 \
      --collect-envs 128 \
      --curriculum-seed "$formal_seed" \
      --curriculum-noise-std 0.03 \
      --formal-seed "$formal_seed" \
      --residual-activation-phase 1 \
      --kit-portable-root "/tmp/xhand_rl_embedded_kit/v9_prefix/$object" \
      --skip-curriculum \
      --prefer-seed-checkpoint \
      --seed-checkpoint "$seed"

  sleep 20
  if ! ps -ww -eo args= | rg -q \
    "migration_4090\.xhand_rl_embedded\.bridge_worker.*--object $object.*v9_prefix/$object"; then
    echo "$(date --iso-8601=seconds) failed to verify object=$object" >> "$log"
    exit 3
  fi
  echo "$(date --iso-8601=seconds) launched object=$object screen=$screen_name device=$device" >> "$log"
}

launch_bridge \
  sphere_small cuda:1 "$sphere_small_seed" 1485 \
  "/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"

launch_bridge \
  cracker_large cuda:2 "$cracker_large_seed" 1486 \
  "/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"

echo "$(date --iso-8601=seconds) all v9 stalled-object repairs verified" >> "$log"
