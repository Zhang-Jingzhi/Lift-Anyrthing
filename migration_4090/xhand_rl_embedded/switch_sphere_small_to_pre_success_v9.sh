#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
seed="$root/runs/sphere_small/curriculum_retry_004/model_300.pt"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"
log="$root/logs/sphere_small/shaped_v9_pre_success_switch.log"

for path in "$python" "$manifest" "$seed" "$nominal"; do
  if [[ ! -f "$path" ]]; then
    echo "$(date --iso-8601=seconds) missing required file: $path" >> "$log"
    exit 2
  fi
done

screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v9_prefix_sphere_small -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped model_349 retry; cooldown_s=90" >> "$log"
sleep 90

screen -L -Logfile "$root/logs/sphere_small/shaped_v9_pre_success_bridge_screen.log" \
  -dmS xhand_formal_v9_prefix_sphere_small_pre_success \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.bridge_worker \
    --manifest "$manifest" \
    --object sphere_small \
    --nominal "$nominal" \
    --root "$root" \
    --isaac-python "$python" \
    --device cuda:1 \
    --target 100 \
    --curriculum-iterations 400 \
    --formal-iterations 4000 \
    --curriculum-envs 128 \
    --formal-envs 128 \
    --collect-envs 128 \
    --curriculum-seed 2485 \
    --curriculum-noise-std 0.03 \
    --formal-seed 2485 \
    --residual-activation-phase 1 \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v9_prefix/sphere_small_pre_success \
    --skip-curriculum \
    --prefer-seed-checkpoint \
    --seed-checkpoint "$seed"

sleep 20
if ! ps -ww -eo args= | rg -q \
  'migration_4090\.xhand_rl_embedded\.bridge_worker.*--object sphere_small.*sphere_small_pre_success'; then
  echo "$(date --iso-8601=seconds) failed to verify pre-success sphere_small bridge" >> "$log"
  exit 3
fi
echo "$(date --iso-8601=seconds) launched model_300 pre-success sphere_small retry" >> "$log"

screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" \
  -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
