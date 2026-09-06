#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"
seed="$root/runs/sphere_small/curriculum_retry_004/model_300.pt"
output="$root/runs/sphere_small/training_retry_019"
log="$root/logs/sphere_small/shaped_v10_pre_success_switch.log"

for path in "$python" "$manifest" "$nominal" "$seed"; do
  if [[ ! -f "$path" ]]; then
    echo "$(date --iso-8601=seconds) missing required file: $path" >> "$log"
    exit 2
  fi
done
if [[ -e "$output" ]]; then
  echo "$(date --iso-8601=seconds) refusing existing output: $output" >> "$log"
  exit 2
fi

screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v9_prefix_sphere_small_pre_success -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped sphere_small retry_018; cooldown_s=90" >> "$log"
sleep 90

# Do not silently create a conflicting Isaac process if the old bridge did not
# tear down during the cooldown window.
if ps -ww -eo args= | rg -q \
  'migration_4090\.xhand_rl_(embedded\.bridge_worker|embedded\.train|curriculum\.train).*--object sphere_small'; then
  echo "$(date --iso-8601=seconds) old sphere_small Isaac process still alive after cooldown" >> "$log"
  exit 3
fi

screen -L -Logfile "$root/logs/sphere_small/shaped_v10_pre_success_training_screen.log" \
  -dmS xhand_formal_v10_prefix_sphere_small \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" \
    --object sphere_small \
    --nominal-dataset "$nominal" \
    --output "$output" \
    --num-envs 128 \
    --max-iterations 4300 \
    --seed 3485 \
    --device cuda:1 \
    --headless \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v10_prefix/sphere_small \
    --penetration-reward-weight -12.0 \
    --terminal-success-weight 2500.0 \
    --gamma 0.999 \
    --init-noise-std 0.03 \
    --entropy-coef 0.0001 \
    --resume-checkpoint "$seed" \
    --residual-activation-phase 1 \
    --instantaneous-residual-weight 0.0 \
    --residual-integration 0.012 \
    --residual-limit-rad 0.5 \
    --arm-action-scale-rad 0.20 \
    --hand-action-scale-rad 0.35 \
    --stable-lift-reward-weight 16.0 \
    --force-closure-reward-weight 12.0 \
    --hard-gate-frontier-reward-weight 20.0 \
    --force-closure-shape-all-hold-contacts \
    --reset-optimizer-on-resume \
    --learning-rate 5e-5 \
    --freeze-policy-noise

sleep 20
if ! ps -ww -eo args= | rg -q \
  'migration_4090\.xhand_rl_embedded\.train.*--object sphere_small.*training_retry_019'; then
  echo "$(date --iso-8601=seconds) failed to verify sphere_small retry_019" >> "$log"
  exit 4
fi
echo "$(date --iso-8601=seconds) launched sphere_small training_retry_019" >> "$log"

screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" \
  -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
