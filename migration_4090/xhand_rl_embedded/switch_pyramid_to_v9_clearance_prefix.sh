#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
seed="$root/runs/pyramid/training_retry_020/model_700.pt"
output="$root/runs/pyramid/training_retry_021"
log="$root/logs/pyramid/shaped_v9_clearance_prefix_switch.log"

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
screen -S xhand_formal_v8_strict_pyramid -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped pyramid retry_020; cooldown_s=90" >> "$log"
sleep 90

screen -L -Logfile "$root/logs/pyramid/shaped_v9_clearance_prefix_training_screen.log" \
  -dmS xhand_formal_v9_prefix_pyramid_clearance \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" \
    --object pyramid \
    --nominal-dataset "$nominal" \
    --output "$output" \
    --num-envs 128 \
    --max-iterations 4700 \
    --seed 1488 \
    --device cuda:5 \
    --headless \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v9_prefix/pyramid_clearance \
    --penetration-reward-weight -20.0 \
    --terminal-success-weight 2500.0 \
    --gamma 0.999 \
    --init-noise-std 0.03 \
    --entropy-coef 0.0001 \
    --resume-checkpoint "$seed" \
    --residual-activation-phase 0 \
    --instantaneous-residual-weight 0.0 \
    --residual-integration 0.01 \
    --residual-limit-rad 0.15 \
    --arm-action-scale-rad 0.20 \
    --hand-action-scale-rad 0.20 \
    --stable-lift-reward-weight 16.0 \
    --force-closure-reward-weight 12.0 \
    --hard-gate-frontier-reward-weight 20.0 \
    --force-closure-shape-all-hold-contacts \
    --reset-optimizer-on-resume \
    --learning-rate 5e-5 \
    --freeze-policy-noise

sleep 20
if ! ps -ww -eo args= | rg -q \
  'migration_4090\.xhand_rl_embedded\.train.*--object pyramid.*training_retry_021'; then
  echo "$(date --iso-8601=seconds) failed to verify pyramid clearance retry" >> "$log"
  exit 3
fi
echo "$(date --iso-8601=seconds) launched pyramid training_retry_021" >> "$log"

screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" \
  -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
