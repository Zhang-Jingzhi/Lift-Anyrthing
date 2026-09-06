#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
source="$root/runs/pyramid/training_retry_022"
seed="$source/hard_prefix_preupdate_iter_744_level_4.pt"
output="$root/runs/pyramid/training_retry_023"
log="$root/logs/pyramid/shaped_v11_disturbance_prefix_switch.log"

for path in "$python" "$manifest" "$nominal" "$seed"; do
  [[ -f "$path" ]] || { echo "$(date --iso-8601=seconds) missing required file: $path" >> "$log"; exit 2; }
done
[[ ! -e "$output" ]] || { echo "$(date --iso-8601=seconds) refusing existing output: $output" >> "$log"; exit 2; }

PYTHONPATH="$repo" "$python" - "$source" "$seed" <<'PY'
import json, sys
from pathlib import Path
from migration_4090.xhand_rl_embedded.contracts import resume_checkpoint_iteration, sha256_file
source, seed = map(lambda x: Path(x).resolve(), sys.argv[1:])
rows = [json.loads(line) for line in (source / "hard_prefix_rollouts.jsonl").read_text().splitlines() if line.strip()]
if not any(Path(row["checkpoint"]).resolve() == seed and row.get("saved_before_ppo_update") is True and int(row.get("max_level", 0)) >= 4 and row.get("checkpoint_sha256") == sha256_file(seed) for row in rows):
    raise SystemExit("no hash-verified level-4 prefix evidence")
if resume_checkpoint_iteration(seed) != 744:
    raise SystemExit("unexpected source iteration")
PY

screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v10_prefix_pyramid -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped pyramid retry_022; cooldown_s=90" >> "$log"
sleep 90
if ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_(embedded\.bridge_worker|embedded\.train|curriculum\.train).*--object pyramid'; then
  echo "$(date --iso-8601=seconds) old pyramid Isaac process still alive after cooldown" >> "$log"; exit 3
fi

screen -L -Logfile "$root/logs/pyramid/shaped_v11_disturbance_prefix_training_screen.log" \
  -dmS xhand_formal_v11_prefix_pyramid \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 "$python" -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" --object pyramid --nominal-dataset "$nominal" --output "$output" \
    --num-envs 128 --max-iterations 4744 --seed 3788 --device cuda:5 --headless \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v11_prefix/pyramid \
    --penetration-reward-weight -30.0 --terminal-success-weight 2500.0 --gamma 0.999 \
    --init-noise-std 0.03 --entropy-coef 0.0001 --resume-checkpoint "$seed" \
    --residual-activation-phase 0 --instantaneous-residual-weight 0.0 \
    --residual-integration 0.006 --residual-limit-rad 0.15 \
    --arm-action-scale-rad 0.10 --hand-action-scale-rad 0.12 \
    --stable-lift-reward-weight 24.0 --force-closure-reward-weight 24.0 \
    --hard-gate-frontier-reward-weight 40.0 --force-closure-shape-all-hold-contacts \
    --reset-optimizer-on-resume --learning-rate 1e-5 --freeze-policy-noise

sleep 20
if ! ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_embedded\.train.*--object pyramid.*training_retry_023'; then
  echo "$(date --iso-8601=seconds) failed to verify pyramid retry_023" >> "$log"; exit 4
fi
echo "$(date --iso-8601=seconds) launched pyramid training_retry_023 from level-4 prefix" >> "$log"
screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
