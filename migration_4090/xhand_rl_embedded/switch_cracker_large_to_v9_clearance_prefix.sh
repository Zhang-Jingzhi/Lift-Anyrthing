#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
source="$root/runs/cracker_large/training_retry_013"
seed="$source/hard_prefix_preupdate_iter_1622_level_2.pt"
output="$root/runs/cracker_large/training_retry_014"
log="$root/logs/cracker_large/shaped_v9_clearance_prefix_switch.log"

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

# Fail closed: use only the immutable checkpoint recorded for the observed
# same-trajectory contact+lift prefix and verify both hash and source iteration.
PYTHONPATH="$repo" "$python" - "$source" "$seed" <<'PY'
import json
import sys
from pathlib import Path

from migration_4090.xhand_rl_embedded.contracts import (
    resume_checkpoint_iteration,
    sha256_file,
)

source = Path(sys.argv[1]).resolve()
seed = Path(sys.argv[2]).resolve()
rows = [
    json.loads(line)
    for line in (source / "hard_prefix_rollouts.jsonl").read_text().splitlines()
    if line.strip()
]
matching = [
    row
    for row in rows
    if Path(row["checkpoint"]).resolve() == seed
    and row.get("saved_before_ppo_update") is True
    and int(row.get("max_level", 0)) >= 2
    and row.get("checkpoint_sha256") == sha256_file(seed)
]
if not matching:
    raise SystemExit("no hash-verified level-2 strict-prefix evidence for seed")
if resume_checkpoint_iteration(seed) != 1622:
    raise SystemExit("unexpected strict-prefix source iteration")
PY

screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v9_prefix_cracker_large -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped cracker_large retry_013; cooldown_s=90" >> "$log"
sleep 90

screen -L -Logfile "$root/logs/cracker_large/shaped_v9_clearance_prefix_training_screen.log" \
  -dmS xhand_formal_v9_prefix_cracker_large_clearance \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.train \
    --manifest "$manifest" \
    --object cracker_large \
    --nominal-dataset "$nominal" \
    --output "$output" \
    --num-envs 128 \
    --max-iterations 5622 \
    --seed 2486 \
    --device cuda:2 \
    --headless \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v9_prefix/cracker_large_clearance \
    --penetration-reward-weight -20.0 \
    --terminal-success-weight 2500.0 \
    --gamma 0.999 \
    --init-noise-std 0.03 \
    --entropy-coef 0.0001 \
    --resume-checkpoint "$seed" \
    --residual-activation-phase 1 \
    --instantaneous-residual-weight 0.0 \
    --residual-integration 0.01 \
    --residual-limit-rad 0.25 \
    --arm-action-scale-rad 0.08 \
    --hand-action-scale-rad 0.10 \
    --stable-lift-reward-weight 16.0 \
    --force-closure-reward-weight 12.0 \
    --hard-gate-frontier-reward-weight 20.0 \
    --force-closure-shape-all-hold-contacts \
    --reset-optimizer-on-resume \
    --learning-rate 5e-5 \
    --freeze-policy-noise

sleep 20
if ! ps -ww -eo args= | rg -q \
  'migration_4090\.xhand_rl_embedded\.train.*--object cracker_large.*training_retry_014'; then
  echo "$(date --iso-8601=seconds) failed to verify cracker_large clearance retry" >> "$log"
  exit 3
fi
echo "$(date --iso-8601=seconds) launched cracker_large training_retry_014 from exact level-2 prefix" >> "$log"

screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" \
  -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
