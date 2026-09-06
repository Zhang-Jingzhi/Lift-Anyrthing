#!/usr/bin/env bash
set -euo pipefail
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
source="$root/runs/cracker_large/training_retry_015"
seed="$source/hard_prefix_preupdate_iter_1826_level_3.pt"
output="$root/runs/cracker_large/training_retry_017"
log="$root/logs/cracker_large/shaped_v12_progressive_disturbance_switch.log"
for path in "$python" "$manifest" "$nominal" "$seed"; do [[ -f "$path" ]] || exit 2; done
[[ ! -e "$output" ]] || exit 2
PYTHONPATH="$repo" "$python" - "$source" "$seed" <<'PY'
import json, sys
from pathlib import Path
from migration_4090.xhand_rl_embedded.contracts import resume_checkpoint_iteration, sha256_file
source, seed = map(lambda x: Path(x).resolve(), sys.argv[1:])
rows = [json.loads(x) for x in (source / "hard_prefix_rollouts.jsonl").read_text().splitlines() if x]
assert any(Path(r["checkpoint"]).resolve() == seed and r["checkpoint_sha256"] == sha256_file(seed) and r.get("saved_before_ppo_update") is True and int(r["max_level"]) >= 3 for r in rows)
assert resume_checkpoint_iteration(seed) == 1826
PY
screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v11_prefix_cracker_large -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped retry_016; cooldown_s=90" >> "$log"
sleep 90
if ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_(embedded\.train|curriculum\.train|embedded\.bridge_worker).*--object cracker_large'; then exit 3; fi
screen -L -Logfile "$root/logs/cracker_large/shaped_v12_progressive_disturbance_training_screen.log" \
  -dmS xhand_formal_v12_progress_cracker_large env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.train \
  --manifest "$manifest" --object cracker_large --nominal-dataset "$nominal" --output "$output" \
  --num-envs 128 --max-iterations 5826 --seed 4086 --device cuda:2 --headless \
  --kit-portable-root /tmp/xhand_rl_embedded_kit/v12_progress/cracker_large \
  --penetration-reward-weight -30 --terminal-success-weight 2500 --gamma 0.999 \
  --init-noise-std 0.03 --entropy-coef 0.0001 --resume-checkpoint "$seed" \
  --residual-activation-phase 0 --instantaneous-residual-weight 0 --residual-integration 0.006 \
  --residual-limit-rad 0.15 --arm-action-scale-rad 0.06 --hand-action-scale-rad 0.08 \
  --stable-lift-reward-weight 24 --force-closure-reward-weight 36 \
  --hard-gate-frontier-reward-weight 40 --force-closure-shape-all-hold-contacts \
  --reset-optimizer-on-resume --learning-rate 1e-5 --freeze-policy-noise
sleep 20
ps -ww -eo args= | rg -q 'xhand_rl_embedded\.train.*--object cracker_large.*training_retry_017' || exit 4
echo "$(date --iso-8601=seconds) launched retry_017 with progressive disturbance shaping" >> "$log"
screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" -dmS xhand_formal_embedded_reconcile_target100 bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
