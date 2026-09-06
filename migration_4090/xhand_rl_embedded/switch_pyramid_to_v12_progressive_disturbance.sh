#!/usr/bin/env bash
set -euo pipefail
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
source="$root/runs/pyramid/training_retry_022"
seed="$source/hard_prefix_preupdate_iter_744_level_4.pt"
output="$root/runs/pyramid/training_retry_024"
log="$root/logs/pyramid/shaped_v12_progressive_disturbance_switch.log"
for path in "$python" "$manifest" "$nominal" "$seed"; do [[ -f "$path" ]] || exit 2; done
[[ ! -e "$output" ]] || exit 2
PYTHONPATH="$repo" "$python" - "$source" "$seed" <<'PY'
import json, sys
from pathlib import Path
from migration_4090.xhand_rl_embedded.contracts import resume_checkpoint_iteration, sha256_file
source, seed = map(lambda x: Path(x).resolve(), sys.argv[1:])
rows = [json.loads(x) for x in (source / "hard_prefix_rollouts.jsonl").read_text().splitlines() if x]
assert any(Path(r["checkpoint"]).resolve() == seed and r["checkpoint_sha256"] == sha256_file(seed) and r.get("saved_before_ppo_update") is True and int(r["max_level"]) >= 4 for r in rows)
assert resume_checkpoint_iteration(seed) == 744
PY
screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v11_prefix_pyramid -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped retry_023; cooldown_s=90" >> "$log"
sleep 90
if ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_(embedded\.train|curriculum\.train|embedded\.bridge_worker).*--object pyramid'; then exit 3; fi
screen -L -Logfile "$root/logs/pyramid/shaped_v12_progressive_disturbance_training_screen.log" \
  -dmS xhand_formal_v12_progress_pyramid env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 \
  "$python" -m migration_4090.xhand_rl_embedded.train \
  --manifest "$manifest" --object pyramid --nominal-dataset "$nominal" --output "$output" \
  --num-envs 128 --max-iterations 4744 --seed 4088 --device cuda:5 --headless \
  --kit-portable-root /tmp/xhand_rl_embedded_kit/v12_progress/pyramid \
  --penetration-reward-weight -30 --terminal-success-weight 2500 --gamma 0.999 \
  --init-noise-std 0.03 --entropy-coef 0.0001 --resume-checkpoint "$seed" \
  --residual-activation-phase 0 --instantaneous-residual-weight 0 --residual-integration 0.006 \
  --residual-limit-rad 0.15 --arm-action-scale-rad 0.10 --hand-action-scale-rad 0.12 \
  --stable-lift-reward-weight 24 --force-closure-reward-weight 24 \
  --hard-gate-frontier-reward-weight 40 --force-closure-shape-all-hold-contacts \
  --reset-optimizer-on-resume --learning-rate 1e-5 --freeze-policy-noise
sleep 20
ps -ww -eo args= | rg -q 'xhand_rl_embedded\.train.*--object pyramid.*training_retry_024' || exit 4
echo "$(date --iso-8601=seconds) launched retry_024 with progressive disturbance shaping" >> "$log"
screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" -dmS xhand_formal_embedded_reconcile_target100 bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
