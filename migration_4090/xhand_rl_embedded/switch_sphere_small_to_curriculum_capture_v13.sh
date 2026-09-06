#!/usr/bin/env bash
set -euo pipefail

repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"
source="$root/runs/sphere_small/curriculum_retry_004/model_300.pt"
output="$root/runs/sphere_small/curriculum_retry_005"
seed_copy="$output/model_300.pt"
log="$root/logs/sphere_small/shaped_v13_curriculum_capture_switch.log"

for path in "$python" "$manifest" "$nominal" "$source"; do
  [[ -f "$path" ]] || { echo "$(date --iso-8601=seconds) missing required file: $path" >> "$log"; exit 2; }
done
[[ ! -e "$output" ]] || { echo "$(date --iso-8601=seconds) refusing existing output: $output" >> "$log"; exit 2; }
mkdir -p "$output"
cp --reflink=auto "$source" "$seed_copy"
PYTHONPATH="$repo" "$python" - "$source" "$seed_copy" <<'PY'
import sys
from pathlib import Path
from migration_4090.xhand_rl_embedded.contracts import sha256_file
source, copied = map(Path, sys.argv[1:])
if sha256_file(source) != sha256_file(copied):
    raise SystemExit("seed copy hash mismatch")
PY
cat > "$output/seed_source.json" <<EOF
{"source":"$source","source_sha256":"$(sha256sum "$source" | awk '{print $1}')","copied":"$seed_copy"}
EOF

screen -S xhand_formal_embedded_reconcile_target100 -X quit >/dev/null 2>&1 || true
screen -S xhand_formal_v12_phase0_sphere_small -X quit >/dev/null 2>&1 || true
echo "$(date --iso-8601=seconds) stopped sphere_small formal retry_021; cooldown_s=90" >> "$log"
sleep 90
if ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_(embedded\.bridge_worker|embedded\.train|curriculum\.train).*--object sphere_small'; then
  echo "$(date --iso-8601=seconds) old sphere_small Isaac process still alive after cooldown" >> "$log"; exit 3
fi

screen -L -Logfile "$root/logs/sphere_small/shaped_v13_curriculum_capture_training_screen.log" \
  -dmS xhand_formal_v13_curriculum_sphere_small \
  env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 "$python" -m migration_4090.xhand_rl_curriculum.train \
    --manifest "$manifest" --object sphere_small --nominal-dataset "$nominal" \
    --output "$output" --num-envs 128 --max-iterations 400 --iteration-chunk 10 \
    --seed 3985 --device cuda:1 --headless \
    --kit-portable-root /tmp/xhand_rl_embedded_kit/v13_curriculum/sphere_small \
    --init-noise-std 0.03 --entropy-coef 0.001 --freeze-policy-noise \
    --stop-on-first-curriculum-success --residual-activation-phase 1 \
    --resume-checkpoint "$seed_copy"

sleep 20
if ! ps -ww -eo args= | rg -q 'migration_4090\.xhand_rl_curriculum\.train.*--object sphere_small.*curriculum_retry_005'; then
  echo "$(date --iso-8601=seconds) failed to verify sphere_small curriculum_retry_005" >> "$log"; exit 4
fi
echo "$(date --iso-8601=seconds) launched sphere_small curriculum_retry_005 with pre-update success capture" >> "$log"
screen -L -Logfile "$root/logs/reconcile_target100_embedded_only_screen.log" -dmS xhand_formal_embedded_reconcile_target100 \
  bash "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
echo "$(date --iso-8601=seconds) embedded-only reconcile restored" >> "$log"
