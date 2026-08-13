#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
gpu=${1:-6}
export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo"

for density_effort in \
  170:0.15 170:0.25 170:0.35 170:0.45 170:0.55 \
  220:0.15 220:0.25 220:0.35 220:0.45 220:0.55 220:0.70 \
  300:0.15 300:0.25 300:0.35 300:0.45 300:0.55 300:0.70 \
  435:0.15 435:0.25 435:0.35 435:0.45 435:0.55 435:0.70
do
  density=${density_effort%%:*}
  effort=${density_effort##*:}
  effort_tag=${effort/./p}
  output="graph_exp/bimanual_data/baseline_power_drill_candidate28_density${density}_effort${effort_tag}_smoke_v1"
  log="migration_4090/logs/baseline_power_drill_candidate28_density${density}_effort${effort_tag}_smoke_v1.log"
  test ! -e "$output"
  test ! -e "$log"
  "$conda_root/bin/conda" run --no-capture-output -n tro \
    python generate_bimanual_pilot.py \
    --source-vis migration_4090/bulk_candidates/baseline_power_drill_candidate28_v1.pt \
    --objects ycb+power_drill \
    --pairs-per-object 1 \
    --roll-count 1 \
    --isaac-batch-size 1 \
    --realized-batch-size 1 \
    --left-robot-name allegro_left \
    --right-robot-name allegro_right \
    --gravity 9.8 \
    --gravity-settle-step 500 \
    --support-during-closure \
    --no-fixture-during-closure \
    --opposition-mode tabletop \
    --max-root-height-fraction 3 \
    --lateral-max-root-z-mm 100 \
    --lateral-max-height-diff-mm 45 \
    --independent-directions \
    --penetration-mm 2 \
    --contact-mm 4 \
    --min-contact-links 1 \
    --robot-friction 1 \
    --object-friction 1 \
    --finger-effort-limit "$effort" \
    --contact-offset 0.002 \
    --object-density "$density" \
    --max-gravity-displacement 0.01 \
    --max-direction-displacement 0.015 \
    --seed 20274824 \
    --gpu 0 \
    --output-dir "$output" \
    --force >"$log" 2>&1
  set +o errexit
  "$conda_root/envs/tro/bin/python" - "$output" "$density" "$effort" <<'PY'
import csv
import sys
from pathlib import Path

output, density, effort = sys.argv[1:]
row = next(csv.DictReader((Path(output) / "sample_results.csv").open()))
print(
    "scan",
    f"density={density}",
    f"effort={effort}",
    f"strict={row['strict_success']}",
    f"settle_mm={row['settle_displacement_mm']}",
    f"gravity_mm={row['gravity_displacement_mm']}",
    f"lift_mm={row['lift_displacement_mm']}",
    f"max6_mm={row['max_direction_displacement_mm']}",
    f"reasons={row['rejection_reasons']}",
    flush=True,
)
if row["strict_success"] == "True":
    raise SystemExit(100)
PY
  status=$?
  set -o errexit
  if [[ "$status" -eq 100 ]]; then
    echo "STRICT_SUCCESS output=$output density=$density effort=$effort"
    exit 0
  fi
done

echo "NO_STRICT_SUCCESS"
exit 1
