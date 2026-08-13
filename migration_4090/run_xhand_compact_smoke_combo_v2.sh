#!/usr/bin/env bash
set -u -o pipefail

GPU="${1:?gpu}"
OBJECT="${2:?object}"
METHOD="${3:?method}"
SIZE_INDEX="${4:-4}"
SEED="${5:-2026081201}"
CANDIDATE_COUNT="${6:-12}"

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
ROOT="${XHAND_COMPACT_SMOKE_ROOT:-$REPO/migration_4090/results/xhand_compact_6x100_v2_smoke}"
DATASET="$ROOT/candidates/$OBJECT/${METHOD}_size$(printf '%03d' "$SIZE_INDEX")_seed${SEED}.pt"
PHYS="$ROOT/physics/$OBJECT/$METHOD/seed${SEED}"
SUMMARY="$PHYS/smoke_success.json"

cd "$REPO"
mkdir -p "$(dirname "$DATASET")" "$PHYS"

export XHAND_FULLBODY_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf
export XHAND_SUPPORT_SIZE_X=0.60 XHAND_SUPPORT_SIZE_Y=0.80
export XHAND_SUPPORT_X=0.50 XHAND_SUPPORT_Y=0.0 XHAND_SUPPORT_Z=0.70
export XHAND_ROBOT_COLLISION_FILTER=1 JAX_PLATFORMS=cpu

if [[ -s "$SUMMARY" ]]; then
  echo "ALREADY_COMPLETE $SUMMARY"
  exit 0
fi

if [[ ! -s "$DATASET" ]]; then
  if ! "$TRO" migration_4090/generate_xhand_compact_candidate_bank_v2.py \
    --object "$OBJECT" --size-index "$SIZE_INDEX" --method "$METHOD" \
    --count "$CANDIDATE_COUNT" --seed "$SEED" --output "$DATASET"; then
    echo "CANDIDATE_GENERATION_ERROR object=$OBJECT method=$METHOD seed=$SEED"
    exit 3
  fi
fi

if [[ ! -s "$DATASET" ]]; then
  echo "CANDIDATE_DATASET_MISSING $DATASET"
  exit 3
fi

ACTUAL_COUNT=$(
  "$TRO" -c 'import sys,torch; print(len(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["samples"]))' "$DATASET"
)

DENSITY=$(
  "$TRO" -c 'import sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(d["samples"][0]["physical_parameters"]["recommended_density_kg_m3"])' "$DATASET"
)
echo "CONFIG gpu=$GPU object=$OBJECT method=$METHOD size=$SIZE_INDEX candidates=$ACTUAL_COUNT density=$DENSITY dataset=$DATASET"
LEFT_SQUEEZE="${XHAND_LEFT_SQUEEZE:-0.75}"
RIGHT_SQUEEZE=0.75
if [[ "$OBJECT" != "sphere" ]]; then
  RIGHT_SQUEEZE=0.82
fi
RIGHT_SQUEEZE="${XHAND_RIGHT_SQUEEZE:-$RIGHT_SQUEEZE}"

json_pass() {
  "$TRO" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["physical_pass"] else 1)' "$1"
}

probe() {
  local index=$1
  local output=$2
  local modes=$3
  mkdir -p "$(dirname "$output")"
  bash migration_4090/run_external_xhand_strict_probe.sh \
    "$GPU" "$DATASET" "$index" "$output" "$modes" \
    1.0 500 500 100 position "$DENSITY" 2.0 \
    "" "" "" 600 300 100 50 2 300 0.5 500 0.75 "$LEFT_SQUEEZE" "$RIGHT_SQUEEZE"
}

for ((INDEX=0; INDEX<ACTUAL_COUNT; INDEX++)); do
  RUN="$PHYS/candidate_$(printf '%03d' "$INDEX")"
  BOTH="$RUN/both.json"
  if [[ ! -s "$BOTH" ]]; then
    echo "BOTH_START object=$OBJECT method=$METHOD index=$INDEX"
    if ! probe "$INDEX" "$BOTH" both; then
      echo "BOTH_ERROR object=$OBJECT method=$METHOD index=$INDEX"
      continue
    fi
  fi
  if ! json_pass "$BOTH"; then
    echo "BOTH_REJECT object=$OBJECT method=$METHOD index=$INDEX"
    continue
  fi

  all_pass=1
  for REPEAT in 1 2 3; do
    REPORT="$RUN/strict_repeat${REPEAT}.json"
    if [[ ! -s "$REPORT" ]]; then
      echo "STRICT_START object=$OBJECT method=$METHOD index=$INDEX repeat=$REPEAT"
      if ! probe "$INDEX" "$REPORT" both,left,right; then
        all_pass=0
        break
      fi
    fi
    if ! json_pass "$REPORT"; then
      echo "STRICT_REJECT object=$OBJECT method=$METHOD index=$INDEX repeat=$REPEAT"
      all_pass=0
      break
    fi
  done
  [[ "$all_pass" -eq 1 ]] || continue

  "$TRO" - "$DATASET" "$INDEX" "$DENSITY" "$SUMMARY" "$RUN" <<'PY'
import json, sys
from pathlib import Path
dataset, index, density, output, run = sys.argv[1:]
reports = [Path(run) / f"strict_repeat{i}.json" for i in (1, 2, 3)]
payloads = [json.loads(path.read_text()) for path in reports]
both = payloads[0]["runs"]["both"]
summary = {
    "schema": "xhand_compact_smoke_success_v2",
    "dataset": dataset,
    "sample_index": int(index),
    "density_kg_m3": float(density),
    "repeat_reports": [str(path) for path in reports],
    "repeat_success_rate": sum(bool(p["physical_pass"]) for p in payloads) / len(payloads),
    "both_physical_pass": bool(payloads[0]["runs"]["both"]["physical_pass"]),
    "left_only_physical_pass": bool(payloads[0]["runs"]["left"]["physical_pass"]),
    "right_only_physical_pass": bool(payloads[0]["runs"]["right"]["physical_pass"]),
    "lift_height_mm": both["lift_object_displacement_m"] * 1000.0,
    "gravity_displacement_mm": both["gravity_displacement_m"] * 1000.0,
    "six_direction_max_mm": max(both["six_direction_displacements_m"]) * 1000.0,
    "left_closure_contact_count": both["closure"]["left_contact_count"],
    "right_closure_contact_count": both["closure"]["right_contact_count"],
    "penetration_pass": bool(both["penetration_pass"]),
}
Path(output).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
  echo "SMOKE_SUCCESS object=$OBJECT method=$METHOD index=$INDEX summary=$SUMMARY"
  exit 0
done

echo "SMOKE_EXHAUSTED object=$OBJECT method=$METHOD candidates=$ACTUAL_COUNT"
exit 2
