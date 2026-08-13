#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu}"
OBJECT="${2:?object}"
METHOD="${3:?method}"
BASE_SEED="${4:?base seed}"

REPO=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
ROOT="$REPO/graph_exp/bimanual_data/xhand_compact_formal_1200_v1/$METHOD/$OBJECT"
ACCEPTED="$ROOT/accepted"
FINAL="$ROOT/final_100/bimanual_dataset.pt"
BANK_COUNT="${XHAND_FORMAL_BANK_COUNT:-24}"
TARGET_PER_SIZE=20

case "$OBJECT" in
  sphere)  SIZE_POOL=(1 2 3 4 5) ;;
  cube)    SIZE_POOL=(1 2 3 4 5) ;;
  cracker) SIZE_POOL=(0 1 2 3 4) ;;
  bleach)  SIZE_POOL=(0 1 2 3 4) ;;
  pitcher) SIZE_POOL=(1 2 3 4 5) ;;
  drill)   SIZE_POOL=(0 1 2 3 4) ;;
  *) echo "unknown object: $OBJECT" >&2; exit 2 ;;
esac
case "$METHOD" in baseline|bidex_v3) ;; *) echo "unknown method: $METHOD" >&2; exit 2 ;; esac

cd "$REPO"
mkdir -p "$ACCEPTED" "$ROOT/batches"

export XHAND_FULLBODY_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf
export XHAND_FULLBODY_IK_URDF=/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf
export XHAND_SUPPORT_SIZE_X=0.60 XHAND_SUPPORT_SIZE_Y=0.80
export XHAND_SUPPORT_X=0.50 XHAND_SUPPORT_Y=0.0 XHAND_SUPPORT_Z=0.70
export XHAND_ROBOT_COLLISION_FILTER=1 JAX_PLATFORMS=cpu
export XHAND_FORMAL_CLEARANCE_EXTRA_M="${XHAND_FORMAL_CLEARANCE_EXTRA_M:-0.025}"
export XHAND_FORMAL_HAND_DELTA_LIMIT="${XHAND_FORMAL_HAND_DELTA_LIMIT:-0.015}"

if [[ -s "$FINAL" ]]; then
  echo "FORMAL_ALREADY_COMPLETE object=$OBJECT method=$METHOD final=$FINAL"
  exit 0
fi

json_pass() {
  "$TRO" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("physical_pass") is True else 1)' "$1"
}

accepted_for_size() {
  "$TRO" - "$ACCEPTED" "$1" <<'PY'
import json, sys
from pathlib import Path
root, size_index = Path(sys.argv[1]), int(sys.argv[2])
print(sum(json.loads(p.read_text()).get("size_index") == size_index for p in root.glob("sample_*.json")))
PY
}

total_accepted() {
  find "$ACCEPTED" -maxdepth 1 -type f -name 'sample_*.json' | wc -l
}

probe() {
  local dataset=$1 index=$2 output=$3 modes=$4 density=$5
  mkdir -p "$(dirname "$output")"
  local right_squeeze=0.82
  [[ "$OBJECT" == sphere ]] && right_squeeze=0.75
  bash migration_4090/run_external_xhand_strict_probe.sh \
    "$GPU" "$dataset" "$index" "$output" "$modes" \
    1.0 500 500 100 position "$density" 2.0 \
    "" "" "" 600 300 100 50 2 300 0.5 500 0.75 0.75 "$right_squeeze"
}

for SIZE_INDEX in "${SIZE_POOL[@]}"; do
  BATCH=0
  while (( $(accepted_for_size "$SIZE_INDEX") < TARGET_PER_SIZE )); do
    SEED=$((BASE_SEED + SIZE_INDEX * 100000 + BATCH * 1000))
    BATCH_ROOT="$ROOT/batches/size_$(printf '%03d' "$SIZE_INDEX")/seed${SEED}"
    DATASET="$BATCH_ROOT/candidates.pt"
    mkdir -p "$BATCH_ROOT"
    if [[ ! -s "$DATASET" ]]; then
      echo "BANK_START gpu=$GPU object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH seed=$SEED"
      if ! "$TRO" migration_4090/generate_xhand_compact_candidate_bank_v2.py \
        --object "$OBJECT" --size-index "$SIZE_INDEX" --method "$METHOD" \
        --count "$BANK_COUNT" --seed "$SEED" --disable-anchor --output "$DATASET"; then
        echo "BANK_ERROR object=$OBJECT method=$METHOD size=$SIZE_INDEX seed=$SEED"
        BATCH=$((BATCH + 1))
        continue
      fi
    fi
    ACTUAL_COUNT=$("$TRO" -c 'import sys,torch; print(len(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["samples"]))' "$DATASET")
    DENSITY=$("$TRO" -c 'import sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(d["samples"][0]["physical_parameters"]["recommended_density_kg_m3"])' "$DATASET")
    echo "BANK_READY gpu=$GPU object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH candidates=$ACTUAL_COUNT density=$DENSITY accepted_size=$(accepted_for_size "$SIZE_INDEX") accepted_total=$(total_accepted)"

    for ((INDEX=0; INDEX<ACTUAL_COUNT; INDEX++)); do
      (( $(accepted_for_size "$SIZE_INDEX") < TARGET_PER_SIZE )) || break
      RUN="$BATCH_ROOT/candidate_$(printf '%03d' "$INDEX")"
      [[ -e "$RUN/accepted.done" || -e "$RUN/mesh_rejected.done" ]] && continue
      BOTH="$RUN/both.json"
      if [[ ! -s "$BOTH" ]]; then
        echo "BOTH_START gpu=$GPU object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX"
        if ! probe "$DATASET" "$INDEX" "$BOTH" both "$DENSITY"; then
          echo "BOTH_ERROR object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX"
          continue
        fi
      fi
      if ! json_pass "$BOTH"; then
        echo "BOTH_REJECT object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX"
        continue
      fi

      ALL_PASS=1
      REPORTS=()
      for REPEAT in 1 2 3; do
        REPORT="$RUN/strict_repeat${REPEAT}.json"
        REPORTS+=("$REPORT")
        if [[ ! -s "$REPORT" ]]; then
          echo "STRICT_START gpu=$GPU object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX repeat=$REPEAT"
          if ! probe "$DATASET" "$INDEX" "$REPORT" both,left,right "$DENSITY"; then
            ALL_PASS=0
            break
          fi
        fi
        if ! json_pass "$REPORT"; then
          echo "STRICT_REJECT object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX repeat=$REPEAT"
          ALL_PASS=0
          break
        fi
      done
      (( ALL_PASS == 1 )) || continue

      if "$TRO" migration_4090/save_xhand_compact_formal_success.py \
        --dataset "$DATASET" --sample-index "$INDEX" --object "$OBJECT" \
        --method "$METHOD" --size-index "$SIZE_INDEX" \
        --reports "${REPORTS[@]}" --accepted-root "$ACCEPTED" \
        --rejection-output "$RUN/visual_mesh_rejection.json"; then
        : > "$RUN/accepted.done"
        echo "FORMAL_ACCEPT object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX accepted_size=$(accepted_for_size "$SIZE_INDEX") accepted_total=$(total_accepted)"
      else
        STATUS=$?
        if (( STATUS == 4 )); then
          : > "$RUN/mesh_rejected.done"
          echo "VISUAL_MESH_REJECT object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX"
        else
          echo "SAVE_ERROR object=$OBJECT method=$METHOD size=$SIZE_INDEX batch=$BATCH index=$INDEX status=$STATUS"
        fi
      fi
    done
    BATCH=$((BATCH + 1))
  done
done

"$TRO" migration_4090/assemble_xhand_compact_formal_combo.py \
  --accepted-root "$ACCEPTED" --object "$OBJECT" --method "$METHOD" \
  --target 100 --output "$FINAL"
echo "FORMAL_COMPLETE object=$OBJECT method=$METHOD accepted=$(total_accepted) final=$FINAL"
