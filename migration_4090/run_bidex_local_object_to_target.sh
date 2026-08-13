#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 GPU OBJECT DENSITY INITIAL_SEED INITIAL_SCREEN TARGET CONTACT_MM [START_LOCAL_ROUND]" >&2
  exit 2
fi
gpu=$1
object=$2
density=$3
initial_seed=$4
initial_screen=$5
target=$6
contact_mm=$7
start_local_round=${8:-1}
repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
slug=${object//+/_}
root="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/$slug"
final="$repo/graph_exp/bimanual_data/bulk_irregular_bidex_v1/final_100_each/$slug"
candidate_root="$repo/migration_4090/bulk_candidates/bidex_v1/$slug"
audit_root="$repo/migration_4090/results/bulk_bidex_v1/$slug"
log="$repo/migration_4090/logs/bulk_target_bidex_local_${slug}_gpu${gpu}.log"

cd "$repo"
test ! -e "$log"
exec > >(tee "$log") 2>&1

verified_count() {
  "$conda_root/envs/tro/bin/python" migration_4090/count_bulk_verified.py \
    --input-root "$root"
}

strict_count() {
  "$conda_root/envs/tro/bin/python" - "$1" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(len(payload.get("samples", [])))
PY
}

verify_seed() {
  local seed=$1
  local dataset="$root/seed_$seed/bimanual_dataset.pt"
  local strict current remaining select_count
  if [[ -f "$root/seed_$seed/repeat_verified/verified_dataset.pt" ]]; then
    echo "seed=$seed already repeat-verified"
    return
  fi
  test -f "$dataset"
  strict=$(strict_count "$dataset")
  if [[ "$strict" -eq 0 ]]; then
    echo "seed=$seed has no strict sample; repeat verification skipped"
    return
  fi
  current=$(verified_count)
  remaining=$((target - current))
  if [[ "$remaining" -le 0 ]]; then return; fi
  select_count=$((remaining * 2 + 20))
  if [[ "$select_count" -gt 240 ]]; then select_count=240; fi
  if [[ "$select_count" -gt "$strict" ]]; then select_count=$strict; fi
  bash migration_4090/repeat_verify_bulk_seed.sh \
    "$gpu" bidex "$object" "$density" "$seed" "$select_count" 1 "$contact_mm"
}

if [[ -f "$final/bimanual_dataset.pt" ]]; then
  echo "COMPLETE already present: $final"
  exit 0
fi

echo "waiting for initial screen=$initial_screen object=$object"
while screen -ls 2>/dev/null | grep -q "[.]${initial_screen}[[:space:]]"; do
  sleep 30
done
test -f "$root/seed_$initial_seed/bimanual_dataset.pt"
verify_seed "$initial_seed"
count=$(verified_count)

# A local BiDex search needs one fully repeat-verified parent. If the initial
# region search produced none, use fresh global region seeds until one exists.
global_round=1
while [[ "$count" -eq 0 && "$global_round" -le 4 ]]; do
  seed=$((initial_seed + global_round * 1000))
  echo "no verified parent; global BiDex top-up round=$global_round seed=$seed"
  bash migration_4090/run_bulk_bidex_object.sh \
    "$gpu" "$object" "$density" "$seed"
  verify_seed "$seed"
  count=$(verified_count)
  global_round=$((global_round + 1))
done
if [[ "$count" -eq 0 ]]; then
  echo "FAILED: no repeat-verified BiDex parent after global searches" >&2
  exit 1
fi

round=$start_local_round
while [[ "$count" -lt "$target" && "$round" -le 4 ]]; do
  parent=$(find "$root" -path '*/repeat_verified/verified_dataset.pt' -type f | sort | head -n 1)
  test -n "$parent"
  seed=$((initial_seed + 900000 + round * 1000))
  candidate="$candidate_root/local_target_round${round}_seed${seed}.pt"
  audit="$audit_root/local_target_round${round}_seed${seed}_audit.json"
  output="$root/seed_$seed"
  echo "local round=$round seed=$seed verified=$count parent=$parent"
  test ! -e "$candidate"
  test ! -e "$audit"
  test ! -e "$output"
  mkdir -p "$candidate_root" "$audit_root"

  grid_args=()
  case "$round" in
    1)
      grid_args=(
        --yaw-degrees -6 -3 0 3 6
        --z-mm -4 -2 0 2 4
        --left-radial-mm -4 -2 0 2 4
        --right-radial-mm -4 -2 0 2 4
      )
      ;;
    2)
      grid_args=(
        --yaw-degrees -7.5 -4.5 -1.5 1.5 4.5 7.5
        --z-mm -5 -3 -1 1 3 5
        --left-radial-mm -5 -3 -1 1 3 5
        --right-radial-mm -5 -3 -1 1 3 5
      )
      ;;
    3)
      grid_args=(
        --yaw-degrees -10 -8 -2 2 8 10
        --z-mm -8 -6 0 6 8
        --left-radial-mm -8 -6 0 6 8
        --right-radial-mm -8 -6 0 6 8
      )
      ;;
    4)
      grid_args=(
        --yaw-degrees -12 -9 -6 6 9 12
        --z-mm -7.5 -5.5 -2.5 2.5 5.5 7.5
        --left-radial-mm -7.5 -5.5 -2.5 2.5 5.5 7.5
        --right-radial-mm -7.5 -5.5 -2.5 2.5 5.5 7.5
      )
      ;;
  esac
  "$conda_root/envs/tro/bin/python" \
    migration_4090/generate_bidex_local_neighborhood.py \
    --source-dataset "$parent" \
    --source-vis data/bimanual/source_vis.pt \
    --object "$object" \
    --sample-index 0 \
    --output "$candidate" \
    --audit-json "$audit" \
    "${grid_args[@]}"
  candidate_count=$("$conda_root/envs/tro/bin/python" - "$candidate" <<'PY'
import sys
import torch

entry = torch.load(sys.argv[1], map_location="cpu", weights_only=False)[0]
print(len(entry["precomputed_bimanual_candidates"]["left_q"]))
PY
  )

  export PATH="$conda_root/envs/isaac/bin:$PATH"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
  export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export OMP_NUM_THREADS=12
  export MKL_NUM_THREADS=12
  "$conda_root/bin/conda" run --no-capture-output -n tro \
    python generate_bimanual_pilot.py \
    --source-vis "$candidate" \
    --objects "$object" \
    --pairs-per-object "$candidate_count" \
    --roll-count 1 \
    --isaac-batch-size 16 \
    --realized-batch-size 4 \
    --left-robot-name allegro_left \
    --right-robot-name allegro_right \
    --gravity 9.8 \
    --gravity-settle-step 500 \
    --support-during-closure \
    --no-fixture-during-closure \
    --opposition-mode tabletop \
    --lateral-max-root-z-mm 100 \
    --lateral-max-height-diff-mm 45 \
    --independent-directions \
    --penetration-mm 2 \
    --contact-mm "$contact_mm" \
    --min-contact-links 1 \
    --robot-friction 1 \
    --object-friction 1 \
    --finger-effort-limit 0.7 \
    --contact-offset 0.002 \
    --object-density "$density" \
    --max-gravity-displacement 0.01 \
    --max-direction-displacement 0.015 \
    --seed "$seed" \
    --gpu 0 \
    --output-dir "$output" \
    --force
  verify_seed "$seed"
  count=$(verified_count)
  round=$((round + 1))
done

if [[ "$count" -lt "$target" ]]; then
  echo "FAILED: only $count unique repeat-verified samples after local rounds" >&2
  exit 1
fi
test ! -e "$final"
"$conda_root/envs/tro/bin/python" migration_4090/assemble_bulk_object.py \
  --input-root "$root" \
  --object-name "$object" \
  --method bidex \
  --target "$target" \
  --output-dir "$final"
echo "COMPLETE method=bidex object=$object count=$target final=$final"
