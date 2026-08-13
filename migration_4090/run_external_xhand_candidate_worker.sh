#!/usr/bin/env bash
set -u -o pipefail

GPU="${1:?gpu}"
SHARD="${2:?shard}"
SHARDS="${3:-8}"
ROOT=migration_4090/results/xhand_external_candidates_v1
OUT=migration_4090/results/xhand_external_strict_seed_search_v1/candidates
TRO=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python

cd /media/home/zhangjingzhi/TRO-Grasp-Reproduction

# Generation is CPU/JAX and runs in separate screens.  Do not begin Isaac
# screening until every object has both method datasets and one canonical mesh.
while :; do
  ready=0
  for object in cracker piggy drill toy pitcher bleach; do
    [[ -s "$ROOT/$object/summary.json" ]] && ready=$((ready + 1))
  done
  [[ "$ready" -eq 6 ]] && break
  echo "WAIT generation ready=$ready/6"
  sleep 20
done

global=0
for object in cracker piggy drill toy pitcher bleach; do
  for dataset in "$ROOT/$object"/baseline__*.pt "$ROOT/$object"/bidex_v3__*.pt; do
    [[ -s "$dataset" ]] || continue
    method=$(basename "$dataset")
    method=${method%%__*}
    count=$("$TRO" -c 'import sys,torch; print(len(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["samples"]))' "$dataset")
    for ((index=0; index<count; index++)); do
      assigned=$((global % SHARDS))
      global=$((global + 1))
      [[ "$assigned" -eq "$SHARD" ]] || continue

      run_dir="$OUT/$object/$method/candidate_$(printf '%03d' "$index")"
      both="$run_dir/both.json"
      strict="$run_dir/strict.json"
      mkdir -p "$run_dir"

      if [[ ! -s "$both" ]]; then
        echo "BOTH gpu=$GPU object=$object method=$method index=$index"
        if ! bash migration_4090/run_external_xhand_strict_probe.sh \
          "$GPU" "$dataset" "$index" "$both" both 1.0; then
          echo "ERROR both object=$object method=$method index=$index"
          continue
        fi
      fi
      if ! "$TRO" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["physical_pass"] else 1)' "$both"; then
        echo "REJECT both object=$object method=$method index=$index"
        continue
      fi

      if [[ ! -s "$strict" ]]; then
        echo "STRICT gpu=$GPU object=$object method=$method index=$index"
        if ! bash migration_4090/run_external_xhand_strict_probe.sh \
          "$GPU" "$dataset" "$index" "$strict" both,left,right 1.0; then
          echo "ERROR strict object=$object method=$method index=$index"
          continue
        fi
      fi
      if ! "$TRO" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["physical_pass"] else 1)' "$strict"; then
        echo "REJECT ablation object=$object method=$method index=$index"
        continue
      fi

      successes=1
      for repeat in 2 3; do
        report="$run_dir/repeat${repeat}.json"
        if [[ ! -s "$report" ]]; then
          echo "REPEAT repeat=$repeat gpu=$GPU object=$object method=$method index=$index"
          if ! bash migration_4090/run_external_xhand_strict_probe.sh \
            "$GPU" "$dataset" "$index" "$report" both,left,right 1.0; then
            echo "ERROR repeat=$repeat object=$object method=$method index=$index"
            continue
          fi
        fi
        if "$TRO" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["physical_pass"] else 1)' "$report"; then
          successes=$((successes + 1))
        fi
      done
      echo "VERIFIED object=$object method=$method index=$index rollouts=$successes/3"
    done
  done
done
echo "WORKER_DONE gpu=$GPU shard=$SHARD/$SHARDS"
