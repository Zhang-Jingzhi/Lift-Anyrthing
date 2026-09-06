#!/usr/bin/env bash
set -u

# Re-sample immutable formal checkpoints that emitted a hard terminal success
# during PPO training but whose original collector was interrupted before it
# materialized a candidate.  This is append-only and runs alongside the current
# six formal trainers.  The normal production bridge remains responsible for
# the locked 100-per-object target; this worker prioritizes the first fully
# strict success and never mutates an existing checkpoint or sample.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
collection_lock="/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"

declare -A checkpoint_sets=(
  [sphere_small]="$root/runs/sphere_small/training/model_3900.pt $root/runs/sphere_small/training/model_3999.pt"
  [cracker]="$root/runs/cracker/training/model_2800.pt $root/runs/cracker/training/model_2900.pt"
  [cube]="$root/runs/cube/training/model_2500.pt $root/runs/cube/training/model_2600.pt"
)
declare -A checkpoint_sha256_sets=(
  [sphere_small]="e44568d975786495d02d60079d5d69d60ab9d539f433beda036b9c7f3975e264 ef6c152cfaacec62d54d7cc54d141588cc766bded34251632760897412b0c6f5"
  [cracker]="92f0eaecc462a444d6b648ca233a3e2ec095c04d2a7627a814de7a309d928429 cebd38a461bccbad1f2d17a1be8ee7ac19fc444f9cf45191d4e9ffc78beb90fa"
  [cube]="00c327a75020a8080f9e7f34ad96229f391d28fcb29389dc79bcf2a175567d75 2295af931bc03dd9d934ca9e0da194e3d2b6fe9fec92563f515f4dd60efb2e62"
)
declare -A nominals=(
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/sphere_small.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cracker.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/cube.pt"
)
declare -A devices=(
  [sphere_small]="cuda:1"
  [cracker]="cuda:3"
  [cube]="cuda:4"
)

embedded_count() {
  local object="$1"
  find "$root/accepted/$object" -mindepth 2 -maxdepth 2 -type f \
    -name receipt.json 2>/dev/null | wc -l
}

strict_pass() {
  local object="$1"
  "$isaac_python" - "$root/accepted/$object" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
for receipt in base.glob("sample_*/strict_post_audit/receipt.json"):
    try:
        if json.loads(receipt.read_text()).get("accepted") is True:
            raise SystemExit(0)
    except (OSError, TypeError, ValueError):
        pass
raise SystemExit(1)
PY
}

verify_inputs() {
  local object checkpoint actual expected nominal recorded_nominal index
  local -a object_checkpoints object_hashes
  for object in sphere_small cracker cube; do
    nominal="${nominals[$object]}"
    read -r -a object_checkpoints <<< "${checkpoint_sets[$object]}"
    read -r -a object_hashes <<< "${checkpoint_sha256_sets[$object]}"
    [[ ${#object_checkpoints[@]} -eq ${#object_hashes[@]} && -f "$nominal" ]] || {
      echo "missing immutable input for $object" >&2
      return 1
    }
    for index in "${!object_checkpoints[@]}"; do
      checkpoint="${object_checkpoints[$index]}"
      expected="${object_hashes[$index]}"
      [[ -f "$checkpoint" ]] || return 1
      actual="$(sha256sum "$checkpoint" | awk '{print $1}')"
      [[ "$actual" == "$expected" ]] || {
        echo "checkpoint hash mismatch for $object: $checkpoint" >&2
        return 1
      }
      env PYTHONPATH="$repo" "$isaac_python" -c \
        'import sys; from pathlib import Path; from migration_4090.xhand_rl_embedded.checkpoint_selection import validate_collection_checkpoint; validate_collection_checkpoint(Path(sys.argv[1]), allow_adjacent_hard_success=True)' \
        "$checkpoint" || return 1
    done
    recorded_nominal="$("$isaac_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["nominal_dataset"])' "$(dirname "$checkpoint")/training_provenance.json")"
    [[ "$recorded_nominal" == "$nominal" ]] || {
      echo "nominal provenance mismatch for $object" >&2
      return 1
    }
  done
}

wait_for_audit() {
  local object="$1" sample_index="$2" receipt
  printf -v receipt '%s/accepted/%s/sample_%06d/strict_post_audit/receipt.json' \
    "$root" "$object" "$sample_index"
  while [[ ! -f "$receipt" ]]; do
    sleep 30
  done
}

verify_inputs || exit 1
mkdir -p "$root/logs/hard_success_collectors"

seed=1001
while :; do
  all_strict=1
  for object in sphere_small cracker cube; do
    if strict_pass "$object"; then
      continue
    fi
    all_strict=0
    read -r -a object_checkpoints <<< "${checkpoint_sets[$object]}"
    checkpoint_index=$(((seed - 1001) % ${#object_checkpoints[@]}))
    checkpoint="${object_checkpoints[$checkpoint_index]}"
    before="$(embedded_count "$object")"
    target=$((before + 1))
    checkpoint_tag="$(basename "$checkpoint" .pt)"
    attempt_id="historical-hard-success-${object}-${checkpoint_tag}-seed-${seed}-target-${target}"
    log="$root/logs/hard_success_collectors/${attempt_id}.log"
    echo "$(date --iso-8601=seconds) start object=$object seed=$seed target=$target checkpoint=$checkpoint" \
      | tee -a "$root/logs/hard_success_collectors/supervisor.log"
    env PYTHONPATH="$repo" PYTHONUNBUFFERED=1 "$isaac_python" \
      -m migration_4090.xhand_rl_embedded.collect \
      --manifest "$manifest" --object "$object" \
      --nominal-dataset "${nominals[$object]}" \
      --checkpoint "$checkpoint" --allow-adjacent-hard-success-checkpoint \
      --root "$root" --target "$target" --num-envs 128 --max-steps 5000 \
      --seed "$seed" --device "${devices[$object]}" --headless --allow-partial \
      --attempt-id "$attempt_id" --collection-lock "$collection_lock" \
      --kit-teardown-cooldown-s 75 \
      --kit-portable-root "/tmp/xhand_rl_embedded_kit/v5r_hard_success/$object/$attempt_id" \
      --penetration-reward-weight -20.0 \
      --instantaneous-residual-weight 0.0 --residual-integration 0.03 \
      --arm-action-scale-rad 0.12 --hand-action-scale-rad 0.24 \
      --residual-activation-phase 2 \
      >>"$log" 2>&1
    code=$?
    after="$(embedded_count "$object")"
    echo "$(date --iso-8601=seconds) finish object=$object seed=$seed exit=$code before=$before after=$after" \
      | tee -a "$root/logs/hard_success_collectors/supervisor.log"
    if [[ $code -eq 134 || $code -eq 139 ]]; then
      # An allocator/segfault exit bypasses Python's finally block.  Reacquire
      # the same global lock and hold a longer backoff before any new Kit start.
      flock -x "$collection_lock" -c "sleep 180"
    fi
    if (( after > before )); then
      # Give the strict watchdog exclusive use of the shared Kit lock and wait
      # for its complete replay/mesh decision before producing another sample.
      wait_for_audit "$object" "$after"
    fi
    seed=$((seed + 1))
  done
  (( all_strict == 1 )) && exit 0
done
