#!/usr/bin/env bash
set -u

# Post-audit embedded RL samples with the full strict replay contract.  The
# embedded collector remains fast and append-only; this watcher performs the
# expensive 3x Isaac replay, left/right ablations, force closure, original
# visual-mesh audit and video only after a candidate exists.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
strict_manifest="$repo/migration_4090/config/xhand_locked_six_rl_v1.json"
collection_lock="/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"
strict_lock="/tmp/xhand_rl_embedded_kit/global_strict_post_audit.lock"
declare -A devices=(
  [sphere]=cuda:0
  [sphere_small]=cuda:1
  [cracker_large]=cuda:2
  [cracker]=cuda:3
  [pyramid]=cuda:5
  [cube]=cuda:4
)

run_one() {
  local object="$1" receipt="$2" sample_root candidate output_dir launch_log
  sample_root="$(dirname "$receipt")"
  candidate="$(PYTHONPATH="$repo" "$isaac_python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["sample_pt"])' "$receipt" 2>/dev/null)"
  [[ -f "$candidate" ]] || return 0
  output_dir="$sample_root/strict_post_audit"
  if [[ -f "$output_dir/receipt.json" ]]; then
    # Return a distinct status so the caller can continue past an already
    # audited rejection and inspect the next append-only candidate.  Returning
    # success here used to make the outer loop break forever on sample_000001.
    return 2
  fi
  if [[ -e "$output_dir" ]]; then
    mv "$output_dir" "${output_dir}_incomplete_$(date +%s)"
  fi
  # strict_validate owns output-directory creation and refuses to run if the
  # target already exists.  Keep the launcher log beside the directory using
  # a unique append-only name; all replay logs remain inside output_dir after
  # the validator creates it.
  launch_log="$sample_root/strict_post_audit_launch_$(date +%s).log"
  echo "strict post-audit: $object candidate=$candidate"
  # Serialize all strict Isaac/Kit startup paths with the collector lock and a
  # separate audit lock.  The lock is held for the complete 9 replay runs and
  # renderer pass, not just process startup.
  (
    flock -x 8
    flock -x 9
    PYTHONPATH="$repo" "$isaac_python" -m migration_4090.xhand_rl.strict_validate \
      --manifest "$strict_manifest" --candidate "$candidate" --sample-index 0 \
      --object "$object" --output-dir "$output_dir" \
      --accepted-root "$root/accepted" --isaac-python "$isaac_python" \
      --device "${devices[$object]}" --embedded-v2 \
      >"$launch_log" 2>&1
    strict_code=$?
    # Hold the shared Kit lock while asynchronous PhysX/Vulkan teardown
    # settles; immediate collector startup has produced allocator corruption.
    sleep 75
    exit "$strict_code"
  ) 8>"$collection_lock" 9>"$strict_lock"
  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "strict post-audit PASS: $object $candidate"
  else
    echo "strict post-audit rejected/failed: $object exit=$code"
  fi
  return 0
}

while :; do
  for object in sphere sphere_small cracker_large cracker pyramid cube; do
    # Once one strict candidate passes, leave later embedded samples available
    # for the 100-sample production target but avoid needless expensive audits.
    strict_pass=$(find "$root/accepted/$object" -mindepth 3 -maxdepth 3 \
      -path '*/strict_post_audit/receipt.json' -type f -print0 2>/dev/null \
      | xargs -0 -r "$isaac_python" -c \
      'import json,sys; print(any(json.load(open(p)).get("accepted") is True for p in sys.argv[1:]))' 2>/dev/null || echo false)
    [[ "$strict_pass" == "True" ]] && continue
    while IFS= read -r -d '' receipt; do
      run_one "$object" "$receipt"
      run_code=$?
      if [[ $run_code -eq 2 ]]; then
        continue
      fi
      # Re-evaluate after each candidate; stop this object once a strict pass
      # is present, then let the outer loop move to the next object.
      break
    done < <(find "$root/accepted/$object" -mindepth 2 -maxdepth 2 \
      -type f -name receipt.json -print0 2>/dev/null | sort -z)
  done
  sleep 30
done
