#!/usr/bin/env bash
set -euo pipefail

# Append-only migration after the v7 100-update audit showed that individually
# good contact/lift/force-closure gates never coincided in one trajectory.
# v8 aligns dense reward with the exact historical-maximum penetration gate,
# removes table-supported force-closure credit, gates lift by continuous
# bimanual support, and adds phase-safe strict-gate frontier credit.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
switch_log="$root/logs/shaped_v8_strict_aligned_switch.log"
objects=(sphere sphere_small cracker_large cracker pyramid cube)
launch_epoch="$(date +%s)"

declare -A devices=(
  [sphere]=cuda:0 [sphere_small]=cuda:1 [cracker_large]=cuda:2
  [cracker]=cuda:3 [cube]=cuda:4 [pyramid]=cuda:5
)
declare -A nominals=(
  [sphere]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/sphere.pt"
  [sphere_small]="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal/sphere_small.pt"
  [cracker_large]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker_large.pt"
  [cracker]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cracker.pt"
  [pyramid]="/media/home/zhangjingzhi/objectflow_xhand_rl_legacy_warmstart_v5f_pose_20260824/nominal/pyramid.pt"
  [cube]="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal/cube.pt"
)
declare -A nominal_hashes=(
  [sphere]="b7d7f40e108496be6f5e65319a554733f07ddc70ac6cbe483e415ddcdad6bc09"
  [sphere_small]="008141c0f78742bf570e3b1dfbf1df75dd9fd023d1e6c1694e6581d12d3009d8"
  [cracker_large]="506d5e07da2994909bb5a4fbf85ec0cbf22b13c1300b3805e7297de5e49bf3ca"
  [cracker]="84c37c36c43f95b72dd3956f4e89f0508a59503b6cd8f68f80e236184e7c764d"
  [pyramid]="b48e4a481d0bf6c06bc56e184fba4d1817bff91f181eab058b7d6696e203de4e"
  [cube]="5a4236eb6056a586c357ad53c8a220c49842eb8c6fa690e9ffc01addbc3eff55"
)

# Objects with no durable formal contact basin return to a short v8 curriculum.
# The other three use immutable checkpoints with direct evidence of a useful
# sub-basin: one reduced-contract success or improving v7 formal gates.
declare -A modes=(
  [sphere]=fresh [sphere_small]=seed [cracker_large]=seed
  [cracker]=fresh [pyramid]=seed [cube]=fresh
)
declare -A checkpoints=(
  [sphere_small]="$root/runs/sphere_small/curriculum_retry_004/model_349.pt"
  [cracker_large]="$root/runs/cracker_large/training_retry_011/model_1600.pt"
  [pyramid]="$root/runs/pyramid/curriculum/model_600.pt"
)
declare -A checkpoint_hashes=(
  [sphere_small]="602994b70a821c43661172d320748645a44750c15dcc0e4310e301c8784a26b4"
  [cracker_large]="8011a7f66d6b6f1f6308639eef7b2dea33c0556eb422ff1f4eb40b63e2edc01b"
  [pyramid]="6de8a01346daa411d4d4bc4dcf352dcfadaab70792d24d8b15d0739815dc15b6"
)
declare -A phases=(
  [sphere]=1 [sphere_small]=1 [cracker_large]=1
  [cracker]=1 [pyramid]=0 [cube]=1
)
declare -A curriculum_seeds=(
  [sphere]=384 [sphere_small]=385 [cracker_large]=386
  [cracker]=387 [pyramid]=388 [cube]=389
)
declare -A formal_seeds=(
  [sphere]=484 [sphere_small]=485 [cracker_large]=486
  [cracker]=487 [pyramid]=488 [cube]=489
)

mkdir -p "$root/logs"

worker_alive() {
  local object="$1"
  ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_(embedded\.(bridge_worker|train)|curriculum\.train)/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") { found=1; exit }
    END { exit(found ? 0 : 1) }
  '
}

restart_watchdog() {
  if ps -ww -eo comm=,args= | awk '
    ($1 == "bash") && $0 ~ /reconcile_target100\.sh/ { found=1; exit }
    END { exit(found ? 0 : 1) }
  '; then return; fi
  echo "$(date --iso-8601=seconds) v8 switch taking over as watchdog" >> "$switch_log"
  exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
}
trap restart_watchdog EXIT

for object in "${objects[@]}"; do
  [[ -f "${nominals[$object]}" ]]
  [[ "$(sha256sum "${nominals[$object]}" | awk '{print $1}')" == "${nominal_hashes[$object]}" ]]
  if [[ "${modes[$object]}" == seed ]]; then
    [[ -f "${checkpoints[$object]}" ]]
    [[ "$(sha256sum "${checkpoints[$object]}" | awk '{print $1}')" == "${checkpoint_hashes[$object]}" ]]
  fi
done
echo "$(date --iso-8601=seconds) preflight passed" >> "$switch_log"

# Stop only this pipeline's superseded trainers/watchdog.  Collector/auditor
# screens remain untouched and all completed directories stay immutable.
screen -S xhand_formal_v7_low_lr_switch -X quit >/dev/null 2>&1 || true
for object in "${objects[@]}"; do
  screen -S "xhand_formal_v7_low_lr_${object}" -X quit >/dev/null 2>&1 || true
done
for pid in $(ps -ww -eo pid=,args= | awk '
  $0 ~ /reconcile_target100\.sh/ { print $1 }
'); do
  [[ "$pid" == "$$" ]] || kill -TERM "$pid" >/dev/null 2>&1 || true
done

for _ in $(seq 1 120); do
  any_alive=0
  for object in "${objects[@]}"; do
    if worker_alive "$object"; then any_alive=1; break; fi
  done
  (( any_alive == 0 )) && break
  sleep 1
done
for object in "${objects[@]}"; do
  if worker_alive "$object"; then
    echo "$(date --iso-8601=seconds) ERROR superseded worker did not stop object=$object" >> "$switch_log"
    exit 1
  fi
done

sleep 90
screen -wipe >/dev/null 2>&1 || true

for object in "${objects[@]}"; do
  screen_name="xhand_formal_v8_strict_${object}"
  log="$root/logs/$object/shaped_v8_strict_bridge_screen.log"
  common=(
    env PYTHONPATH="$repo" "$isaac_python"
    -m migration_4090.xhand_rl_embedded.bridge_worker
    --manifest "$manifest" --object "$object"
    --nominal "${nominals[$object]}" --root "$root"
    --isaac-python "$isaac_python" --device "${devices[$object]}"
    --target 100 --curriculum-iterations 400 --formal-iterations 4000
    --curriculum-envs 128 --formal-envs 128 --collect-envs 128
    --curriculum-seed "${curriculum_seeds[$object]}"
    --curriculum-noise-std 0.03 --formal-seed "${formal_seeds[$object]}"
    --residual-activation-phase "${phases[$object]}"
    --kit-portable-root "/tmp/xhand_rl_embedded_kit/v8_strict/$object"
  )
  if [[ "${modes[$object]}" == fresh ]]; then
    common+=(--fresh-curriculum)
  else
    common+=(
      --skip-curriculum --prefer-seed-checkpoint
      --seed-checkpoint "${checkpoints[$object]}"
    )
  fi
  echo "$(date --iso-8601=seconds) launch object=$object mode=${modes[$object]} device=${devices[$object]}" >> "$switch_log"
  env -u STY screen -L -Logfile "$log" -dmS "$screen_name" "${common[@]}"

  verified=0
  for _ in $(seq 1 180); do
    if worker_alive "$object" && "$isaac_python" - \
      "$root/runs/$object" "${modes[$object]}" "$launch_epoch" \
      "${nominal_hashes[$object]}" "${checkpoint_hashes[$object]:-}" <<'PY'
import json
import sys
from pathlib import Path

object_root = Path(sys.argv[1])
mode = sys.argv[2]
launch_epoch = float(sys.argv[3])
nominal_hash = sys.argv[4]
checkpoint_hash = sys.argv[5]

if mode == "fresh":
    candidates = sorted(
        object_root.glob("curriculum_retry_*/curriculum_training_attempt_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert candidates and candidates[0].stat().st_mtime >= launch_epoch
    payload = json.loads(candidates[0].read_text())
    assert payload["reward_revision"] == "short_horizon_strict_gate_aligned_persistent_barrier_v5"
    assert payload["hard_gate_frontier_reward_weight"] == 20.0
    assert payload["nominal_dataset_sha256"] == nominal_hash
else:
    candidates = sorted(
        object_root.glob("training_retry_*/training_provenance.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert candidates and candidates[0].stat().st_mtime >= launch_epoch
    payload = json.loads(candidates[0].read_text())
    assert payload["reward_shaping_revision"] == "strict_gate_aligned_v8_persistent_max_barrier"
    assert payload["hard_gate_frontier_reward_weight"] == 20.0
    assert payload["learning_rate"] == 5e-5
    assert payload["ppo_transition_revision"] == "low_lr_strict_gate_alignment_v2"
    assert payload["resume_optimizer_state_loaded"] is False
    assert payload["resume_checkpoint_sha256"] == checkpoint_hash
    assert payload["nominal_dataset_sha256"] == nominal_hash
    assert payload["exact_training_rollout_capture"] is True
PY
    then verified=1; break; fi
    sleep 1
  done
  if (( verified == 0 )); then
    echo "$(date --iso-8601=seconds) ERROR v8 worker/provenance verification failed object=$object" >> "$switch_log"
    exit 1
  fi
  echo "$(date --iso-8601=seconds) verified object=$object screen=$screen_name" >> "$switch_log"
  sleep 20
done

echo "$(date --iso-8601=seconds) all v8 workers verified" >> "$switch_log"
trap - EXIT
exec "$repo/migration_4090/xhand_rl_embedded/reconcile_target100.sh"
