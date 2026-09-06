#!/usr/bin/env bash
set -u

# Keep the active v5r root moving toward its immutable 100-per-object target.
# Existing bridge screens are left alone; a follow-up bridge is launched only
# after the current screen disappears or a worker has crashed.  The bridge
# itself is responsible for resuming the newest intact local checkpoint.

root="/media/home/zhangjingzhi/objectflow_xhand_rl_embedded_formal_v5r_curriculumbridge_20260824"
repo="/media/home/zhangjingzhi/TRO-Grasp-Reproduction"
isaac_python="/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/objectflow_xhand_isaaclab/bin/python"
manifest="$repo/migration_4090/config/xhand_locked_six_rl_embedded_v2.json"
nominal_root="/media/home/zhangjingzhi/objectflow_xhand_rl_locked_pose_warmstart_all_20260825/nominal"
corrected_nominal_root="/media/home/zhangjingzhi/objectflow_xhand_rl_corrected_nominal_20260825/nominal"
meshaware_nominal_root="/media/home/zhangjingzhi/objectflow_xhand_rl_meshaware_nominal_v4c_closed_20260826_final3/nominal"
declare -A nominal_files=(
  # v4c is an append-only, mesh-aware canonical-frame warm-start bank.  It
  # retargets all six objects to their locked mesh extents, puts each object
  # upright on the table, uses closed-hand seed poses only where the real mesh
  # audit found bilateral zero-penetration contact, and keeps the small sphere
  # on the collision-safe open-hand seed.  It is a seed only; the embedded
  # gates still decide acceptance.
  [sphere]="$meshaware_nominal_root/sphere.pt"
  [sphere_small]="$meshaware_nominal_root/sphere_small.pt"
  [cracker_large]="$meshaware_nominal_root/cracker_large.pt"
  [cracker]="$meshaware_nominal_root/cracker.pt"
  [pyramid]="$meshaware_nominal_root/pyramid.pt"
  [cube]="$meshaware_nominal_root/cube.pt"
)
declare -A devices=(
  [sphere]=cuda:0
  [sphere_small]=cuda:1
  [cracker_large]=cuda:2
  [cracker]=cuda:3
  # The active v5r migrated workers run pyramid/cube on cuda:5/cuda:4.
  # Keep automatic restarts on the same allocation; cuda:6/cuda:7 are
  # reserved for the concurrently running collaborator experiments.
  [pyramid]=cuda:5
  [cube]=cuda:4
)

gpu_ready() {
  # Production allocates one GPU per object on cuda:0..5.  If the NVIDIA
  # driver/device nodes disappear, launching another Isaac Kit process only
  # creates a dead retry and can obscure the last useful prefix.  This guard
  # is deliberately outside acceptance: it only pauses the supervisor until
  # the runtime is usable again.
  local count
  [[ -e /dev/nvidiactl && -e /dev/nvidia-uvm ]] || return 1
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
  [[ "$count" =~ ^[0-9]+$ && "$count" -ge 6 ]]
}
count_embedded_receipts() {
  local object="$1"
  find "$root/accepted/$object" -maxdepth 2 -type f -name receipt.json \
    2>/dev/null | wc -l
}

# Curriculum retries are append-only.  When a bounded curriculum attempt
# reaches its horizon without a hard success, keep the best/latest policy
# basin and continue it in a newly allocated retry directory instead of
# restarting from the neutral policy every time.  This preserves every old
# attempt while allowing the controller to cross a narrow contact/lift basin.
latest_curriculum_checkpoint() {
  local object="$1"
  local nominal="$2"
  # Prefer the checkpoint explicitly selected by a completed curriculum that
  # actually observed a hard success.  A newer zero-success retry is not a
  # better seed merely because its mtime is newer.  If no hard-success marker
  # exists, fall back to the newest intact curriculum checkpoint as before.
  "$isaac_python" - "$root/runs/$object" "$nominal" <<'PY'
import json
import sys
import hashlib
from pathlib import Path

object_root = Path(sys.argv[1])
nominal = Path(sys.argv[2]).resolve()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def same_nominal(checkpoint: Path) -> bool:
    parent = checkpoint.parent
    paths = [parent / "training_provenance.json"]
    paths.extend(parent.glob("curriculum_training_attempt_*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if Path(str(payload.get("nominal_dataset", ""))).resolve() == nominal:
            return True
    return False

# Prefer the strongest hash-verified prefix from the same nominal lineage.
# This includes formal and curriculum prefix snapshots; a level-4 curriculum
# hard success is represented as the strongest possible warm start.
prefixes = []
markers = list(object_root.glob("training_retry_*/hard_prefix_rollouts.jsonl"))
markers.extend(object_root.glob("curriculum_retry_*/hard_prefix_rollouts.jsonl"))
for marker in markers:
    try:
        rows = marker.read_text().splitlines()
    except OSError:
        continue
    for line in rows:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            level = int(row["max_level"])
            checkpoint = Path(row["checkpoint"]).resolve()
            expected = str(row["checkpoint_sha256"])
            abandoned_formal_prefix = (
                checkpoint.parent.name.startswith("training_retry_") and level >= 3
            )
            if (
                level < 1
                or row.get("saved_before_ppo_update") is not True
                or (
                    (checkpoint.parent / "ABANDONED.json").is_file()
                    and not abandoned_formal_prefix
                )
                or not checkpoint.is_file()
                or sha256_file(checkpoint) != expected
                or not same_nominal(checkpoint)
            ):
                continue
            # Formal prefixes and curriculum prefixes are not equivalent.  A
            # curriculum "level 4" only means reduced contact/lift/clear/
            # stable-hold gates; it does not include formal force closure,
            # disturbances, or ablations.  Prefer a same-lineage formal
            # level-3/4 prefix over that reduced curriculum success, while
            # still using the curriculum seed when no formal prefix exists.
            source_rank = (
                float(level)
                if checkpoint.parent.name.startswith("training_retry_")
                else min(float(level), 2.5)
            )
            prefixes.append((source_rank, float(row.get("time", checkpoint.stat().st_mtime)), checkpoint))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

for marker in object_root.glob("curriculum_retry_*/curriculum_hard_success_rollouts.jsonl"):
    if (marker.parent / "ABANDONED.json").is_file():
        continue
    try:
        rows = marker.read_text().splitlines()
    except OSError:
        continue
    for line in rows:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            checkpoint = Path(row["checkpoint"]).resolve()
            expected = str(row["checkpoint_sha256"])
            if (
                row.get("saved_before_ppo_update") is not True
                or (checkpoint.parent / "ABANDONED.json").is_file()
                or not checkpoint.is_file()
                or sha256_file(checkpoint) != expected
                or not same_nominal(checkpoint)
            ):
                continue
            # This marker belongs to the reduced curriculum contract.  Keep it
            # below a formal level-3 prefix so a formal strict basin is not
            # discarded merely because the curriculum calls its stable-hold
            # stage "level 4".
            prefixes.append((2.5, float(checkpoint.stat().st_mtime), checkpoint))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

successful = []
for marker in object_root.glob("curriculum*/CURRICULUM_TRAINING_COMPLETE.json"):
    if (marker.parent / "ABANDONED.json").is_file():
        continue
    try:
        payload = json.loads(marker.read_text())
        count = int(payload.get("curriculum_success_serial_count", 0))
        checkpoint = Path(payload["checkpoint"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if count > 0 and checkpoint.is_file() and same_nominal(checkpoint):
        exact = sorted(
            marker.parent.glob("hard_success_preupdate_iter_*_serial_*.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        # A pre-update curriculum success is a stronger formal warm start
        # than the post-update model selected by the ordinary checkpoint
        # policy.  It is still only a seed; formal embedded acceptance remains
        # mandatory downstream.
        # Reduced curriculum hard-success is a strong warm start, but it is
        # not a formal level-4 force-closure/disturbance result.
        successful.append((2.5, count, marker.stat().st_mtime, exact[-1] if exact else checkpoint))
if successful or prefixes:
    # Prefix rows do not have a curriculum-success serial count.  Keep their
    # ranking deterministic instead of accidentally reusing the loop variable
    # from the preceding successful-curriculum scan.
    candidates = [(rank, 0, stamp, checkpoint) for rank, stamp, checkpoint in prefixes]
    candidates.extend(successful)
    print(max(candidates, key=lambda row: (row[0], row[1], row[2]))[3])
    raise SystemExit(0)

fallback = []
for checkpoint in object_root.glob("curriculum*/model_*.pt"):
    if (checkpoint.parent / "ABANDONED.json").is_file():
        continue
    if checkpoint.is_file() and same_nominal(checkpoint):
        fallback.append((checkpoint.stat().st_mtime, checkpoint))
if fallback:
    print(max(fallback, key=lambda row: row[0])[1])
PY
}

successful_curriculum_checkpoint() {
  local checkpoint="$1"
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  "$isaac_python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()

# An abandoned retry is append-only evidence of a failed basin, not a valid
# warm start.  Do not resurrect it just because its old prefix level is high.
if (checkpoint.parent / "ABANDONED.json").is_file():
    # A failed retry remains immutable evidence, but a hash-verified level-3
    # or level-4 pre-update prefix is still a useful seed for an independent
    # rescue lineage.  It is never accepted directly and must pass the new
    # retry's full strict terminal contract again.
    marker = checkpoint.parent / "hard_prefix_rollouts.jsonl"
    allowed = False
    if checkpoint.parent.name.startswith("training_retry_") and marker.is_file():
        try:
            import hashlib
            digest = hashlib.sha256()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            expected_hash = digest.hexdigest()
            for line in marker.read_text().splitlines():
                row = json.loads(line)
                if (
                    Path(row.get("checkpoint", "")).resolve() == checkpoint
                    and row.get("saved_before_ppo_update") is True
                    and int(row.get("max_level", 0)) >= 3
                    and str(row.get("checkpoint_sha256", "")) == expected_hash
                ):
                    allowed = True
                    break
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            allowed = False
    if not allowed:
        raise SystemExit(1)

# A formal pre-update prefix is already a valid same-trajectory formal seed.
# It has no CURRICULUM_TRAINING_COMPLETE.json beside it, so checking only the
# curriculum marker would incorrectly route the next retry through a fresh
# curriculum and discard the formal clearance/force-closure basin.  Verify the
# immutable hash record directly before allowing a direct formal retry.
if checkpoint.name.startswith("hard_prefix_preupdate_iter_"):
    marker = checkpoint.parent / "hard_prefix_rollouts.jsonl"
    if marker.is_file():
        import hashlib
        digest = hashlib.sha256()
        try:
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            expected_hash = digest.hexdigest()
            for line in marker.read_text().splitlines():
                try:
                    row = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    Path(row.get("checkpoint", "")).resolve() == checkpoint
                    and row.get("saved_before_ppo_update") is True
                    and str(row.get("checkpoint_sha256", "")) == expected_hash
                    and int(row.get("max_level", 0)) >= 1
                ):
                    raise SystemExit(0)
        except (OSError, TypeError, ValueError):
            pass

for marker in checkpoint.parent.glob("CURRICULUM_TRAINING_COMPLETE.json"):
    try:
        payload = json.loads(marker.read_text())
        selected = Path(payload["checkpoint"]).resolve()
        successes = int(payload.get("curriculum_success_serial_count", 0))
        hard = bool(
            payload.get("checkpoint_selection", {}).get(
                "training_hard_success_observed", False
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if successes > 0 and hard:
        # The ordinary completion payload points at the nearest post-update
        # ``model_<n>.pt``.  The reconcile seed selector intentionally prefers
        # the immutable PPO-preupdate snapshot because it contains the exact
        # rare successful rollout.  Recognize that snapshot as a successful
        # curriculum seed too; otherwise reconcile would incorrectly launch
        # another curriculum and discard the useful basin.
        if selected == checkpoint:
            raise SystemExit(0)
        marker = checkpoint.parent / "curriculum_hard_success_rollouts.jsonl"
        if marker.is_file():
            for line in marker.read_text().splitlines():
                try:
                    exact = Path(json.loads(line)["checkpoint"]).resolve()
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if exact == checkpoint:
                    raise SystemExit(0)
        # Formal prefix checkpoints are also safe direct-training seeds.  They
        # are recognized by the append-only retry selector above and must not
        # be sent through another short curriculum, which could erase the
        # strict force-closure/disturbance progress.
        if checkpoint.name.startswith("hard_prefix_preupdate_iter_"):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

next_curriculum_retry_index() {
  local object="$1"
  find "$root/runs/$object" -maxdepth 1 -type d \
    -name 'curriculum_retry_[0-9][0-9][0-9]' -printf '%f\n' 2>/dev/null \
    | sed 's/^curriculum_retry_//' | sort -n | tail -n 1
}

next_formal_retry_index() {
  local object="$1" index
  index="$(find "$root/runs/$object" -maxdepth 1 -type d \
    -name 'training_retry_[0-9][0-9][0-9]' -printf '%f\n' 2>/dev/null \
    | sed 's/^training_retry_//' | sort -n | tail -n 1)"
  echo $((10#${index:-0} + 1))
}

checkpoint_activation_phase() {
  local checkpoint="$1"
  # Read the action semantic from the immutable lineage that produced the
  # checkpoint.  The retry profile currently being scheduled is not a valid
  # substitute: a curriculum success from phase 1 must not be transferred to
  # formal PPO at phase 0 merely because the next retry is exploring phase 0.
  "$isaac_python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
for path in (
    checkpoint.parent / "training_provenance.json",
    *sorted(checkpoint.parent.glob("curriculum_training_attempt_*.json")),
):
    try:
        payload = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    value = payload.get("residual_activation_phase")
    if value in (0, 1, 2, 3):
        print(int(value))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

checkpoint_source_seed() {
  local checkpoint="$1"
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  "$isaac_python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
for metadata_path in (
    checkpoint.parent / "training_provenance.json",
    *sorted(checkpoint.parent.glob("curriculum_training_attempt_*.json")),
):
    try:
        payload = json.loads(metadata_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    value = payload.get("seed")
    if value is not None:
        print(int(value))
        raise SystemExit(0)
raise SystemExit(1)
PY
}

# Rotate only newly launched attempts.  The active worker command lines are
# never changed in place, so this is safe while six Isaac jobs are running.
curriculum_retry_options() {
  local object="$1" index seed noise phase checkpoint
  index="$(next_curriculum_retry_index "$object")"
  index=$((10#${index:-0} + 1))
  # A different seed prevents a failed retry from reproducing the exact same
  # rollout sequence.  Keep the schedule bounded for reproducibility.
  seed=$((184 + (index % 11)))
  case $((index % 5)) in
    1) noise="0.03" ;;
    2) noise="0.05" ;;
    3) noise="0.08" ;;
    4) noise="0.12" ;;
    0) noise="0.18" ;;
  esac
  # The locked nominal warm starts have a measured 5--6 mm mesh overlap.
  # Give the first continuation attempts residual control during approach so
  # the policy can move out of that basin before the close phase.  Later
  # retries return to the contact-preserving phase-1 schedule.
  if [[ "$object" == "sphere_small" || "$object" == "cracker" || "$object" == "cube" ]]; then
    # These three current traces are proximity-dominated and lose contact
    # before lift.  Future retries must expose residual control during the
    # approach/clearance phase so the policy can leave the nominal overlap;
    # delaying residuals until lift only reinforces the no-contact basin.
    phase=0
  elif (( index <= 7 )); then
    phase=0
  else
    phase=1
  fi
  # The mesh-aware bank starts in the canonical table frame.  New retries use
  # a three-profile schedule rather than replaying the same PPO update: a
  # contact-lock profile protects a rare bilateral basin, an approach profile
  # gives moderate clearance freedom, and an explore profile widens residuals
  # only when the first two have not crossed the basin boundary.
  if [[ "${nominal_files[$object]}" == "$meshaware_nominal_root"/* ]]; then
    # v4c keeps the nominal approach trajectory intact and provides a
    # collision-audited closed-hand seed for five objects; residual control is
    # still bounded and the formal gates remain authoritative.  Do not let
    # integrated residuals act during close for those seeds: the static mesh
    # audit already found the contact basin, and phase-1 residuals were
    # empirically erasing it before the continuity gate was measured.
    phase=2
    if [[ "$meshaware_nominal_root" == *"nominal_v4c_closed_20260826"* ]]; then
      # v4c is collision-safe but still has an 8--16 mm real-mesh gap in
      # several nominal wrist poses.  A phase-2 retry cannot ever change the
      # approach/close state, so it produces a clean zero-contact trace.  Keep
      # all new v4c curricula on phase 0, with bounded residuals active from
      # approach through close; the policy must first discover actual contact,
      # after which the strict prefix gates constrain lift and hold.  The
      # small sphere remains the most difficult closure case, but no longer
      # receives a different action semantic that would make formal transfer
      # ambiguous.
      phase=0
      case $((index % 3)) in
        0) noise="0.008" ;;
        1) noise="0.012" ;;
        2) noise="0.020" ;;
      esac
    else
      case $((index % 3)) in
        0) noise="0.008" ;;
        1) noise="0.020" ;;
        2) noise="0.030" ;;
      esac
    fi
  fi
  # Keep this final guard immediately before checkpoint selection so an old
  # profile branch cannot accidentally re-enable phase 2 for the active v4c
  # root.  v4c needs residuals during approach/close to discover the real
  # mesh contact band; phase 2 is reserved for a future nominal that has
  # passed the stricter contact-band audit.
  if [[ "${nominal_files[$object]}" == *"nominal_v4c_closed"* ]]; then
    phase=0
  fi
  checkpoint="$(latest_curriculum_checkpoint "$object" "${nominal_files[$object]}")"
  printf '%s\n' "$index" "$seed" "$noise" "$phase" "$checkpoint"
}

screen_active() {
  local object="$1"
  # A detached screen can remain as a stale bash window after its bridge
  # child exits.  Check the actual per-object bridge/train process instead of
  # treating the socket alone as liveness.  A collector is deliberately not a
  # production worker: allowing it to mask a dead trainer can stall PPO while
  # an auxiliary replay occupies the object.  This also recognizes
  # corrected/strict-warm suffixes used by geometry retries.
  # Isaac commands are much longer than ps' default display width.  Without
  # -ww the --object argument can be truncated, which makes a healthy direct
  # trainer look dead and lets the watchdog launch a conflicting retry.
  local found=0 pid args active_output
  while IFS='|' read -r pid args; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    active_output="$(awk '{for (i=1; i<=NF; i++) { if ($i == "--output" && i < NF) { print $(i+1); exit } if ($i ~ /^--output=/) { sub(/^--output=/, "", $i); print $i; exit } }}' <<< "$args")"
    if [[ -n "$active_output" && -f "$active_output/ABANDONED.json" ]]; then
      kill -TERM "$pid" >/dev/null 2>&1 || true
      echo "$(date --iso-8601=seconds) terminate_abandoned_worker object=$object attempt=$active_output pid=$pid source=screen_active" \
        >> "$root/logs/reconcile_target100.log"
      continue
    fi
    found=1
  done < <(ps -ww -eo pid=,comm=,args= | awk -v object="$object" '
    ($2 == "python" || $2 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_(embedded\.bridge_worker|curriculum\.train|embedded\.train)/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") {
      line=$0; sub(/^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", line)
      print $1 "|" line
    }
  ')
  (( found ))
}

hopeless_retry_info() {
  local object="$1"
  # A retry with no prefix progress is a local-basin retry.  Level-3/4 runs
  # get a longer grace window because they already reached clearance or
  # force-closure territory; a level-4 run that still has no disturbance
  # prefix after 600 iterations is rotated into a low-LR refinement.
  local active_output
  active_output="$(ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_embedded\.train/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") {
      for (i = 1; i <= NF; i++) if ($i == "--output" && i < NF) { print $(i + 1); exit }
    }
  ')"
  [[ -n "$active_output" ]] || return 1
  "$isaac_python" - "$root/runs/$object" "$active_output" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
active = Path(sys.argv[2]).resolve()
if active.parent != run.resolve() or (active / "ABANDONED.json").is_file():
    raise SystemExit(1)
models = []
for path in active.glob("model_*.pt"):
    try:
        models.append((int(path.stem.rsplit("_", 1)[1]), path))
    except (IndexError, ValueError):
        pass
if not models:
    raise SystemExit(1)
latest = max(models, key=lambda row: row[0])[0]
# Checkpoint/model numbers are absolute within a resumed retry.  The
# watchdog thresholds below are deliberately expressed in updates of the
# current retry, otherwise a retry resumed from iteration 296 would be
# misclassified as having already trained 296 updates on its first write.
start_iteration = 0
restored_prefix_level = 0
for metadata_path in (
    active / "training_provenance.json",
    *sorted(active.glob("curriculum_training_attempt_*.json")),
):
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    value = metadata.get("start_iteration")
    if value is not None:
        try:
            start_iteration = max(0, int(value))
        except (TypeError, ValueError):
            pass
    restored_value = metadata.get("restored_prefix_level")
    if restored_value is not None:
        try:
            restored_prefix_level = max(0, int(restored_value))
        except (TypeError, ValueError):
            restored_prefix_level = 0
    if value is not None:
        break
progress = max(0, latest - start_iteration)
rows = []
marker = active / "hard_prefix_rollouts.jsonl"
if marker.is_file():
    for line in marker.read_text().splitlines():
        try:
            row = json.loads(line)
            rows.append((int(row.get("iteration", -1)), int(row.get("max_level", 0))))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
max_level = max((level for _, level in rows), default=0)
last_prefix = max((iteration for iteration, _ in rows), default=-1)
prefix_progress = last_prefix - start_iteration if last_prefix >= 0 else -1
# Read the current write-time prefix separately from the historical marker
# list. The capture hook only saves strictly higher levels, so a policy can
# recover a level-3 basin without producing a new marker; the watchdog must
# not rotate that recovered basin merely because last_prefix is old.
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = sorted(active.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    accumulator = EventAccumulator(str(event_files[-1]), size_guidance={"scalars": 0}) if event_files else None
    if accumulator is not None:
        accumulator.Reload()
except Exception:
    accumulator = None
def scalar_values(tag):
    if accumulator is None or tag not in accumulator.Tags().get("scalars", []):
        return []
    return [float(event.value) for event in accumulator.Scalars(tag)]
prefix_current = []
for tag in (
    "embedded/hard_prefix_contact_fraction",
    "embedded/hard_prefix_lifted_fraction",
    "embedded/hard_prefix_clear_fraction",
    "embedded/hard_prefix_stable_force_closure_fraction",
):
    values = accumulator.Scalars(tag) if accumulator is not None and tag in accumulator.Tags().get("scalars", []) else []
    prefix_current.append(max(values, key=lambda row: row.wall_time).value if values else 0.0)
current_level = 0
for value in prefix_current:
    if value > 0.0:
        current_level += 1
    else:
        break
current_coverage = min(prefix_current[:current_level], default=0.0)
force_closure_reward_values = scalar_values("reward/force_closure")
formal_force_closure_values = scalar_values(
    "embedded/gate_formal_force_closure_fraction"
)
force_closure_reward_max = max(force_closure_reward_values, default=0.0)
formal_force_closure_max = max(formal_force_closure_values, default=0.0)
reason = None
if (
    restored_prefix_level > 0
    and progress >= 40
    and current_level < restored_prefix_level
    and prefix_progress <= progress - 20
):
    # Exact-state transfer is intended to begin inside an already observed
    # physical prefix.  If the first bounded rollout cannot reproduce that
    # prefix, waiting thousands of PPO updates only trains from a different
    # basin.  Rotate early; the immutable checkpoint/state pair is retained
    # for the next append-only retry after the phase-boundary fix/profile.
    reason = "restored_prefix_not_recovered"
elif progress >= 120 and max_level == 0 and prefix_progress <= progress - 80:
    # A formal retry seeded from a strong checkpoint should show at least one
    # same-trajectory contact prefix quickly.  If it has none after 120
    # updates, the seed/noise transfer did not preserve the contact basin; a
    # fresh append-only retry with the low-noise strong-seed policy is more
    # informative than waiting for the full 4k horizon.
    reason = "formal_empty_contact_basin"
elif progress >= 150 and max_level <= 1:
    # A strong-seed retry that has not moved beyond contact after 150 updates
    # is in a low-lift basin. Rotate it before the long 4k tail; the next
    # append-only retry can use a different PPO seed/profile.
    reason = "formal_contact_prefix_without_lift"
elif progress >= 300 and max_level <= 1 and prefix_progress <= progress - 100:
    reason = "no_contact_lift_prefix_progress_for_100_iterations"
elif progress >= 300 and 2 <= max_level <= 3 and prefix_progress <= progress - 200 and current_level < max_level:
    # A level-2/3 basin has contact/lift/clearance evidence, but repeating
    # hundreds of updates without advancing the same-trajectory prefix is
    # usually reward-gate divergence rather than useful exploration.  Rotate
    # from the strongest hash-verified prefix so the next formal seed can use
    # a different PPO seed/profile without discarding the basin.
    reason = "no_strict_prefix_progress_for_200_iterations"
elif (
    progress >= 160
    and max_level >= 3
    and current_level < max_level
    and force_closure_reward_max > 0.25
    and formal_force_closure_max < 1.0e-6
):
    # A dense wrench-quality reward without any formal force-closure gate is
    # a known shortcut: the policy is spending updates on table-supported or
    # otherwise incomplete contacts.  Rotate from the immutable prefix into
    # the clearance-gated v12 reward profile instead of waiting for a 4k tail.
    reason = "force_closure_shaping_without_formal_gate"
elif progress >= 180 and max_level == 3 and prefix_progress <= progress - 100 and not (current_level >= 3 and current_coverage >= 0.02):
    # A level-3 capture followed by a long current-policy collapse is a known
    # failure mode for the residual transfer.  Prefix fractions are batch
    # statistics and can dip for several updates, so allow a 100-iteration
    # recovery window before rotating; the next retry still preserves the
    # immutable prefix checkpoint and uses stronger LR protection.
    reason = "prefix_basin_collapsed_after_level3_capture"
elif progress >= 100 and max_level >= 4 and prefix_progress <= progress - 200 and not (current_level >= 4 and current_coverage >= 0.02):
    # Level 4 is the rare stable-force-closure basin.  Disturbance learning
    # can temporarily make the current batch lose the same-trajectory prefix;
    # keep a 200-update recovery window before rotating, while the later
    # no-disturbance rule still bounds a genuinely stalled refinement.
    reason = "level4_prefix_collapsed_after_capture"
elif progress >= 1200 and max_level <= 2 and prefix_progress <= progress - 300:
    reason = "no_strict_prefix_progress_for_300_iterations"
elif progress >= 1800 and max_level == 4 and prefix_progress <= progress - 600 and not (current_level >= 4 and current_coverage >= 0.02):
    # Level 4 means stable force closure was reached, but no disturbance
    # prefix appeared for a long interval.  Re-run from the immutable level-4
    # checkpoint with the low-LR disturbance profile instead of spending the
    # entire horizon repeating the same hold behavior.
    reason = "no_disturbance_prefix_progress_for_600_iterations"
if reason is not None:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": latest,
        "start_iteration": start_iteration,
        "progress_iterations": progress,
        "max_prefix_level": max_level,
        "last_prefix_iteration": last_prefix,
        "current_prefix_level": current_level,
        "current_prefix_coverage": current_coverage,
        "reason": reason,
    }, separators=(",", ":")))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

strategy_mismatch_info() {
  local object="$1"
  # Rotate workers trained against an obsolete nominal before they consume a
  # full 4k horizon.  A positive shaped reward on the old penetrating v3
  # geometry is not evidence that the strict physical prefixes can succeed.
  # Strong-seed formal retries must also use the residual phase selected from
  # the live gate curves; old attempts remain append-only evidence.
  local active_output
  active_output="$(ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_(embedded\.train|curriculum\.train)/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") {
      for (i = 1; i <= NF; i++) if ($i == "--output" && i < NF) { print $(i + 1); exit }
    }
  ')"
  [[ -n "$active_output" ]] || return 1
  "$isaac_python" - "$root/runs/$object" "$active_output" "$object" "${nominal_files[$object]}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1]).resolve()
active = Path(sys.argv[2]).resolve()
object_name = str(sys.argv[3])
target_nominal = Path(sys.argv[4]).resolve()
if active.parent != run or (active / "ABANDONED.json").is_file():
    raise SystemExit(1)
provenance_paths = [active / "training_provenance.json"]
provenance_paths.extend(sorted(active.glob("curriculum_training_attempt_*.json")))
payload = None
for path in provenance_paths:
    try:
        candidate = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if candidate.get("nominal_dataset"):
        payload = candidate
        break
if payload is None:
    raise SystemExit(1)
active_nominal = Path(str(payload.get("nominal_dataset", ""))).resolve()
if active_nominal != target_nominal:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": int(payload.get("start_iteration", 0)),
        "max_prefix_level": 0,
        "last_prefix_iteration": -1,
        "reason": "nominal_lineage_mismatch_to_current_meshaware_nominal",
        "active_nominal": str(active_nominal),
        "target_nominal": str(target_nominal),
        "object": object_name,
    }, separators=(",", ":")))
    raise SystemExit(0)
phase = int(payload.get("residual_activation_phase", 2))
resume = str(payload.get("resume_checkpoint", ""))
strong = "hard_success_preupdate_iter_" in resume or "hard_prefix_preupdate_iter_" in resume
# Spheres previously lost strict continuity when phase 1 was enabled.  For
# non-spherical level-3 seeds the live failure is contact-without-lift, so a
# close-stage residual rescue is intentionally allowed.  A formal level-4
# seed already contains stable force closure under its source phase and must
# preserve phase 0 during the transfer.
level4_formal_source = (
    Path(resume).parent.name.startswith("training_retry_")
    and "_level_4.pt" in Path(resume).name
)
formal_source = Path(resume).parent.name.startswith("training_retry_")
def checkpoint_phase(path_text):
    if not path_text:
        return None
    path = Path(path_text).resolve()
    for metadata_path in (
        path.parent / "training_provenance.json",
        *sorted(path.parent.glob("curriculum_training_attempt_*.json")),
    ):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        value = metadata.get("residual_activation_phase")
        if value in (0, 1, 2, 3):
            return int(value)
    return None

# A strong formal seed must keep the activation phase recorded by the seed;
# changing it here is exactly the action-semantic mismatch that previously
# erased sphere_small's rare contact event.  Level-3/4 prefix seeds now use
# an intentional hold-only refinement phase: approach, close, and lift stay
# on the saved physical basin while PPO refines force closure during hold.
source_phase = checkpoint_phase(resume)
hold_refinement_source = formal_source and (
    "hard_prefix_preupdate_iter_" in resume
    and ("_level_3.pt" in resume or "_level_4.pt" in resume)
)
target_phase = (
    source_phase
    if hold_refinement_source and source_phase in (0, 1, 2)
    else (0 if level4_formal_source else (source_phase if source_phase is not None else phase))
)
integration = float(payload.get("residual_integration", 0.0))
arm_scale = float(payload.get("arm_action_scale_rad", 0.0))
hand_scale = float(payload.get("hand_action_scale_rad", 0.0))
limit = float(payload.get("residual_limit_rad", 0.0))
if strong and not formal_source and (
    integration > 0.004 + 1.0e-9
    or arm_scale > 0.12 + 1.0e-9
    or hand_scale > 0.25 + 1.0e-9
    or limit > 0.30 + 1.0e-9
):
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": int(payload.get("start_iteration", 0)),
        "max_prefix_level": 0,
        "last_prefix_iteration": -1,
        "reason": "formal_strong_seed_action_semantics_mismatch_to_conservative_caps",
    }, separators=(",", ":")))
    raise SystemExit(0)
if strong and phase != target_phase:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": int(payload.get("start_iteration", 0)),
        "max_prefix_level": 0,
        "last_prefix_iteration": -1,
        "reason": f"formal_strong_seed_residual_phase_mismatch_{phase}_to_{target_phase}",
    }, separators=(",", ":")))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

hopeless_curriculum_info() {
  local object="$1"
  # A curriculum run that has not produced a modest continuous-contact/lift
  # signal by this point is usually optimizing proximity while destroying the
  # nominal basin. Rotate it append-only so the next retry can use the
  # contact-lock profile; do not interrupt early startup or a run with lift.
  local active_output
  active_output="$(ps -ww -eo comm=,args= | awk -v object="$object" '
    ($1 == "python" || $1 ~ /^python[0-9.]+$/) &&
    $0 ~ /migration_4090\.xhand_rl_curriculum\.train/ &&
    $0 ~ ("--object[ =]+" object "([ =]|$)") {
      for (i = 1; i <= NF; i++) if ($i == "--output" && i < NF) { print $(i + 1); exit }
    }
  ')"
  [[ -n "$active_output" ]] || return 1
  "$isaac_python" - "$root/runs/$object" "$active_output" "${nominal_files[$object]}" <<'PY'
import json
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = Path(sys.argv[1]).resolve()
active = Path(sys.argv[2]).resolve()
target_nominal = Path(sys.argv[3]).resolve()
if active.parent != run or (active / "ABANDONED.json").is_file():
    raise SystemExit(1)
models = []
for path in active.glob("model_*.pt"):
    try:
        models.append((int(path.stem.rsplit("_", 1)[1]), path))
    except (IndexError, ValueError):
        pass
if not models:
    raise SystemExit(1)
latest = max(models, key=lambda row: row[0])[0]
if latest < 50:
    raise SystemExit(1)
events = sorted(active.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
if not events:
    raise SystemExit(1)
try:
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    acc.Reload()
except Exception:
    raise SystemExit(1)
tags = set(acc.Tags().get("scalars", []))

def values(tag):
    return [float(x.value) for x in acc.Scalars(tag)] if tag in tags else []

terminal = values("embedded/terminal_success_fraction")
contact = values("embedded/gate_bilateral_contact_continuity_fraction")
lift = values("embedded/gate_lift_height_fraction")
prefix = []
for tag in (
    "embedded/hard_prefix_contact_fraction",
    "embedded/hard_prefix_lifted_fraction",
    "embedded/hard_prefix_clear_fraction",
    "embedded/hard_prefix_stable_force_closure_fraction",
):
    prefix.extend(values(tag))
max_terminal = max(terminal, default=0.0)
max_contact = max(contact, default=0.0)
max_lift = max(lift, default=0.0)
max_prefix = max(prefix, default=0.0)
recent_contact = sum(contact[-10:]) / min(10, len(contact)) if contact else 0.0
recent_lift = sum(lift[-10:]) / min(10, len(lift)) if lift else 0.0

# Mesh-aware nominal banks use a longer, explicit grace window than the old
# v3/v4 formal retries.  v4c has a collision-audited closed-hand seed for
# five objects and an open-hand seed for sphere_small; both remain subject to
# the same strict embedded gates.
is_meshaware_new = False
for provenance in [active / "curriculum_training_attempt.json", *sorted(active.glob("curriculum_training_attempt_*.json")), active / "training_provenance.json"]:
    try:
        payload = json.loads(provenance.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    nominal = payload.get("nominal_dataset")
    if nominal and Path(str(nominal)).resolve() == target_nominal:
        is_meshaware_new = (
            "nominal_v4b" in str(target_nominal)
            or "nominal_v4c_closed" in str(target_nominal)
            or payload.get("nominal_lineage") in {"meshaware_v4b", "meshaware_v4c"}
        )
        break

if is_meshaware_new:
    source_phase = int(payload.get("residual_activation_phase", 0))
    # v4c is mesh-safe but not guaranteed to be inside the real PhysX contact
    # band.  A phase-1/phase-2 retry can therefore spend its whole horizon
    # with zero force because the residual is disabled during approach.  Such
    # a trace is not evidence that the object is ungraspable; it is evidence
    # that the action semantic cannot reach the contact basin.  Rotate it into
    # the phase-0 discovery schedule, which is active before close and is
    # bounded by the same strict penetration gate.
    if (
        "nominal_v4c_closed" in str(target_nominal)
        and source_phase in {1, 2}
        and latest >= 70
        and max_terminal <= 0.0
        and max_contact < 0.05
        and max_lift < 0.02
        and max_prefix < 1.0
    ):
        print(json.dumps({
            "attempt": active.name,
            "latest_iteration": latest,
            "max_contact": max_contact,
            "max_lift": max_lift,
            "max_prefix": max_prefix,
            "reason": "meshaware_v4c_late_residual_empty_contact_switch_to_phase0",
        }, separators=(",", ":")))
        raise SystemExit(0)
    # Keep phase-0 discovery alive longer than the old 120-update switch.  A
    # no-contact phase-0 trace needs a wider residual profile, not a phase-1
    # retry that again disables approach motion.  The retry index changes the
    # bounded noise/scale profile; the append-only watchdog rotates only after
    # the longer grace window below.
    # Keep the open-hand curriculum alive for at least 180 updates.  Only an
    # essentially empty basin after 260 updates is rotated; a rare contact
    # or lift signal is preserved through the 400-update horizon.
    if latest >= 260 and max_terminal <= 0.0 and max_contact < 0.005 and max_lift < 0.005:
        print(json.dumps({
            "attempt": active.name,
            "latest_iteration": latest,
            "max_contact": max_contact,
            "max_lift": max_lift,
            "max_prefix": max_prefix,
            "reason": "meshaware_curriculum_empty_contact_basin_after_grace",
        }, separators=(",", ":")))
        raise SystemExit(0)
    if latest >= 320 and max_terminal <= 0.0 and max_prefix < 1.0 and recent_contact < 0.002 and recent_lift < 0.002:
        print(json.dumps({
            "attempt": active.name,
            "latest_iteration": latest,
            "max_contact": max_contact,
            "max_lift": max_lift,
            "max_prefix": max_prefix,
            "recent_contact": recent_contact,
            "recent_lift": recent_lift,
            "reason": "meshaware_curriculum_flat_after_grace",
        }, separators=(",", ":")))
        raise SystemExit(0)
    # Do not apply the v3 early-rotation rules below to mesh-aware banks.  A delayed but
    # genuine contact basin is exactly what this curriculum is designed to
    # discover.
    raise SystemExit(1)
# A level-0/1 curriculum that has gone flat after the warm-start window is
# not benefiting from exploratory updates.  Rotate it before the long-tail
# collapse rule so the next retry can use the contact-lock profile.
if latest >= 80 and max_prefix <= 1.0 and recent_contact < 0.005 and recent_lift < 0.005:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": latest,
        "max_contact": max_contact,
        "max_lift": max_lift,
        "max_prefix": max_prefix,
        "recent_contact": recent_contact,
        "recent_lift": recent_lift,
        "reason": "curriculum_low_level_no_progress",
    }, separators=(",", ":")))
    raise SystemExit(0)
# A completely empty basin is different from a rare near-miss: after 50
# iterations with essentially no continuous contact/lift signal, switch
# profiles early instead of spending the entire 400-iteration curriculum on
# a proximity-only policy.
if latest >= 50 and max_terminal <= 0.0 and max_lift < 0.05 and max_contact < 0.10 and max_prefix < 1.0:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": latest,
        "max_contact": max_contact,
        "max_lift": max_lift,
        "max_prefix": max_prefix,
        "reason": "curriculum_empty_contact_basin",
    }, separators=(",", ":")))
    raise SystemExit(0)
# A previous prefix can be real but still be unusable if the policy has
# collapsed for a long tail of updates.  Once the current event has reached
# 150 iterations, rotate only this recent-collapse case when no level-4
# evidence exists; future retries can now preserve any new prefix pre-update.
if latest >= 150 and max_terminal <= 0.0 and max_prefix < 4.0 and recent_contact < 0.01 and recent_lift < 0.01:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": latest,
        "max_contact": max_contact,
        "max_lift": max_lift,
        "max_prefix": max_prefix,
        "recent_contact": recent_contact,
        "recent_lift": recent_lift,
        "reason": "curriculum_recent_contact_collapse",
    }, separators=(",", ":")))
    raise SystemExit(0)
# Do not rotate a run that has found a genuine contact/lift basin; it may only
# need the remaining curriculum horizon to stabilize clearance.
if max_terminal <= 0.0 and max_lift < 0.05 and max_contact < 0.20 and max_prefix < 2.0:
    print(json.dumps({
        "attempt": active.name,
        "latest_iteration": latest,
        "max_contact": max_contact,
        "max_lift": max_lift,
        "max_prefix": max_prefix,
        "reason": "curriculum_no_contact_lift_progress",
    }, separators=(",", ":")))
    raise SystemExit(0)
raise SystemExit(1)
PY
}

abandon_hopeless_retry() {
  local object="$1"
  local info="$2"
  local attempt
  attempt="$($isaac_python - "$root/runs/$object" "$info" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
info = json.loads(sys.argv[2])
attempt = run / info["attempt"]
marker = attempt / "ABANDONED.json"
if not marker.exists():
    marker.write_text(json.dumps({
        "schema": "xhand_append_only_abandoned_retry_v1",
        "reason": info["reason"],
        "latest_iteration": info["latest_iteration"],
        "max_prefix_level": info.get("max_prefix_level", info.get("max_prefix", 0)),
        "last_prefix_iteration": info.get("last_prefix_iteration", -1),
        "current_prefix_level": info.get("current_prefix_level", 0),
        "current_prefix_coverage": info.get("current_prefix_coverage", 0.0),
        "max_contact": info.get("max_contact"),
        "max_lift": info.get("max_lift"),
        "recent_contact": info.get("recent_contact"),
        "recent_lift": info.get("recent_lift"),
    }, indent=2) + "\n")
print(attempt)
PY
  )"
  # A bridge may have already written ABANDONED.json while its Isaac child is
  # still unwinding.  Do not let that stale child mask the object forever:
  # terminate only a trainer whose exact --output path is this retry.  This is
  # deliberately narrower than pkill-by-object, so a newly launched append-
  # only retry on the same GPU cannot be touched.
  local attempt_root="$root/runs/$object/$attempt"
  while read -r pid args; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    case "$args" in
      *migration_4090.xhand_rl_embedded.train*|*migration_4090.xhand_rl_curriculum.train*)
        if [[ "$args" == *"--output $attempt_root"* || "$args" == *"--output=$attempt_root"* ]]; then
          kill -TERM "$pid" >/dev/null 2>&1 || true
          echo "$(date --iso-8601=seconds) terminate_abandoned_worker object=$object attempt=$attempt pid=$pid" \
            >> "$root/logs/reconcile_target100.log"
        fi
        ;;
    esac
  done < <(ps -ww -eo pid=,args=)
  screen -S "xhand_formal_v5r_${object}_bridge" -X quit >/dev/null 2>&1 || true
  echo "$(date --iso-8601=seconds) abandon_hopeless object=$object attempt=$attempt reason=$info" \
    >> "$root/logs/reconcile_target100.log"
}

while :; do
  if ! gpu_ready; then
    echo "$(date --iso-8601=seconds) gpu_unavailable pause_launches" >> "$root/logs/reconcile_target100.log"
    sleep 60
    continue
  fi
  all_complete=1
  for object in sphere sphere_small cracker_large cracker pyramid cube; do
    embedded_count="$(count_embedded_receipts "$object")"
    # The current production contract evaluates contact continuity, arm lift,
    # lift height, gravity hold, 6+6 disturbances, historical PhysX
    # penetration, formal force closure, and both single-hand ablations inside
    # the RL episode.  The intentionally omitted multistage visual-mesh audit
    # is not a second acceptance requirement in this run.
    if (( embedded_count < 100 )); then
      all_complete=0
    fi
    if (( embedded_count < 100 )) && info="$(strategy_mismatch_info "$object")"; then
      abandon_hopeless_retry "$object" "$info"
      sleep 75
      continue
    fi
    if (( embedded_count < 100 )) && info="$(hopeless_retry_info "$object")"; then
      abandon_hopeless_retry "$object" "$info"
      sleep 75
      continue
    fi
    if (( embedded_count < 100 )) && info="$(hopeless_curriculum_info "$object")"; then
      abandon_hopeless_retry "$object" "$info"
      sleep 75
      continue
    fi
    # Once the embedded strict store reaches its immutable capacity, no
    # external replay validator is needed for this production revision.
    if (( embedded_count >= 100 )) || screen_active "$object"; then
      continue
    fi
    # A just-finished Isaac process can disappear from ps before Vulkan,
    # PhysX, and allocator resources are fully quiescent.  Wait before every
    # missing-worker launch, then re-check so another supervisor cannot create
    # a duplicate during the cooldown window.
    echo "$(date --iso-8601=seconds) cooldown_before_launch object=$object seconds=90" \
      >> "$root/logs/reconcile_target100.log"
    sleep 90
    if ! gpu_ready; then
      echo "$(date --iso-8601=seconds) gpu_unavailable after_cooldown object=$object" >> "$root/logs/reconcile_target100.log"
      break
    fi
    if screen_active "$object"; then
      continue
    fi
    mapfile -t retry_options < <(curriculum_retry_options "$object")
    retry_index="${retry_options[0]}"
    curriculum_seed="${retry_options[1]}"
    curriculum_noise="${retry_options[2]}"
    curriculum_phase="${retry_options[3]}"
    curriculum_checkpoint="${retry_options[4]:-}"
    # Per-retry curriculum profiles.  These affect only a newly allocated
    # directory; active attempts and every old checkpoint remain immutable.
    curriculum_entropy="0.001"
    curriculum_learning_rate="0.00015"
    curriculum_residual_integration="0.020"
    curriculum_residual_limit="0.400"
    curriculum_arm_scale="0.160"
    curriculum_hand_scale="0.300"
    curriculum_penetration="-2.0"
    curriculum_clear="4.0"
    curriculum_proximity="4.0"
    curriculum_frontier="8.0"
    curriculum_bilateral="18.0"
    curriculum_continuity="12.0"
    curriculum_diversity="3.0"
    curriculum_lift="16.0"
    case $((retry_index % 3)) in
      0)
        # Contact-lock: preserve a rare physical contact basin while PPO
        # learns the lift/hold continuation.  The low LR/std prevents the
        # first policy update from erasing the nominal contact geometry.
        curriculum_entropy="0.0"
        curriculum_learning_rate="0.00005"
        curriculum_residual_integration="0.008"
        curriculum_residual_limit="0.200"
        curriculum_arm_scale="0.080"
        curriculum_hand_scale="0.180"
        curriculum_penetration="-0.5"
        curriculum_clear="2.0"
        curriculum_proximity="1.0"
        curriculum_frontier="30.0"
        curriculum_bilateral="35.0"
        curriculum_continuity="30.0"
        curriculum_diversity="6.0"
        curriculum_lift="30.0"
        ;;
      1)
        # Approach: moderate clearance correction without sacrificing contact.
        curriculum_entropy="0.0005"
        curriculum_learning_rate="0.00010"
        curriculum_residual_integration="0.012"
        curriculum_residual_limit="0.300"
        curriculum_arm_scale="0.120"
        curriculum_hand_scale="0.250"
        curriculum_penetration="-1.0"
        curriculum_clear="4.0"
        curriculum_proximity="2.0"
        curriculum_frontier="16.0"
        curriculum_bilateral="24.0"
        curriculum_continuity="18.0"
        curriculum_diversity="4.0"
        curriculum_lift="22.0"
        ;;
      2) ;;
    esac
    if [[ "$meshaware_nominal_root" == *"nominal_v4c_closed_20260826"* ]]; then
      # Closed-hand mesh-aware seeds already contain a physical contact
      # basin.  Use conservative residuals to learn lift/hold without
      # erasing it; sphere_small is the exception because its seed remains
      # open-handed and needs broader closure exploration.
      if [[ "$object" == "sphere_small" ]]; then
        case $((retry_index % 3)) in
          0) curriculum_noise="0.080" ;;
          1) curriculum_noise="0.120" ;;
          2) curriculum_noise="0.160" ;;
        esac
        curriculum_residual_integration="0.025"
        curriculum_residual_limit="0.550"
        curriculum_arm_scale="0.220"
        curriculum_hand_scale="0.500"
        curriculum_proximity="6.0"
        curriculum_frontier="18.0"
        curriculum_bilateral="28.0"
        curriculum_continuity="26.0"
        curriculum_diversity="5.0"
        curriculum_lift="24.0"
      else
        # v4c is collision-safe but the real mesh can still be outside the
        # PhysX contact offset.  Keep residuals active from approach/close so
        # PPO can discover that contact band; phase 2 would be unreachable
        # when the nominal starts with zero force.
        curriculum_phase="0"
        case $((retry_index % 3)) in
          0) curriculum_noise="0.008" ;;
          1) curriculum_noise="0.012" ;;
          2) curriculum_noise="0.020" ;;
        esac
        curriculum_entropy="0.0005"
        curriculum_learning_rate="0.00005"
        curriculum_residual_integration="0.006"
        curriculum_residual_limit="0.150"
        curriculum_arm_scale="0.060"
        curriculum_hand_scale="0.140"
        curriculum_penetration="-1.0"
        curriculum_clear="4.0"
        curriculum_proximity="2.0"
        curriculum_frontier="28.0"
        curriculum_bilateral="40.0"
        curriculum_continuity="36.0"
        curriculum_diversity="5.0"
        curriculum_lift="34.0"
      fi
    fi
    # Sphere and cracker have shown repeated long zero-contact curriculum
    # tails with the conservative v4c residual band.  For a new retry with no
    # reusable prefix, widen only the discovery workspace so PPO can reach the
    # real-mesh contact offset; this is still phase-0, bounded, and subject to
    # the same penetration/strict-prefix gates.  Strong prefix checkpoints
    # below override this rescue profile with contact-lock semantics.
    if [[ ("$object" == "sphere" || "$object" == "cracker") && -z "$curriculum_checkpoint" ]] && (( retry_index >= 12 )); then
      curriculum_noise="0.020"
      curriculum_entropy="0.0005"
      curriculum_learning_rate="0.00005"
      curriculum_residual_integration="0.012"
      curriculum_residual_limit="0.300"
      curriculum_arm_scale="0.120"
      curriculum_hand_scale="0.240"
      curriculum_penetration="-1.5"
      curriculum_clear="5.0"
      curriculum_proximity="3.0"
      curriculum_frontier="30.0"
      curriculum_bilateral="42.0"
      curriculum_continuity="38.0"
      curriculum_diversity="5.0"
      curriculum_lift="34.0"
    fi
    # An exact pre-update hard-success/prefix checkpoint is a rare physical
    # basin.  Regardless of retry index, protect it with the contact-lock
    # profile instead of applying an exploratory schedule that could erase it
    # before the formal stage sees it.
    if [[ "$curriculum_checkpoint" == *hard_success_preupdate_iter_* || "$curriculum_checkpoint" == *hard_prefix_preupdate_iter_*level_4.pt ]]; then
      curriculum_entropy="0.0"
      curriculum_learning_rate="0.00005"
      curriculum_residual_integration="0.008"
      curriculum_residual_limit="0.200"
      curriculum_arm_scale="0.080"
      curriculum_hand_scale="0.180"
      curriculum_noise="0.008"
      curriculum_penetration="-0.5"
      curriculum_clear="2.0"
      curriculum_proximity="1.0"
      curriculum_frontier="30.0"
      curriculum_bilateral="35.0"
      curriculum_continuity="30.0"
      curriculum_diversity="6.0"
      curriculum_lift="30.0"
    fi
    # A checkpoint is a valid seed for the new nominal only when its own
    # immutable training metadata names the exact same nominal file.  This
    # prevents a fresh mesh-aware v2 curriculum from silently resuming a
    # checkpoint trained against the old tilted/penetrating geometry, while
    # still allowing later v2 retries to continue their own best basin.
    if [[ -n "$curriculum_checkpoint" ]]; then
      if ! "$isaac_python" - "$curriculum_checkpoint" "${nominal_files[$object]}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
nominal = Path(sys.argv[2]).resolve()
parent = checkpoint.parent
metadata = []
metadata.extend(parent.glob("curriculum_training_attempt_*.json"))
metadata.extend(parent.glob("training_provenance.json"))
for path in metadata:
    try:
        row = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        continue
    if Path(str(row.get("nominal_dataset", ""))).resolve() == nominal:
        raise SystemExit(0)
raise SystemExit(1)
PY
      then
        curriculum_checkpoint=""
      fi
    fi
    seed_arg=""
    if [[ -n "$curriculum_checkpoint" && -f "$curriculum_checkpoint" ]]; then
      seed_arg="--seed-checkpoint '$curriculum_checkpoint'"
    fi
    bridge_mode="--fresh-curriculum"
    formal_seed_arg=""
    formal_noise_arg=""
    formal_residual_phase_arg=""
    # A checkpoint selected at an observed curriculum hard success is already
    # the scarce contact/lift/clearance basin we need.  Training another
    # curriculum continuation from it can drift away before formal PPO starts
    # (the 200% pyramid model_600 had 15 successes, while its later retry
    # collapsed to zero contact).  Start a new append-only formal lineage
    # directly from that immutable checkpoint.  Vary the formal seed across
    # retries so a failed lineage is not reproduced exactly.
    # A verified hard-success or level-3/4 pre-update prefix is already a
    # physical contact/lift/clearance basin.  Send every object directly to
    # formal PPO from that immutable checkpoint; routing it through another
    # short curriculum repeatedly erased the rare basin (especially for
    # sphere_small and cube).  Level-1/2 prefixes still use curriculum to
    # acquire the missing contact/lift stages.
    strong_prefix_checkpoint=0
    case "$curriculum_checkpoint" in
      *hard_success_preupdate_iter_*|*hard_prefix_preupdate_iter_*_level_3.pt|*hard_prefix_preupdate_iter_*_level_4.pt)
        strong_prefix_checkpoint=1
        ;;
    esac
    if (( strong_prefix_checkpoint )) && successful_curriculum_checkpoint "$curriculum_checkpoint"; then
      bridge_mode="--skip-curriculum --prefer-seed-checkpoint"
      # Use the append-only formal retry index, not the curriculum retry
      # index.  Several consecutive strong-prefix retries can legitimately
      # share one curriculum checkpoint; tying the PPO seed to that unchanged
      # curriculum index reproduces the same rollout sequence and can loop on
      # one failed basin indefinitely.
      formal_retry_seed_index="$(next_formal_retry_index "$object")"
      source_formal_seed="$(checkpoint_source_seed "$curriculum_checkpoint" 2>/dev/null || true)"
      if [[ -n "$source_formal_seed" ]]; then
        # Reuse the RNG seed that produced the immutable prefix for the first
        # rollout of the new lineage. The directory/checkpoint remains new;
        # only the initial stochastic basin is replayed deterministically.
        formal_seed_arg="--formal-seed '$source_formal_seed'"
      else
        formal_seed_arg="--formal-seed '$((84 + formal_retry_seed_index))'"
      fi
      # V21 could reproduce rare prefixes, but 0.008 action noise still made
      # level-3 contact/lift basins disappear within a few updates.  Preserve
      # the source basin with a small non-zero refinement distribution.  The
      # bridge freezes this std, while PPO can still improve its mean action.
      formal_noise_arg="--formal-noise-std '0.002'"
      # Preserve the activation phase that produced the verified seed.  The
      # previous v5r handoff hard-coded phase 2 for both spheres and phase 1
      # for the other objects; that silently changed the physical meaning of
      # a strong checkpoint during formal transfer.  In particular,
      # sphere_small's curriculum seed was phase 1 but its formal retry was
      # launched at phase 2, so residual closure was disabled and contact
      # dropped to zero even though the curriculum event had a rare success.
      # Level-3/4 formal prefix seeds already contain contact, lift, and
      # penetration-clear evidence. Preserve the source action phase so the
      # checkpoint reproduces its physical contact/lift basin; the new dense
      # hold audits then provide the force-closure refinement signal.
      formal_phase="$(checkpoint_activation_phase "$curriculum_checkpoint" 2>/dev/null || true)"
      if [[ -z "$formal_phase" ]]; then
        formal_phase="$curriculum_phase"
      fi
      formal_residual_phase_arg="--formal-residual-activation-phase '$formal_phase'"
    fi
    mkdir -p "$root/logs/$object"
    echo "$(date --iso-8601=seconds) launch object=$object retry=$retry_index seed=$curriculum_seed noise=$curriculum_noise phase=$curriculum_phase formal_residual_phase=${formal_residual_phase_arg:-source} curriculum_profile=$((retry_index % 3)) lr=$curriculum_learning_rate entropy=$curriculum_entropy integration=$curriculum_residual_integration checkpoint=${curriculum_checkpoint:-none} bridge_mode=$bridge_mode ${formal_seed_arg:-}" \
      >> "$root/logs/reconcile_target100.log"
    # ``screen_active`` has already proved that no Python worker exists.  A
    # previous bridge can nevertheless leave a SCREEN/bash wrapper behind;
    # remove only that stale socket so the fixed per-object screen name can be
    # reused for the new append-only retry.
    screen -S "xhand_formal_v5r_${object}_bridge" -X quit >/dev/null 2>&1 || true
    screen -dmS "xhand_formal_v5r_${object}_bridge" bash -lc \
      "exec env PYTHONPATH='$repo' '$isaac_python' -m migration_4090.xhand_rl_embedded.bridge_worker \
      --manifest '$manifest' --object '$object' --nominal '${nominal_files[$object]}' \
      --root '$root' --isaac-python '$isaac_python' --device '${devices[$object]}' \
      --target 100 --curriculum-iterations 400 --formal-iterations 4000 \
      --curriculum-envs 128 --formal-envs 128 --collect-envs 128 \
      $bridge_mode \
      --curriculum-seed '$curriculum_seed' \
      --curriculum-noise-std '$curriculum_noise' \
      --residual-activation-phase '$curriculum_phase' \
      --kit-portable-root '/tmp/xhand_rl_embedded_kit/v5r/$object' \
      --curriculum-penetration-reward-weight '$curriculum_penetration' \
      --curriculum-penetration-clear-reward-weight '$curriculum_clear' \
      --curriculum-proximity-reward-weight '$curriculum_proximity' \
      --curriculum-hard-gate-frontier-reward-weight '$curriculum_frontier' \
      --curriculum-bilateral-contact-reward-weight '$curriculum_bilateral' \
      --curriculum-contact-continuity-reward-weight '$curriculum_continuity' \
      --curriculum-contact-diversity-reward-weight '$curriculum_diversity' \
      --curriculum-lift-height-reward-weight '$curriculum_lift' \
      --curriculum-entropy-coef '$curriculum_entropy' \
      --curriculum-learning-rate '$curriculum_learning_rate' \
      --curriculum-residual-integration '$curriculum_residual_integration' \
      --curriculum-residual-limit-rad '$curriculum_residual_limit' \
      --curriculum-arm-action-scale-rad '$curriculum_arm_scale' \
      --curriculum-hand-action-scale-rad '$curriculum_hand_scale' \
      $formal_seed_arg \
      $formal_noise_arg \
      $formal_residual_phase_arg \
      $seed_arg \
      > '$root/logs_${object}_bridge_reconcile.log' 2>&1"
    # Avoid concurrent Omniverse startup/teardown races.
    sleep 75
  done
  if (( all_complete )); then
    exit 0
  fi
  sleep 30
done
