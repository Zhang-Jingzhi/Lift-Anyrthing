#!/usr/bin/env python3
"""Run a short-horizon residual curriculum before formal embedded PPO.

This worker is intentionally separate from :mod:`worker`.  The existing v2
worker remains the formal-only baseline; this entry point writes to a caller
supplied root and never mutates an older run.  The curriculum checkpoint has
the same 145-observation/38-action actor-critic contract as the formal
environment, so it can be loaded as a warm start by the formal trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .contracts import (
    latest_policy_checkpoint,
    resume_checkpoint_iteration,
    sha256_file,
    validate_manifest,
    validate_root,
)
from .storage import accepted_count


def _run(command: list[str], *, repo: Path, log: Path) -> int:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    # Keep long-running Isaac/PPO and collector logs observable while the
    # child process is alive.  This is especially important for a retry: a
    # gate failure should be diagnosable before the whole stage exits.
    environment["PYTHONUNBUFFERED"] = "1"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        return subprocess.run(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT).returncode


def _event(path: Path, **payload: object) -> None:
    row = {"schema": "xhand_rl_embedded_bridge_timing_v1", "time": time.time(), **payload}
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _checkpoint_iteration(path: Path) -> int:
    try:
        return resume_checkpoint_iteration(path)
    except Exception as error:
        raise RuntimeError(f"invalid training-resume checkpoint name: {path}") from error


def _copy_seed(seed: Path, curriculum_root: Path) -> Path:
    curriculum_root.mkdir(parents=True, exist_ok=True)
    target = curriculum_root / seed.name
    if target.exists():
        if sha256_file(target) != sha256_file(seed):
            raise RuntimeError(f"curriculum seed exists with a different hash: {target}")
        return target
    shutil.copy2(seed, target)
    return target


def _checkpoint_action_semantics(
    checkpoint: Path, *, fallback_phase: int
) -> dict[str, float | int | bool]:
    """Recover the action units that produced a checkpoint.

    Curriculum and formal policies share their network contract, but the same
    normalized action has a different physical meaning when integration,
    joint scales, activation phase, or residual bounds change.  A transferred
    checkpoint must therefore keep all five values, not just its weights.
    """

    defaults: dict[str, float | int] = {
        "instantaneous_residual_weight": 0.0,
        "residual_integration": 0.02,
        "arm_action_scale_rad": 0.08,
        "hand_action_scale_rad": 0.20,
        "residual_activation_phase": int(fallback_phase),
        "residual_limit_rad": 0.40,
    }
    candidates = [checkpoint.parent / "training_provenance.json"]
    candidates.extend(
        sorted(
            checkpoint.parent.glob("curriculum_training_attempt_*.json"),
            reverse=True,
        )
    )
    payload: dict[str, object] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, TypeError, ValueError):
            continue
        break
    for key in defaults:
        value = payload.get(key)
        if value is not None:
            defaults[key] = int(value) if key == "residual_activation_phase" else float(value)
    schema = str(payload.get("schema", ""))
    # Curriculum attempt metadata records the horizon as episode_length_s;
    # older formal-transfer code only looked for source_episode_length_s and
    # therefore skipped the conservative transfer clamp for those seeds.
    source_episode_length_s = payload.get("source_episode_length_s")
    if source_episode_length_s is None:
        source_episode_length_s = payload.get("episode_length_s")
    if schema.startswith("xhand_rl_curriculum_") and source_episode_length_s is not None:
        defaults["source_is_curriculum"] = True
        defaults["source_episode_length_s"] = float(source_episode_length_s)
    return defaults


def _formal_transfer_action_semantics(
    checkpoint: Path,
    *,
    fallback_phase: int,
    formal_episode_length_s: float,
) -> dict[str, float | int | bool]:
    """Preserve residual action meaning across curriculum/formal horizons.

    Integrated residuals are accumulated once per policy step.  Copying the
    same coefficient from a 3.2 s curriculum into an 8.0 s formal episode
    gives the policy 2.5x the integrated action budget and can destroy the
    contact basin before lift.  Normalize only curriculum transfers by the
    source/target horizon ratio.  Formal-to-formal retries remain unchanged
    except for an explicit level-3 lift-rescue profile; level-4 seeds retain
    their exact action units.
    """

    semantics = _checkpoint_action_semantics(
        checkpoint, fallback_phase=fallback_phase
    )
    if semantics.get("source_is_curriculum") is True:
        source_episode_length_s = float(semantics["source_episode_length_s"])
        if source_episode_length_s <= 0.0 or formal_episode_length_s <= 0.0:
            raise ValueError("episode lengths must be positive for residual normalization")
        original = float(semantics["residual_integration"])
        semantics["residual_integration"] = (
            original * source_episode_length_s / formal_episode_length_s
        )
        # Curriculum attempts from the older exploratory profiles carried
        # 0.20/0.35-rad arm/hand scales into formal PPO.  That transfer is
        # too aggressive for a rare hard-success seed: one update can leave
        # the mesh contact basin before the strict prefix is measured.  Clamp
        # only newly created formal lineages; the source checkpoint and all
        # historical retries remain unchanged.
        # Preserve the 3.2 s -> 8.0 s normalized budget (0.008 for the
        # canonical 0.02 curriculum setting).  The former 0.004 cap left
        # several objects unable to reproduce their measured lift prefix;
        # level-3 formal seeds receive the same explicit rescue profile below.
        semantics["residual_integration"] = min(
            float(semantics["residual_integration"]), 0.008
        )
        semantics["arm_action_scale_rad"] = min(
            float(semantics["arm_action_scale_rad"]), 0.12
        )
        semantics["hand_action_scale_rad"] = min(
            float(semantics["hand_action_scale_rad"]), 0.25
        )
        semantics["residual_limit_rad"] = min(
            float(semantics["residual_limit_rad"]), 0.30
        )
        semantics["residual_integration_before_horizon_normalization"] = original
        semantics["formal_episode_length_s"] = float(formal_episode_length_s)
        semantics["curriculum_horizon_normalized"] = True
    # Recent level-3 retries retained a real contact/clearance prefix but
    # inherited only 0.0032--0.004 rad/step of residual authority.  A first
    # rescue probe that also enlarged arm/hand scales destroyed contact
    # entirely, so keep those physical scales exact and make only a small
    # integrated-residual increase.  Level-4 checkpoints are deliberately
    # left untouched because their stable-closure action semantics are valuable.
    if checkpoint.name.endswith("_level_3.pt"):
        semantics["residual_integration"] = max(
            float(semantics["residual_integration"]), 0.005
        )
        semantics["level3_lift_rescue_override"] = True
    return semantics


def _apply_canonical_nominal_override(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """Force production retries onto the append-only nominal in the root marker.

    The long-lived shell supervisor may have been started before a nominal
    correction and therefore still pass a v3 path.  Bridge workers are
    short-lived Python processes, so enforce the current production lineage at
    this boundary as well.  A mismatched seed cannot be reused with the new
    mesh: start a fresh curriculum/formal retry instead of silently combining
    v3 policy weights with v4b geometry.  This is scoped to roots that
    explicitly declare the v4b nominal bank and never edits the old attempt.
    """

    marker_path = args.root / "EMBEDDED_RL_PIPELINE_ROOT.json"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if marker.get("schema") != "xhand_rl_embedded_generation_root_v2":
        return None
    nominal_root = str(marker.get("nominal_root", ""))
    if (
        "meshaware_nominal_v4b_20260826" not in nominal_root
        and "meshaware_nominal_v4c_closed_20260826" not in nominal_root
    ):
        return None
    target = (Path(nominal_root) / f"{args.object}.pt").resolve()
    current = args.nominal.resolve()
    if not target.is_file() or current == target:
        return None

    # Do not transfer checkpoints or curriculum metadata across nominal
    # lineages.  The fresh allocation is append-only and leaves every old v3
    # run available for audit/debugging.
    old_nominal = current
    args.nominal = target
    args.seed_checkpoint = None
    args.prefer_seed_checkpoint = False
    args.skip_curriculum = False
    args.fresh_curriculum = True
    override_log = args.root / "logs" / args.object / "canonical_nominal_override.jsonl"
    override_log.parent.mkdir(parents=True, exist_ok=True)
    with override_log.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "xhand_rl_embedded_canonical_nominal_override_v1",
                    "time": time.time(),
                    "object": args.object,
                    "old_nominal": str(old_nominal),
                    "new_nominal": str(target),
                    "reason": "root_marker_current_meshaware_nominal_supersedes_stale_worker_argument",
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    return old_nominal, target


def _training_complete_checkpoint(training_root: Path) -> Path | None:
    """Return the strongest safe training-resume checkpoint for an attempt.

    Production collection still uses the checkpoint recorded in
    ``TRAINING_COMPLETE.json`` and therefore still requires hard-success
    evidence.  For another PPO retry, however, a persisted same-trajectory
    strict prefix is a better seed than a later policy that forgot that basin.
    """

    marker = training_root / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text())
        checkpoint = Path(payload["checkpoint"]).resolve()
    except (OSError, KeyError, TypeError, ValueError):
        return None
    selection = payload.get("checkpoint_selection", {})
    if selection.get("training_hard_success_observed") is not True:
        prefix = _best_prefix_checkpoint(training_root)
        if prefix is not None:
            return prefix
    return checkpoint if checkpoint.is_file() else None


def _best_prefix_checkpoint(training_root: Path) -> Path | None:
    """Return the strongest hash-verified pre-update strict-prefix policy."""

    marker = training_root / "hard_prefix_rollouts.jsonl"
    if not marker.is_file():
        return None
    candidates: list[tuple[int, float, float, Path]] = []
    for line in marker.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            checkpoint = Path(row["checkpoint"]).resolve()
            if (
                checkpoint.parent != training_root.resolve()
                or row.get("saved_before_ppo_update") is not True
                or not checkpoint.is_file()
                or sha256_file(checkpoint) != row["checkpoint_sha256"]
            ):
                continue
            candidates.append(
                (
                    int(row["max_level"]),
                    float(row.get("mean_score", 0.0)),
                    float(row.get("time", 0.0)),
                    checkpoint,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(candidates, default=None, key=lambda row: row[:3])[3] if candidates else None


def _prefix_state_for_checkpoint(checkpoint: Path) -> Path | None:
    """Return the hash-verified Isaac state paired with a prefix checkpoint.

    RSL-RL stores policy/value weights but not the simulator state that made a
    rare contact prefix.  The capture hooks append the state path and hash to
    the marker beside the checkpoint.  Never guess a state from a different
    checkpoint or accept an un-hashed file: in that case the retry remains a
    normal policy-only resume.
    """

    checkpoint = checkpoint.resolve()
    marker = checkpoint.parent / "hard_prefix_rollouts.jsonl"
    if not marker.is_file():
        return None
    candidates: list[tuple[int, float, Path]] = []
    for line in marker.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_checkpoint = Path(row["checkpoint"]).resolve()
            state_value = row.get("prefix_state")
            state_hash = row.get("prefix_state_sha256")
            state_path = Path(state_value).resolve() if state_value else None
            if (
                row_checkpoint != checkpoint
                or row.get("saved_before_ppo_update") is not True
                or state_path is None
                or not state_path.is_file()
                or not state_hash
                or sha256_file(state_path) != str(state_hash)
            ):
                continue
            candidates.append(
                (
                    int(row.get("max_level", 0)),
                    float(row.get("time", 0.0)),
                    state_path,
                )
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(candidates, default=None, key=lambda row: row[:2])[2] if candidates else None


def _best_prefix_checkpoint_with_state(
    object_root: Path, *, nominal: Path
) -> tuple[Path, Path] | None:
    """Find the strongest same-nominal prefix checkpoint with exact state."""

    candidates: list[tuple[int, float, Path, Path]] = []
    for marker in object_root.glob("training_retry_*/hard_prefix_rollouts.jsonl"):
        for line in marker.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                checkpoint = Path(row["checkpoint"]).resolve()
                state_value = row.get("prefix_state")
                state_hash = row.get("prefix_state_sha256")
                state_path = Path(state_value).resolve() if state_value else None
                if (
                    row.get("saved_before_ppo_update") is not True
                    or state_path is None
                    or not checkpoint.is_file()
                    or not state_path.is_file()
                    or sha256_file(checkpoint) != str(row["checkpoint_sha256"])
                    or not state_hash
                    or sha256_file(state_path) != str(state_hash)
                ):
                    continue
                metadata = checkpoint.parent / "training_provenance.json"
                if not metadata.is_file():
                    continue
                payload = json.loads(metadata.read_text())
                if Path(str(payload.get("nominal_dataset", ""))).resolve() != nominal.resolve():
                    continue
                candidates.append(
                    (
                        int(row.get("max_level", 0)),
                        float(row.get("time", checkpoint.stat().st_mtime)),
                        checkpoint,
                        state_path,
                    )
                )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if not candidates:
        return None
    _, _, checkpoint, state_path = max(candidates, key=lambda row: row[:2])
    return checkpoint, state_path


def _training_hard_success_observed(training_root: Path) -> bool:
    """Return whether the completed formal attempt ever saw a hard success.

    A completed run with no terminal hard success has no evidence that replaying
    the same immutable checkpoint will produce a production sample.  Such a run
    should move to an independent PPO retry after one bounded collection
    attempt, while runs that did observe a hard success still receive the
    normal operational retries for startup/interruption failures.
    """

    marker = training_root / "TRAINING_COMPLETE.json"
    try:
        payload = json.loads(marker.read_text())
        selection = payload.get("checkpoint_selection", {})
        return bool(selection.get("training_hard_success_observed", False))
    except (OSError, TypeError, ValueError):
        return False


def _select_formal_attempt(
    object_root: Path, production_target: int
) -> tuple[Path, Path | None, bool]:
    """Choose the formal directory and optional seed for this bridge pass.

    The first formal attempt remains ``training``.  Once it is complete but
    has not produced the locked target count, later bridge invocations create
    ``training_retry_###`` directories and continue PPO from the previous
    completed checkpoint.  This preserves every prior attempt and prevents a
    failed collector from looping forever on one unchanged policy.
    """

    primary = object_root / "training"
    if accepted_count(object_root.parent.parent, object_root.name) >= production_target:
        return primary, None, False
    if not (primary / "TRAINING_COMPLETE.json").is_file():
        return primary, None, False

    # Give the original formal checkpoint one complete production-collection
    # attempt before allocating a retry training directory.  This is important
    # for roots created by an earlier first-success run: that run may have
    # materialized one receipt with ``--target 1`` even though the locked
    # production target is 100.  A failed collection is recorded in timing;
    # only then should the next invocation advance to a new PPO attempt.
    timing_path = object_root / "timing.jsonl"
    last_collect_return = None
    failed_collect_attempts = 0
    last_collect_payload = None
    if timing_path.is_file():
        try:
            for line in timing_path.read_text().splitlines():
                payload = json.loads(line)
                if payload.get("stage") == "formal_collect":
                    last_collect_payload = payload
                    last_collect_return = payload.get("return_code")
                    if payload.get("return_code") not in (0, None):
                        failed_collect_attempts += 1
        except (OSError, TypeError, ValueError):
            last_collect_return = None
    # Isaac Sim can terminate the collector on SIGINT while the Python
    # subprocess still reports return code 0.  A normal collector always
    # appends a final heartbeat row before returning; a missing ``final`` row
    # therefore identifies an interrupted attempt and must not be mistaken
    # for a successful collection.  Count it as a failed bounded attempt so
    # the retry policy can advance without repeating the same checkpoint.
    if last_collect_return == 0 and last_collect_payload is not None:
        heartbeat_path = object_root / "collection_heartbeat.jsonl"
        try:
            heartbeat_rows = [
                json.loads(line)
                for line in heartbeat_path.read_text().splitlines()
                if line.strip()
            ]
            last_heartbeat = heartbeat_rows[-1] if heartbeat_rows else None
        except (OSError, TypeError, ValueError):
            last_heartbeat = None
        accepted_now = accepted_count(object_root.parent.parent, object_root.name)
        if accepted_now < production_target and last_heartbeat is not None and not last_heartbeat.get("final", False):
            last_collect_return = -signal.SIGINT
            failed_collect_attempts += 1
    if last_collect_return in (None, 0):
        return primary, None, False
    # Failed collections can be retried with the same immutable formal
    # checkpoint only for a likely Isaac/PhysX interruption.  A normal
    # bounded-horizon miss (the collector's RuntimeError, return code 1) is
    # evidence that this policy did not find a production basin, so move to
    # an independent PPO retry immediately.  This avoids spending another
    # multi-hour collection pass on an unchanged policy.
    # If formal PPO never emitted a hard success, repeating the same checkpoint
    # after a full bounded collection only replays a policy with no observed
    # production basin.  Advance to an independent retry promptly.  A run that
    # did emit a hard success keeps the longer operational-retry allowance so a
    # transient Isaac/PhysX startup failure does not trigger unnecessary PPO
    # retraining.
    hard_success_observed = _training_hard_success_observed(primary)
    # Keep one same-checkpoint retry for Isaac/PhysX crashes only.  If the
    # last bounded attempt reached its horizon (return code 1), do not repeat
    # it even when training had an isolated hard-success event.
    crash_like = last_collect_return in (-6, -9, -15, - signal.SIGABRT, - signal.SIGKILL)
    max_operational_retries = 1 if (hard_success_observed and crash_like) else 0
    if failed_collect_attempts <= max_operational_retries:
        return primary, None, False

    completed = [(0, primary, _training_complete_checkpoint(primary))]
    for path in sorted(object_root.glob("training_retry_*")):
        try:
            index = int(path.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoint = _training_complete_checkpoint(path)
        if checkpoint is not None:
            completed.append((index, path, checkpoint))

    # A retry directory left without a completion marker may contain an intact
    # checkpoint after a process interruption.  Resume it in place instead of
    # creating a duplicate attempt.  An explicitly abandoned directory is
    # immutable historical evidence and must not be resumed; this is used when
    # a locked nominal warm-start is corrected (for example, the table-cleared
    # 200% pyramid requires a matching arm-height adaptation).
    for path in sorted(object_root.glob("training_retry_*"), reverse=True):
        if (path / "ABANDONED.json").is_file():
            continue
        if not (path / "TRAINING_COMPLETE.json").is_file():
            checkpoint = _best_prefix_checkpoint(path) or latest_policy_checkpoint(path)
            if checkpoint is not None:
                return path, checkpoint, True

    _, _, seed = max(completed, key=lambda row: row[0])
    if seed is None:
        raise RuntimeError("formal completion marker has no intact checkpoint")
    # Include abandoned/incomplete directory indices when allocating the next
    # attempt, so a preserved retry_001 is never reused or overwritten.
    existing_indices = []
    for path in object_root.glob("training_retry_*"):
        try:
            existing_indices.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    next_index = max([index for index, _, _ in completed] + existing_indices) + 1
    return object_root / f"training_retry_{next_index:03d}", seed, True


def _allocate_fresh_formal_retry(object_root: Path) -> Path:
    """Allocate an append-only formal retry directory.

    A fresh curriculum is a new policy lineage.  It must not reuse an
    incomplete or historical formal directory, because doing so can silently
    resume an older checkpoint and discard the newly learned curriculum
    policy.  Directory indices include abandoned and interrupted attempts so
    no historical path is ever overwritten.
    """

    existing_indices: list[int] = []
    for path in object_root.glob("training_retry_*"):
        try:
            existing_indices.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    next_index = max(existing_indices, default=0) + 1
    return object_root / f"training_retry_{next_index:03d}"


def _formal_retry_hyperparameters(
    formal_root: Path, *, is_formal_retry: bool, residual_activation_phase: int = 2
) -> tuple[float, float, float, float, float, bool, int]:
    """Return a non-repeating exploration schedule for formal attempts.

    A collector failure is evidence that replaying the same formal checkpoint
    with the same exploration distribution is not making progress.  Keep the
    first attempt conservative because it starts from the curriculum policy,
    then widen exploration on independent ``training_retry_###`` directories.
    The directory name is part of the immutable attempt lineage, so this does
    not alter an existing run or overwrite an earlier checkpoint.
    """

    if not is_formal_retry:
        return -20.0, 30.0, 0.995, 0.05, 0.0001, True, 0
    match = re.fullmatch(r"training_retry_(\d+)", formal_root.name)
    if match is None:
        raise RuntimeError(f"invalid formal retry directory: {formal_root}")
    retry_index = int(match.group(1))
    # The hard penetration gate remains 0.5 mm.  The first retry is deliberately
    # clearance-dominant: the observed retry_001 policies acquire bilateral
    # contact but remain several millimetres inside the object, so a soft
    # barrier cannot reach the strict gate.  The terminal bonus is increased
    # only for credit assignment; acceptance itself is unchanged.
    # Reachability shaping is intentionally softer than the historical
    # millimetre-scale penetration barrier.  A large negative coefficient
    # makes the first contact basin unreachable for a residual policy even
    # though the hard terminal gate remains unchanged at 0.5 mm.
    schedule = (
        (-16.0, 1800.0, 0.999, 0.08, 0.0001),
        (-18.0, 2200.0, 0.999, 0.06, 0.0001),
        (-20.0, 2600.0, 0.999, 0.04, 0.0001),
    )
    penetration, terminal, gamma, noise, entropy = schedule[min(retry_index - 1, len(schedule) - 1)]
    # A formal retry is normally seeded from an exact pre-update prefix that
    # contains a rare bilateral-contact/lift event.  High Normal exploration
    # destroys that basin before PPO can refine clearance, so preserve it with
    # a small fixed std for every object and every coarse retry phase.  The
    # old 0.12 exploratory std was useful for the phase-2 search, but it can
    # erase the very contact that v10's dense force-closure shaping is meant
    # to refine.  This only affects newly allocated retry directories; active
    # runs remain immutable and are never changed in place.
    if is_formal_retry and residual_activation_phase in (0, 1, 2, 3):
        noise = min(noise, 0.04)
    # Keep retry exploration bounded.  On a resumed RSL-RL checkpoint the
    # learned policy std is restored from the checkpoint; without freezing it,
    # the entropy term can inflate the std well above the scheduled 0.10--0.20
    # range and destroy otherwise useful bilateral contact during lift/hold.
    # This only affects future independent retries; an active retry directory
    # is never modified or restarted in place.
    return penetration, terminal, gamma, noise, entropy, True, retry_index


def _maybe_handoff_reconcile_supervisor(root: Path, repo: Path) -> None:
    """Reload the append-only reconcile loop once, from a live bridge worker.

    The supervisor is normally launched in a detached ``screen`` and therefore
    does not reload a modified shell script in place.  A bridge worker is
    already running on the Isaac host, so it is a safe in-scope control point
    for a one-shot handoff.  The marker is explicit and append-only; the lock
    prevents six simultaneous workers from starting duplicate supervisors.
    """
    marker = next(
        (
            candidate
            for candidate in (
                root / "SUPERVISOR_HANDOFF_REQUEST_V24.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V23.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V22.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V21.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V20.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V19.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V18.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V17.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V16.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V15.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V14.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V13.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V12.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V11.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V10.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V9.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V8.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V7.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V6.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V5.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V4.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V3.json",
                root / "SUPERVISOR_HANDOFF_REQUEST_V2.json",
                root / "SUPERVISOR_HANDOFF_REQUEST.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if marker is None:
        return
    try:
        payload = json.loads(marker.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if payload.get("root") and Path(str(payload["root"])).resolve() != root.resolve():
        return
    if payload.get("do_not_delete") is not True:
        return
    if marker.name.endswith("_V24.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V24"
    elif marker.name.endswith("_V23.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V23"
    elif marker.name.endswith("_V22.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V22"
    elif marker.name.endswith("_V21.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V21"
    elif marker.name.endswith("_V20.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V20"
    elif marker.name.endswith("_V19.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V19"
    elif marker.name.endswith("_V18.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V18"
    elif marker.name.endswith("_V17.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V17"
    elif marker.name.endswith("_V16.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V16"
    elif marker.name.endswith("_V15.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V15"
    elif marker.name.endswith("_V14.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V14"
    elif marker.name.endswith("_V13.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V13"
    elif marker.name.endswith("_V12.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V12"
    elif marker.name.endswith("_V11.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V11"
    elif marker.name.endswith("_V10.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V10"
    elif marker.name.endswith("_V9.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V9"
    elif marker.name.endswith("_V8.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V8"
    elif marker.name.endswith("_V7.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V7"
    elif marker.name.endswith("_V6.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V6"
    elif marker.name.endswith("_V5.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V5"
    elif marker.name.endswith("_V4.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V4"
    elif marker.name.endswith("_V2.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V2"
    elif marker.name.endswith("_V3.json"):
        lock_name = "SUPERVISOR_HANDOFF_STARTED_V3"
    else:
        lock_name = "SUPERVISOR_HANDOFF_STARTED"
    lock = root / lock_name
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"time": time.time(), "pid": os.getpid()}, handle)
            handle.write("\n")
        log = root / "logs" / "reconcile_target100.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "event": "bridge_worker_supervisor_handoff",
                        "object_worker_pid": os.getpid(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        for session in (
            "xhand_formal_embedded_reconcile_target100",
            "xhand_formal_v5r_watchdog",
        ):
            subprocess.run(
                ["screen", "-S", session, "-X", "quit"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        supervisor = repo / "migration_4090" / "xhand_rl_embedded" / "reconcile_target100.sh"
        command = (
            f"exec '{supervisor}' >> '{root / 'logs' / 'reconcile_target100.log'}' 2>&1"
        )
        subprocess.run(
            ["screen", "-dmS", "xhand_formal_embedded_reconcile_target100", "bash", "-lc", command],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        # Leave the marker and lock as append-only evidence.  A later manual
        # handoff can remove only the lock after verifying no new supervisor
        # exists; never mutate retry data here.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--curriculum-iterations", type=int, default=400)
    parser.add_argument("--formal-iterations", type=int, default=4000)
    parser.add_argument("--curriculum-envs", type=int, default=128)
    parser.add_argument("--formal-envs", type=int, default=128)
    parser.add_argument("--collect-envs", type=int, default=64)
    parser.add_argument(
        "--collection-lock",
        type=Path,
        default=Path("/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"),
        help="host-wide advisory lock used to serialize Isaac/Omniverse collectors",
    )
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument(
        "--prefer-seed-checkpoint",
        action="store_true",
        help="use --seed-checkpoint for the next independent retry even when older retries exist",
    )
    parser.add_argument(
        "--skip-curriculum",
        action="store_true",
        help="use the supplied/available checkpoint directly for a new formal retry",
    )
    parser.add_argument(
        "--fresh-curriculum",
        action="store_true",
        help=(
            "ignore prior curriculum/checkpoint state and allocate a new curriculum "
            "directory; used when the nominal geometry has been corrected"
        ),
    )
    parser.add_argument("--curriculum-seed", type=int, default=184)
    parser.add_argument(
        "--curriculum-noise-std",
        type=float,
        default=0.03,
        help="fixed PPO exploration std for a new curriculum warm start",
    )
    parser.add_argument("--curriculum-entropy-coef", type=float, default=0.001)
    parser.add_argument("--curriculum-learning-rate", type=float)
    parser.add_argument("--curriculum-residual-integration", type=float)
    parser.add_argument("--curriculum-residual-limit-rad", type=float)
    parser.add_argument("--curriculum-arm-action-scale-rad", type=float)
    parser.add_argument("--curriculum-hand-action-scale-rad", type=float)
    parser.add_argument("--curriculum-penetration-reward-weight", type=float)
    parser.add_argument("--curriculum-penetration-clear-reward-weight", type=float)
    parser.add_argument("--curriculum-proximity-reward-weight", type=float)
    parser.add_argument("--curriculum-hard-gate-frontier-reward-weight", type=float)
    parser.add_argument("--curriculum-bilateral-contact-reward-weight", type=float)
    parser.add_argument("--curriculum-contact-continuity-reward-weight", type=float)
    parser.add_argument("--curriculum-contact-diversity-reward-weight", type=float)
    parser.add_argument("--curriculum-lift-height-reward-weight", type=float)
    parser.add_argument("--formal-seed", type=int, default=84)
    parser.add_argument(
        "--formal-noise-std",
        type=float,
        help=(
            "optional fixed PPO exploration std for a strong pre-update seed; "
            "when omitted, use the retry schedule"
        ),
    )
    parser.add_argument(
        "--formal-residual-activation-phase",
        type=int,
        choices=(0, 1, 2, 3),
        help=(
            "override only the formal retry residual activation phase; "
            "2 keeps approach/close on the nominal trajectory and enables "
            "residual control at lift; 3 delays it until hold refinement"
        ),
    )
    parser.add_argument(
        "--residual-activation-phase",
        type=int,
        choices=(0, 1, 2, 3),
        default=2,
        help="first coarse phase in which new retry residuals are enabled",
    )
    parser.add_argument("--kit-portable-root", type=Path)
    args = parser.parse_args()

    if args.target <= 0 or args.curriculum_iterations <= 0 or args.formal_iterations <= 0:
        raise ValueError("target and iteration counts must be positive")
    _apply_canonical_nominal_override(args)
    if not args.nominal.is_file():
        raise FileNotFoundError(args.nominal)
    if not args.isaac_python.is_file():
        raise FileNotFoundError(args.isaac_python)

    repo = Path(__file__).resolve().parents[2]
    _maybe_handoff_reconcile_supervisor(args.root, repo)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    validate_root(args.root, args.manifest, manifest)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    # The locked manifest is the production contract.  A worker may be
    # launched with a smaller smoke/first-success target, but it must never
    # stop the production bridge below the immutable per-object target.  This
    # prevents an old pilot-style ``--target 1`` invocation from silently
    # terminating after one receipt in a root whose formal target is 100.
    production_target = max(int(args.target), int(manifest["target_per_object"]))

    object_root = args.root / "runs" / args.object
    # cracker_large has accumulated only policy-only formal prefixes: no
    # hash-verified Isaac state exists for any of them, so replaying another
    # formal PPO retry cannot reproduce the contact basin.  Route that
    # particular no-state lineage back through a new, append-only curriculum
    # discovery run instead of spending another 4k-iteration formal tail on
    # the same unreachable seed.  This does not alter any existing attempt.
    if (
        args.skip_curriculum
        and args.object == "cracker_large"
        and args.seed_checkpoint is not None
        and args.seed_checkpoint.is_file()
        and _prefix_state_for_checkpoint(args.seed_checkpoint) is None
        and _best_prefix_checkpoint_with_state(
            object_root, nominal=args.nominal
        ) is None
    ):
        args.skip_curriculum = False
        args.fresh_curriculum = True
    curriculum_root = object_root / "curriculum"
    if args.fresh_curriculum:
        # Never overwrite a prior curriculum, even if it has a completion
        # marker.  This is needed when a resized object receives a corrected
        # nominal pose: an old PPO checkpoint is tied to the old observation
        # and residual basin and is not a valid fresh warm start.
        existing_curriculum_indices = []
        for path in object_root.glob("curriculum_retry_*"):
            try:
                existing_curriculum_indices.append(int(path.name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        next_curriculum_index = max(existing_curriculum_indices, default=0) + 1
        curriculum_root = object_root / f"curriculum_retry_{next_curriculum_index:03d}"
    formal_root = object_root / "training"
    logs_root = args.root / "logs" / args.object
    timing = object_root / "timing.jsonl"
    object_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    root_token = hashlib.sha256(str(args.root.resolve()).encode()).hexdigest()[:16]
    kit_base = args.kit_portable_root or (
        Path(tempfile.gettempdir()) / "xhand_rl_embedded_kit" / root_token / args.object
    )

    # Prefer an explicitly supplied checkpoint, then a formal checkpoint, and
    # finally the newest in-progress curriculum checkpoint.  The last branch
    # is required for a supervisor retry: restarting a non-empty curriculum
    # directory without a resume checkpoint would fail closed even though an
    # intact checkpoint is available locally.  Old v5n/v5o roots are never
    # inspected or modified by this worker.
    # A fresh curriculum intentionally starts from the policy's neutral
    # initialization.  Explicit seeds still win for targeted experiments, but
    # the normal bridge path must not silently inherit the old nominal basin.
    seed = args.seed_checkpoint.resolve() if args.seed_checkpoint else (
        None if args.fresh_curriculum else latest_policy_checkpoint(formal_root)
    )
    if seed is None and not (curriculum_root / "CURRICULUM_TRAINING_COMPLETE.json").is_file():
        seed = latest_policy_checkpoint(curriculum_root)
    if seed is not None and not seed.is_file():
        raise FileNotFoundError(seed)

    curriculum_complete = curriculum_root / "CURRICULUM_TRAINING_COMPLETE.json"
    if args.skip_curriculum and args.fresh_curriculum:
        raise ValueError("--fresh-curriculum and --skip-curriculum are mutually exclusive")
    if args.skip_curriculum:
        # A curriculum checkpoint from the current root can be a known
        # no-contact basin.  Targeted retries may start directly from an
        # immutable checkpoint from an earlier exploratory root; formal
        # provenance still records the current locked manifest and the new
        # reward/workspace revision.  No old directory is modified.
        curriculum_checkpoint = seed
        if curriculum_checkpoint is None or not curriculum_checkpoint.is_file():
            raise RuntimeError("--skip-curriculum requires an intact seed checkpoint")
    elif not curriculum_complete.is_file():
        curriculum_seed = None
        curriculum_prefix_state = None
        if seed is not None:
            curriculum_seed = _copy_seed(seed, curriculum_root)
            curriculum_prefix_state = _prefix_state_for_checkpoint(seed)
        curriculum_max_iterations = args.curriculum_iterations
        if curriculum_seed is not None:
            curriculum_max_iterations += _checkpoint_iteration(curriculum_seed)
        curriculum_envs = int(args.curriculum_envs)
        # The mesh-aware v4c nominal still leaves a measurable approach gap
        # for the sphere and cracker families.  Repeated fresh curricula for
        # those two objects were therefore flat at zero contact: the policy
        # had no action range large enough to reach the surface.  After the
        # supervisor has already tried the conservative profiles (retry >= 12),
        # widen only the *new* curriculum's bounded residual search and dense
        # proximity signal.  Formal acceptance and all physical gates remain
        # unchanged; this is an append-only reachability profile, not a
        # relaxation of strict validation.
        curriculum_noise_std = float(args.curriculum_noise_std)
        curriculum_residual_integration = args.curriculum_residual_integration
        curriculum_residual_limit = args.curriculum_residual_limit_rad
        curriculum_arm_scale = args.curriculum_arm_action_scale_rad
        curriculum_hand_scale = args.curriculum_hand_action_scale_rad
        curriculum_proximity = args.curriculum_proximity_reward_weight
        try:
            curriculum_retry_number = int(curriculum_root.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            curriculum_retry_number = 0
        if (
            args.fresh_curriculum
            and args.object in {"sphere", "cracker", "cracker_large"}
            and curriculum_retry_number >= 12
        ):
            curriculum_noise_std = max(curriculum_noise_std, 0.04)
            curriculum_residual_integration = max(
                float(curriculum_residual_integration or 0.0), 0.016
            )
            curriculum_residual_limit = max(
                float(curriculum_residual_limit or 0.0), 0.55
            )
            curriculum_arm_scale = max(float(curriculum_arm_scale or 0.0), 0.24)
            curriculum_hand_scale = max(float(curriculum_hand_scale or 0.0), 0.40)
            curriculum_proximity = max(float(curriculum_proximity or 0.0), 6.0)
        # The large cracker's 128-env curriculum can exceed the available
        # PhysX tensor memory during scene initialization on cuda:2.  Use a
        # smaller vectorized batch for this discovery stage only; formal PPO
        # remains on its normal production environment count.
        if args.object == "cracker_large" and args.fresh_curriculum:
            curriculum_envs = min(curriculum_envs, 64)
        command = [
            str(args.isaac_python), "-m", "migration_4090.xhand_rl_curriculum.train",
            "--manifest", str(args.manifest.resolve()), "--object", args.object,
            "--nominal-dataset", str(args.nominal.resolve()), "--output", str(curriculum_root.resolve()),
            "--num-envs", str(curriculum_envs), "--max-iterations", str(curriculum_max_iterations),
            "--iteration-chunk", "1",
            "--seed", str(args.curriculum_seed), "--device", args.device, "--headless",
            "--kit-portable-root", str(kit_base / "curriculum"),
            "--init-noise-std", str(curriculum_noise_std),
            "--entropy-coef", str(args.curriculum_entropy_coef), "--freeze-policy-noise",
            "--stop-on-first-curriculum-success",
            "--residual-activation-phase", str(args.residual_activation_phase),
        ]
        for flag, value in (
            ("--learning-rate", args.curriculum_learning_rate),
            ("--residual-integration", curriculum_residual_integration),
            ("--residual-limit-rad", curriculum_residual_limit),
            ("--arm-action-scale-rad", curriculum_arm_scale),
            ("--hand-action-scale-rad", curriculum_hand_scale),
        ):
            if value is not None:
                command += [flag, str(value)]
        for flag, value in (
            ("--penetration-reward-weight", args.curriculum_penetration_reward_weight),
            ("--penetration-clear-reward-weight", args.curriculum_penetration_clear_reward_weight),
            ("--proximity-reward-weight", curriculum_proximity),
            ("--hard-gate-frontier-reward-weight", args.curriculum_hard_gate_frontier_reward_weight),
            ("--bilateral-contact-reward-weight", args.curriculum_bilateral_contact_reward_weight),
            ("--contact-continuity-reward-weight", args.curriculum_contact_continuity_reward_weight),
            ("--contact-diversity-reward-weight", args.curriculum_contact_diversity_reward_weight),
            ("--lift-height-reward-weight", args.curriculum_lift_height_reward_weight),
        ):
            if value is not None:
                command += [flag, str(value)]
        if curriculum_seed is not None:
            command += ["--resume-checkpoint", str(curriculum_seed.resolve())]
        if curriculum_prefix_state is not None:
            command += ["--resume-prefix-state", str(curriculum_prefix_state)]
        started = time.monotonic()
        code = _run(command, repo=repo, log=logs_root / "curriculum.log")
        _event(timing, stage="curriculum", duration_s=time.monotonic() - started, return_code=code)
        if code != 0 or not curriculum_complete.is_file():
            raise RuntimeError(f"curriculum training failed for {args.object}; see {logs_root / 'curriculum.log'}")

    if not args.skip_curriculum:
        curriculum_payload = json.loads(curriculum_complete.read_text())
        curriculum_checkpoint = Path(curriculum_payload["checkpoint"]).resolve()
        if not curriculum_checkpoint.is_file():
            raise RuntimeError(f"curriculum checkpoint is missing: {curriculum_checkpoint}")
        if sha256_file(curriculum_checkpoint) != curriculum_payload["checkpoint_sha256"]:
            raise RuntimeError("curriculum checkpoint hash changed")
        # The ordinary completion checkpoint is saved after PPO updates.  A
        # rare successful rollout can be erased by that update, so prefer the
        # immutable pre-update checkpoint when its append-only marker and hash
        # verify.  It remains only a formal warm start, never an acceptance.
        exact_marker = curriculum_root / "curriculum_hard_success_rollouts.jsonl"
        exact_successes = []
        if exact_marker.is_file():
            for line in exact_marker.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    checkpoint = Path(row["checkpoint"]).resolve()
                    expected = str(row["checkpoint_sha256"])
                    serial = int(row["success_serial"])
                    iteration = int(row["iteration"])
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    row.get("saved_before_ppo_update") is True
                    and checkpoint.is_file()
                    and checkpoint.parent == curriculum_root.resolve()
                    and sha256_file(checkpoint) == expected
                ):
                    exact_successes.append((serial, iteration, checkpoint))
        if exact_successes:
            curriculum_checkpoint = max(exact_successes, key=lambda row: (row[0], row[1]))[2]

    formal_root, formal_retry_seed, is_formal_retry = _select_formal_attempt(
        object_root, production_target
    )
    if (args.fresh_curriculum and not args.skip_curriculum) or (
        args.skip_curriculum and args.seed_checkpoint is not None
    ):
        # A current fresh curriculum (or an explicitly supplied curriculum
        # seed in --skip-curriculum mode) is intentionally independent of all
        # historical formal attempts.  Always allocate a new formal retry and
        # seed it from the checkpoint selected above; otherwise the presence
        # of an old incomplete retry could make _select_formal_attempt choose
        # that stale lineage instead.
        formal_root = _allocate_fresh_formal_retry(object_root)
        formal_retry_seed = curriculum_checkpoint
        is_formal_retry = True
    formal_complete = formal_root / "TRAINING_COMPLETE.json"
    (
        penetration_reward_weight,
        terminal_success_weight,
        gamma,
        formal_noise_std,
        formal_entropy_coef,
        freeze_formal_noise,
        formal_retry_index,
    ) = _formal_retry_hyperparameters(
        formal_root,
        is_formal_retry=is_formal_retry,
        residual_activation_phase=args.residual_activation_phase,
    )
    if args.formal_noise_std is not None:
        if args.formal_noise_std < 0.0:
            raise ValueError("formal noise std must be non-negative")
        # Strong pre-update prefix checkpoints are rare contact basins.  A
        # retry-wide exploratory std can erase the basin before the first
        # force-closure update, so allow the supervisor to request a small,
        # deterministic refinement distribution for those seeds.
        formal_noise_std = float(args.formal_noise_std)
    if not formal_complete.is_file():
        formal_checkpoint = latest_policy_checkpoint(formal_root)
        completed_retry_dirs = [
            path
            for path in object_root.glob("training_retry_*")
            if (path / "TRAINING_COMPLETE.json").is_file()
        ]
        # For the first retry of a new shaping revision, an explicit
        # contact-preserving seed must outrank the old primary formal
        # checkpoint.  Once that retry completes, subsequent retries resume
        # from the newest checkpoint in this root as usual.
        first_explicit_retry = (
            args.skip_curriculum
            and args.seed_checkpoint is not None
            and not completed_retry_dirs
        )
        resume = (
            seed
            if (first_explicit_retry or args.prefer_seed_checkpoint)
            else (formal_checkpoint or formal_retry_seed or curriculum_checkpoint)
        )
        original_resume = resume
        resume_prefix_state = (
            _prefix_state_for_checkpoint(resume) if resume is not None else None
        )
        # A legacy level-4 policy-only checkpoint can outrank a newer level-3
        # checkpoint that has the exact Isaac state needed to reproduce its
        # basin.  Prefer the latter whenever the selected seed has no paired
        # state; this is a training-seed choice only and never changes receipt
        # acceptance or historical artifacts.
        if is_formal_retry and resume_prefix_state is None:
            stateful = _best_prefix_checkpoint_with_state(
                object_root, nominal=args.nominal
            )
            if stateful is not None:
                resume, resume_prefix_state = stateful
        action_semantics = _formal_transfer_action_semantics(
            resume,
            fallback_phase=args.residual_activation_phase,
            formal_episode_length_s=float(manifest["episode_schedule"]["episode_length_s"]),
        )
        if args.formal_residual_activation_phase is not None:
            # This is an explicit strategy change for a new append-only
            # formal lineage.  Keep all other action units from the source
            # checkpoint, but prevent PPO residuals from moving the hands
            # away from the nominal contact basin during approach/close.
            action_semantics["residual_activation_phase"] = int(
                args.formal_residual_activation_phase
            )
        formal_max_iterations = args.formal_iterations
        if is_formal_retry and resume is not None:
            # A retry should add a full formal horizon instead of immediately
            # producing a completion marker because the previous checkpoint
            # already reached the old max iteration.
            formal_max_iterations = _checkpoint_iteration(resume) + args.formal_iterations
        command = [
            str(args.isaac_python), "-m", "migration_4090.xhand_rl_embedded.train",
            "--manifest", str(args.manifest.resolve()), "--object", args.object,
            "--nominal-dataset", str(args.nominal.resolve()), "--output", str(formal_root.resolve()),
            "--num-envs", str(args.formal_envs), "--max-iterations", str(formal_max_iterations),
            "--seed", str(args.formal_seed), "--device", args.device, "--headless",
            "--kit-portable-root", str(kit_base / "formal"),
                "--penetration-reward-weight", str(penetration_reward_weight),
                "--penetration-clear-reward-weight", "16.0",
                "--proximity-reward-weight", "3.0",
                "--terminal-success-weight", str(terminal_success_weight),
            "--gamma", str(gamma),
            "--init-noise-std", str(formal_noise_std),
            "--entropy-coef", str(formal_entropy_coef),
            "--resume-checkpoint", str(resume.resolve()),
            "--residual-activation-phase", str(
                action_semantics["residual_activation_phase"]
                if is_formal_retry
                else args.residual_activation_phase
            ),
        ]
        if resume_prefix_state is not None:
            command += ["--resume-prefix-state", str(resume_prefix_state)]
        # Preserve every residual/action unit recorded beside the selected
        # checkpoint.  Changing only a joint scale or residual bound changes
        # the physical meaning of the policy output and previously destroyed
        # otherwise valid curriculum contact basins during formal transfer.
        # Existing attempts are never modified or restarted in place.
        if is_formal_retry:
            command += [
                "--instantaneous-residual-weight", str(action_semantics["instantaneous_residual_weight"]),
                "--residual-integration", str(action_semantics["residual_integration"]),
                "--residual-limit-rad", str(action_semantics["residual_limit_rad"]),
                "--arm-action-scale-rad", str(action_semantics["arm_action_scale_rad"]),
                "--hand-action-scale-rad", str(action_semantics["hand_action_scale_rad"]),
                # New retries already reach contact/lift/clearance (level 3)
                # but stall at stable force closure.  Give the dense
                # force-closure near-miss signal enough scale to compete with
                # the accumulated frontier reward.  This changes only future
                # retry lineages; the boolean force-closure gate and every old
                # checkpoint remain immutable.
                "--stable-lift-reward-weight", "28.0",
                # Hold-only refinement retries receive a denser force-
                # closure signal from periodic hold audits.  This is shaping
                # only; the terminal force-closure gate remains unchanged.
                "--force-closure-reward-weight", "144.0",
                "--hard-gate-frontier-reward-weight", "32.0",
                "--disturbance-direction-reward-weight", "12.0",
                "--force-closure-shape-all-hold-contacts",
                "--force-closure-requires-clearance",
                "--disturbance-reward-requires-force-closure",
                "--reset-optimizer-on-resume",
                # Prefix checkpoints are rare basins.  Historical retries
                # used "--learning-rate", "5e-5"; that update was
                # large enough to erase contact/lift after one PPO step in
                # the current curves, so future independent retries refine
                # them at 1e-5; the adaptive hook lowers it again after a
                # newly captured prefix.  V21 still lost level-3 basins at
                # 5e-6, so future append-only retries use 2e-6; the exact
                # boolean gates and source checkpoint remain unchanged.
                "--learning-rate", "2e-6",
                # Preserve a newly captured contact/lift/clear basin more
                # aggressively while the policy searches the next strict
                # gate.  The hook only changes future updates in this new
                # append-only lineage.
                # A level-3 prefix still collapsed within a few PPO updates
                # at 0.10 on sphere_small.  Use a near-freeze factor for the
                # next append-only retry; the base LR remains 1e-5 before a
                # new prefix is captured, and training_capture clamps the
                # protected LR to its 1e-6 floor.
                "--adaptive-prefix-lr-factor", "0.01",
            ]
        if freeze_formal_noise:
            command.append("--freeze-policy-noise")
        started = time.monotonic()
        code = _run(command, repo=repo, log=logs_root / "formal_training.log")
        _event(
            timing,
            stage="formal_training_retry" if is_formal_retry else "formal_training",
            output=str(formal_root.resolve()),
            resume=str(resume),
            original_resume=(
                str(original_resume) if original_resume is not None else None
            ),
            max_iterations=formal_max_iterations,
            formal_retry_index=formal_retry_index,
            penetration_reward_weight=penetration_reward_weight,
            terminal_success_weight=terminal_success_weight,
            gamma=gamma,
            init_noise_std=formal_noise_std,
            entropy_coef=formal_entropy_coef,
            freeze_policy_noise=freeze_formal_noise,
            action_semantics=action_semantics,
            resume_prefix_state=(
                str(resume_prefix_state) if resume_prefix_state is not None else None
            ),
            resume_prefix_state_sha256=(
                sha256_file(resume_prefix_state)
                if resume_prefix_state is not None
                else None
            ),
            formal_residual_activation_phase_override=(
                args.formal_residual_activation_phase
            ),
            hard_gate_frontier_reward_weight=20.0 if is_formal_retry else 0.0,
            disturbance_direction_reward_weight=8.0 if is_formal_retry else 0.0,
            duration_s=time.monotonic() - started,
            return_code=code,
        )
        if code != 0 or not formal_complete.is_file():
            raise RuntimeError(f"formal training failed for {args.object}; see {logs_root / 'formal_training.log'}")

    formal_payload = json.loads(formal_complete.read_text())
    checkpoint = Path(formal_payload["checkpoint"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != formal_payload["checkpoint_sha256"]:
        raise RuntimeError("formal completed checkpoint is missing or changed")

    if accepted_count(args.root, args.object) < production_target:
        collection_attempts = 0
        if timing.is_file():
            try:
                collection_attempts = sum(
                    1
                    for line in timing.read_text().splitlines()
                    if json.loads(line).get("stage") == "formal_collect"
                )
            except (OSError, TypeError, ValueError):
                collection_attempts = 0
        # The collector samples the PPO Normal distribution.  Incrementing
        # the seed for every bounded attempt prevents an operational retry
        # from replaying the exact same stochastic trajectory sequence.
        collection_seed = 85 + collection_attempts
        attempt_id = (
            f"{formal_root.name}-checkpoint-{checkpoint.stem}-"
            f"seed-{collection_seed}-attempt-{collection_attempts + 1}"
        )
        command = [
            str(args.isaac_python), "-m", "migration_4090.xhand_rl_embedded.collect",
            "--manifest", str(args.manifest.resolve()), "--object", args.object,
            "--nominal-dataset", str(args.nominal.resolve()), "--checkpoint", str(checkpoint),
            "--root", str(args.root.resolve()), "--target", str(production_target),
            "--num-envs", str(args.collect_envs), "--seed", str(collection_seed),
            "--device", args.device, "--headless",
            "--collection-lock", str(args.collection_lock.resolve()),
            "--attempt-id", attempt_id,
            "--kit-teardown-cooldown-s", "75",
            "--kit-portable-root", str(kit_base / "collect"),
            "--penetration-reward-weight", str(penetration_reward_weight),
            "--residual-activation-phase", str(
                formal_payload.get(
                    "residual_activation_phase", args.residual_activation_phase
                )
            ),
        ]
        if is_formal_retry:
            command += [
                "--instantaneous-residual-weight", str(
                    formal_payload["instantaneous_residual_weight"]
                ),
                "--residual-integration", str(formal_payload["residual_integration"]),
                "--residual-limit-rad", str(formal_payload["residual_limit_rad"]),
                "--arm-action-scale-rad", str(formal_payload["arm_action_scale_rad"]),
                "--hand-action-scale-rad", str(formal_payload["hand_action_scale_rad"]),
                "--stable-lift-reward-weight", "28.0",
                "--force-closure-reward-weight", "144.0",
                "--penetration-clear-reward-weight", "16.0",
                "--proximity-reward-weight", "3.0",
                "--hard-gate-frontier-reward-weight", "32.0",
                "--force-closure-shape-all-hold-contacts",
                "--force-closure-requires-clearance",
                "--disturbance-reward-requires-force-closure",
            ]
        started = time.monotonic()
        # The collector itself acquires the host-wide lock before AppLauncher
        # starts Kit.  The bridge must not hold the same lock while waiting for
        # the child: doing so creates a parent/child flock deadlock.  Lock wait
        # timing and final-attempt state are written by the collector heartbeat.
        code = _run(command, repo=repo, log=logs_root / "collect.log")
        heartbeat_path = object_root / "collection_heartbeat.jsonl"
        final_heartbeat = None
        try:
            for line in heartbeat_path.read_text().splitlines():
                row = json.loads(line)
                if row.get("attempt_id") == attempt_id and row.get("final") is True:
                    final_heartbeat = row
        except (OSError, TypeError, ValueError):
            final_heartbeat = None
        raw_code = code
        if code == 0 and accepted_count(args.root, args.object) < production_target:
            # Isaac/Kit has historically swallowed SIGINT during shutdown and
            # returned zero.  A successful collector either reaches the target
            # or records a final bounded-attempt heartbeat.  Treat a zero exit
            # without target completion as a failed attempt so the supervisor
            # cannot silently advance as if a receipt had been produced.
            code = 1
        _event(
            timing,
            stage="formal_collect",
            duration_s=time.monotonic() - started,
            attempt_id=attempt_id,
            raw_return_code=raw_code,
            final_heartbeat_observed=final_heartbeat is not None,
            lock_wait_s=(final_heartbeat or {}).get("lock_wait_s"),
            collection_lock=str(args.collection_lock),
            return_code=code,
        )
        if code != 0:
            raise RuntimeError(f"formal collector failed for {args.object}; see {logs_root / 'collect.log'}")


if __name__ == "__main__":
    main()
