#!/usr/bin/env python3
"""Continuously collect from every persisted hard-success PPO lineage.

The formal bridge remains responsible for training and the immutable
100-per-object production target.  This append-only supervisor watches all six
objects for completed PPO attempts that persisted a terminal hard-success
event, then samples the checkpoints immediately before/after that event until
the strict post-audit has accepted at least one sample per object.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .checkpoint_selection import validate_collection_checkpoint
from .contracts import LOCKED_OBJECTS, sha256_file, validate_manifest, validate_root


DEVICES = {
    "sphere": "cuda:0",
    "sphere_small": "cuda:1",
    "cracker_large": "cuda:2",
    "cracker": "cuda:3",
    "cube": "cuda:4",
    "pyramid": "cuda:5",
}
POLICY_STD_SCALES = (1.0, 0.5, 1.5, 2.0)


def _checkpoint_iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def discover_hard_success_lineages(root: Path, object_name: str) -> list[dict[str, Any]]:
    """Return hash-verified adjacent checkpoints for every successful lineage."""

    object_root = root / "runs" / object_name
    entries: list[dict[str, Any]] = []
    for marker in object_root.glob("training*/TRAINING_COMPLETE.json"):
        training_root = marker.parent
        if (training_root / "ABANDONED.json").is_file():
            continue
        try:
            complete = json.loads(marker.read_text())
            selection = complete["checkpoint_selection"]
            if selection.get("training_hard_success_observed") is not True:
                continue
            success_iteration = int(selection["training_hard_success_iteration"])
            selected = Path(complete["checkpoint"]).resolve()
            if not selected.is_file() or sha256_file(selected) != complete["checkpoint_sha256"]:
                continue
            provenance_path = training_root / "training_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            nominal = Path(provenance["nominal_dataset"]).resolve()
            if not nominal.is_file() or sha256_file(nominal) != provenance["nominal_dataset_sha256"]:
                continue
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        checkpoints = sorted(
            (_checkpoint_iteration(path), path.resolve())
            for path in training_root.glob("model_*.pt")
            if path.is_file()
        )
        preceding = [row for row in checkpoints if row[0] <= success_iteration]
        following = [row for row in checkpoints if row[0] >= success_iteration]
        adjacent: dict[Path, int] = {}
        if preceding:
            adjacent[preceding[-1][1]] = preceding[-1][0]
        if following:
            adjacent[following[0][1]] = following[0][0]
        for checkpoint, iteration in adjacent.items():
            try:
                validation = validate_collection_checkpoint(
                    checkpoint, allow_adjacent_hard_success=True
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
            entries.append(
                {
                    "object": object_name,
                    "training_root": str(training_root.resolve()),
                    "training_lineage": training_root.name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "checkpoint_iteration": iteration,
                    "checkpoint_mode": "adjacent_hard_success_lineage",
                    "success_iteration": success_iteration,
                    "distance_to_success_iteration": abs(iteration - success_iteration),
                    "nominal": str(nominal),
                    "nominal_sha256": provenance["nominal_dataset_sha256"],
                    "penetration_reward_weight": float(
                        provenance.get("penetration_reward_weight", -20.0)
                    ),
                    "instantaneous_residual_weight": float(
                        provenance.get("instantaneous_residual_weight", 0.0)
                    ),
                    "residual_integration": float(provenance.get("residual_integration", 0.03)),
                    "arm_action_scale_rad": float(provenance.get("arm_action_scale_rad", 0.12)),
                    "hand_action_scale_rad": float(provenance.get("hand_action_scale_rad", 0.24)),
                    # Historical primary runs predate this provenance field and
                    # were launched by the phase-2 bridge contract.
                    "residual_activation_phase": int(
                        provenance.get("residual_activation_phase", 2)
                    ),
                    "marker_mtime": marker.stat().st_mtime,
                }
            )

    # New formal runs persist the exact stochastic rollout and the policy
    # snapshot that existed before PPO updated it.  These checkpoints are
    # stronger evidence than a 100-iteration-adjacent model and are available
    # immediately, even while the containing training attempt is still live.
    seen_checkpoints = {Path(row["checkpoint"]).resolve() for row in entries}
    for exact_marker in object_root.glob("training*/hard_success_rollouts.jsonl"):
        training_root = exact_marker.parent
        if (training_root / "ABANDONED.json").is_file():
            continue
        provenance_path = training_root / "training_provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text())
            nominal = Path(provenance["nominal_dataset"]).resolve()
            nominal_hash = provenance["nominal_dataset_sha256"]
            if not nominal.is_file() or sha256_file(nominal) != nominal_hash:
                continue
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        for line in exact_marker.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                checkpoint = Path(row["checkpoint"]).resolve()
                success_iteration = int(row["iteration"])
                if checkpoint in seen_checkpoints or row.get("saved_before_ppo_update") is not True:
                    continue
                validation = validate_collection_checkpoint(
                    checkpoint, allow_exact_training_hard_success=True
                )
            except (
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            seen_checkpoints.add(checkpoint)
            entries.append(
                {
                    "object": object_name,
                    "training_root": str(training_root.resolve()),
                    "training_lineage": training_root.name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "checkpoint_iteration": success_iteration,
                    "checkpoint_mode": "exact_training_rollout_preupdate",
                    "success_iteration": success_iteration,
                    "distance_to_success_iteration": 0,
                    "nominal": str(nominal),
                    "nominal_sha256": nominal_hash,
                    "penetration_reward_weight": float(
                        provenance.get("penetration_reward_weight", -20.0)
                    ),
                    "instantaneous_residual_weight": float(
                        provenance.get("instantaneous_residual_weight", 0.0)
                    ),
                    "residual_integration": float(
                        provenance.get("residual_integration", 0.03)
                    ),
                    "arm_action_scale_rad": float(
                        provenance.get("arm_action_scale_rad", 0.12)
                    ),
                    "hand_action_scale_rad": float(
                        provenance.get("hand_action_scale_rad", 0.24)
                    ),
                    "residual_activation_phase": int(
                        provenance.get("residual_activation_phase", 2)
                    ),
                    "marker_mtime": exact_marker.stat().st_mtime,
                }
            )
    return sorted(
        entries,
        key=lambda row: (
            -float(row["marker_mtime"]),
            int(row["distance_to_success_iteration"]),
            int(row["checkpoint_iteration"]),
        ),
    )


def _embedded_samples(root: Path, object_name: str) -> list[Path]:
    return sorted((root / "accepted" / object_name).glob("sample_*/receipt.json"))


def _strict_pass(root: Path, object_name: str) -> bool:
    for receipt in (root / "accepted" / object_name).glob(
        "sample_*/strict_post_audit/receipt.json"
    ):
        try:
            if json.loads(receipt.read_text()).get("accepted") is True:
                return True
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return False


def _event(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "xhand_rl_discovered_hard_success_collection_v1",
        "time": time.time(),
        **payload,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()


def _crash_backoff(lock_path: Path, seconds: float) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            time.sleep(seconds)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _wait_for_audits(samples: list[Path], poll_seconds: float) -> None:
    for embedded_receipt in samples:
        strict_receipt = embedded_receipt.parent / "strict_post_audit" / "receipt.json"
        while not strict_receipt.is_file():
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-lock", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--kit-cooldown-seconds", type=float, default=75.0)
    parser.add_argument("--crash-backoff-seconds", type=float, default=180.0)
    parser.add_argument("--base-seed", type=int, default=2001)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    validate_root(args.root, args.manifest, manifest)
    if set(manifest["objects"]) != set(LOCKED_OBJECTS):
        raise RuntimeError("manifest object set differs from the locked six")
    for path in (args.repo, args.isaac_python, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.num_envs <= 0 or args.max_steps <= 0 or args.poll_seconds <= 0.0:
        raise ValueError("collection sizes and polling interval must be positive")

    logs_root = args.root / "logs" / "discovered_hard_success_collectors"
    events_path = logs_root / "attempts.jsonl"
    logs_root.mkdir(parents=True, exist_ok=True)
    attempt_count = 0
    object_attempt_counts = {name: 0 for name in LOCKED_OBJECTS}
    if events_path.is_file():
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if event.get("stage") != "collection_attempt":
                continue
            attempt_count += 1
            object_name = event.get("object")
            if object_name in object_attempt_counts:
                object_attempt_counts[object_name] += 1

    while True:
        if all(_strict_pass(args.root, name) for name in LOCKED_OBJECTS):
            _event(events_path, stage="all_six_strict_complete")
            return
        launched = False
        for object_name in LOCKED_OBJECTS:
            if _strict_pass(args.root, object_name):
                continue
            entries = discover_hard_success_lineages(args.root, object_name)
            if not entries:
                continue
            object_attempt = object_attempt_counts[object_name]
            combinations = [
                (entry, std_scale)
                for entry in entries
                for std_scale in POLICY_STD_SCALES
            ]
            entry, policy_std_scale = combinations[object_attempt % len(combinations)]
            before_samples = _embedded_samples(args.root, object_name)
            target = len(before_samples) + 1
            seed = args.base_seed + attempt_count
            checkpoint = Path(entry["checkpoint"])
            attempt_id = (
                f"discovered-{object_name}-{entry['training_lineage']}-"
                f"{checkpoint.stem}-std-{policy_std_scale:g}-seed-{seed}-target-{target}"
            )
            command = [
                str(args.isaac_python),
                "-m",
                "migration_4090.xhand_rl_embedded.collect",
                "--manifest",
                str(args.manifest.resolve()),
                "--object",
                object_name,
                "--nominal-dataset",
                entry["nominal"],
                "--checkpoint",
                entry["checkpoint"],
                "--root",
                str(args.root.resolve()),
                "--target",
                str(target),
                "--num-envs",
                str(args.num_envs),
                "--max-steps",
                str(args.max_steps),
                "--seed",
                str(seed),
                "--policy-std-scale",
                str(policy_std_scale),
                "--device",
                DEVICES[object_name],
                "--headless",
                "--allow-partial",
                "--attempt-id",
                attempt_id,
                "--collection-lock",
                str(args.collection_lock.resolve()),
                "--kit-teardown-cooldown-s",
                str(args.kit_cooldown_seconds),
                "--kit-portable-root",
                str(
                    Path("/tmp/xhand_rl_embedded_kit/v5r_discovered_hard_success")
                    / object_name
                    / attempt_id
                ),
                "--penetration-reward-weight",
                str(entry["penetration_reward_weight"]),
                "--instantaneous-residual-weight",
                str(entry["instantaneous_residual_weight"]),
                "--residual-integration",
                str(entry["residual_integration"]),
                "--arm-action-scale-rad",
                str(entry["arm_action_scale_rad"]),
                "--hand-action-scale-rad",
                str(entry["hand_action_scale_rad"]),
                "--residual-activation-phase",
                str(entry["residual_activation_phase"]),
            ]
            if entry.get("checkpoint_mode") == "exact_training_rollout_preupdate":
                command.append("--allow-exact-training-hard-success-checkpoint")
            else:
                command.append("--allow-adjacent-hard-success-checkpoint")
            launched = True
            log_path = logs_root / f"{attempt_id}.log"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(args.repo) + os.pathsep + environment.get(
                "PYTHONPATH", ""
            )
            environment["PYTHONUNBUFFERED"] = "1"
            started = time.monotonic()
            with log_path.open("a") as log:
                code = subprocess.run(
                    command,
                    cwd=args.repo,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            after_samples = _embedded_samples(args.root, object_name)
            attempt_count += 1
            object_attempt_counts[object_name] += 1
            _event(
                events_path,
                stage="collection_attempt",
                object=object_name,
                attempt_id=attempt_id,
                seed=seed,
                policy_std_scale=policy_std_scale,
                target=target,
                checkpoint=entry["checkpoint"],
                checkpoint_sha256=entry["checkpoint_sha256"],
                nominal=entry["nominal"],
                nominal_sha256=entry["nominal_sha256"],
                hard_success_iteration=entry["success_iteration"],
                return_code=code,
                duration_s=time.monotonic() - started,
                embedded_before=len(before_samples),
                embedded_after=len(after_samples),
            )
            if code in (134, 139, -6, -11):
                _crash_backoff(args.collection_lock, args.crash_backoff_seconds)
            new_samples = sorted(set(after_samples) - set(before_samples))
            if new_samples:
                _wait_for_audits(new_samples, args.poll_seconds)
        if not launched:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
