"""Select a collection checkpoint from persisted PPO hard-success evidence."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .contracts import checkpoint_iteration, latest_policy_checkpoint, sha256_file


def validate_collection_checkpoint(
    checkpoint: Path,
    *,
    allow_adjacent_hard_success: bool = False,
    allow_exact_training_hard_success: bool = False,
) -> dict[str, Any]:
    """Validate a collection checkpoint against immutable training evidence.

    The normal production path uses the checkpoint selected in
    ``TRAINING_COMPLETE.json``.  A hard-success rollout can occur between the
    100-iteration RSL-RL save points, however, and the nearest checkpoint after
    it may already have drifted away.  Targeted resampling may therefore use
    only the immediately preceding or following saved checkpoint around that
    persisted success iteration.  No arbitrary checkpoint from the directory
    is accepted, and the selected marker itself must still pass its hash check.
    """

    checkpoint = checkpoint.resolve()
    # Formal PPO and curriculum PPO use the same immutable pre-update success
    # contract, but the curriculum marker has a distinct filename/schema.
    # Allowing the latter here is a targeted seed-probe path only; the normal
    # collector still requires TRAINING_COMPLETE.json.
    exact_markers = [checkpoint.parent / "hard_success_rollouts.jsonl"]
    curriculum_marker = checkpoint.parent / "curriculum_hard_success_rollouts.jsonl"
    if curriculum_marker.is_file():
        exact_markers.append(curriculum_marker)
    if allow_exact_training_hard_success and any(path.is_file() for path in exact_markers):
        checkpoint_hash = sha256_file(checkpoint)
        matching = []
        for exact_marker in exact_markers:
            if not exact_marker.is_file():
                continue
            for line in exact_marker.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    persisted = Path(row["checkpoint"]).resolve()
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    persisted == checkpoint
                    and row.get("checkpoint_sha256") == checkpoint_hash
                    and row.get("saved_before_ppo_update") is True
                ):
                    matching.append((row, exact_marker))
        if matching:
            row, marker_path = max(
                matching,
                key=lambda item: (int(item[0]["iteration"]), float(item[0].get("time", 0.0))),
            )
            return {
                "mode": "exact_preupdate_rollout_seed",
                "checkpoint_sha256": checkpoint_hash,
                "training_selected_checkpoint": str(checkpoint),
                "training_selected_checkpoint_sha256": checkpoint_hash,
                "training_hard_success_iteration": int(row["iteration"]),
                "training_hard_success_serial": int(row["success_serial"]),
                "exact_success_marker": str(marker_path.resolve()),
            }

    marker = checkpoint.parent / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError("collector requires completed v2 training")
    training = json.loads(marker.read_text())
    selected = Path(training["checkpoint"]).resolve()
    if selected.parent != checkpoint.parent or checkpoint.parent != marker.parent.resolve():
        raise RuntimeError("collector checkpoint is outside the completed training lineage")
    if not selected.is_file() or sha256_file(selected) != training.get("checkpoint_sha256"):
        raise RuntimeError("selected training checkpoint is missing or changed")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint == selected and checkpoint_hash == training["checkpoint_sha256"]:
        return {
            "mode": "training_complete_selected",
            "checkpoint_sha256": checkpoint_hash,
            "training_selected_checkpoint": str(selected),
            "training_selected_checkpoint_sha256": training["checkpoint_sha256"],
            "training_hard_success_iteration": training.get("checkpoint_selection", {}).get(
                "training_hard_success_iteration"
            ),
        }
    if not allow_adjacent_hard_success:
        raise RuntimeError("collector checkpoint differs from training completion marker")
    selection = training.get("checkpoint_selection", {})
    if selection.get("training_hard_success_observed") is not True:
        raise RuntimeError("adjacent collection requires persisted hard-success evidence")
    success_iteration = int(selection["training_hard_success_iteration"])
    checkpoints = sorted(
        (checkpoint_iteration(path), path.resolve())
        for path in checkpoint.parent.glob("model_*.pt")
        if path.is_file()
    )
    preceding = [row for row in checkpoints if row[0] <= success_iteration]
    following = [row for row in checkpoints if row[0] >= success_iteration]
    adjacent = set()
    if preceding:
        adjacent.add(preceding[-1][1])
    if following:
        adjacent.add(following[0][1])
    if checkpoint not in adjacent:
        raise RuntimeError("checkpoint is not adjacent to the persisted hard-success iteration")
    return {
        "mode": "adjacent_hard_success_lineage",
        "checkpoint_sha256": checkpoint_hash,
        "training_selected_checkpoint": str(selected),
        "training_selected_checkpoint_sha256": training["checkpoint_sha256"],
        "training_hard_success_iteration": success_iteration,
        "adjacent_checkpoint_iterations": sorted(checkpoint_iteration(path) for path in adjacent),
    }


def select_collection_checkpoint(directory: Path) -> tuple[Path, dict[str, Any]]:
    checkpoints = sorted(
        (
            (checkpoint_iteration(path), path)
            for path in directory.glob("model_*.pt")
            if path.is_file()
        ),
        key=lambda row: row[0],
    )
    if not checkpoints:
        raise RuntimeError("no RSL-RL checkpoints are available for collection")
    evidence: list[tuple[float, int, float, str]] = []
    for event_path in directory.glob("events.out.tfevents.*"):
        try:
            accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            accumulator.Reload()
            for value in accumulator.Scalars("embedded/terminal_success_fraction"):
                if math.isfinite(value.value) and value.value > 0.0:
                    evidence.append(
                        (float(value.value), int(value.step), float(value.wall_time), str(event_path.resolve()))
                    )
        except Exception:
            continue
    latest = latest_policy_checkpoint(directory)
    if latest is None:
        raise RuntimeError("latest RSL-RL checkpoint disappeared during selection")
    if not evidence:
        return latest, {
            "strategy": "latest_without_training_hard_success",
            "training_hard_success_observed": False,
            "selected_iteration": checkpoint_iteration(latest),
        }
    success_fraction, success_step, wall_time, event_file = max(
        evidence, key=lambda row: (row[0], row[1], row[2])
    )
    following = [row for row in checkpoints if row[0] >= success_step]
    selected_iteration, selected = following[0] if following else checkpoints[-1]
    return selected, {
        "strategy": "nearest_checkpoint_at_or_after_max_training_hard_success",
        "training_hard_success_observed": True,
        "training_hard_success_fraction": success_fraction,
        "training_hard_success_iteration": success_step,
        "training_hard_success_wall_time": wall_time,
        "training_hard_success_event_file": event_file,
        "selected_iteration": selected_iteration,
    }
