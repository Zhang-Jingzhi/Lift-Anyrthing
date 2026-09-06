#!/usr/bin/env python3
"""Build a read-only, monotonic-step TensorBoard view of live PPO events.

The RSL-RL event stream can contain resumed points whose step is lower than a
previous point.  TensorBoard is still useful for the raw event file, but that
ordering makes a multi-object strict-gate comparison hard to read.  This
utility writes only under the visualization directory and never changes the
training event files or acceptance receipts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


PREFIXES = ("contact", "lifted", "clear", "stable_force_closure", "disturbed", "ablated")
TAGS = (
    "Train/mean_reward",
    "reward/total_scaled",
    "reward/hard_gate_frontier",
    "reward/approach_proximity",
    "reward/bilateral_contact",
    "reward/contact_continuity",
    "reward/lift_height",
    "reward/gravity_margin",
    "reward/force_closure",
    "reward/penetration",
    "reward/penetration_clear",
    "embedded/gate_bilateral_contact_continuity_fraction",
    "embedded/gate_arm_joint_lift_fraction",
    "embedded/gate_lift_height_fraction",
    "embedded/gate_gravity_hold_fraction",
    "embedded/gate_translation_disturbance_fraction",
    "embedded/gate_rotation_disturbance_fraction",
    "embedded/gate_physx_penetration_fraction",
    "embedded/gate_formal_force_closure_fraction",
    "embedded/gate_single_hand_ablations_fraction",
    "embedded/terminal_success_fraction",
    *(f"embedded/hard_prefix_{name}_fraction" for name in PREFIXES),
)


def _read(path: Path) -> EventAccumulator | None:
    try:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        return acc
    except Exception:
        return None


def _emit(writer: EventFileWriter, tag: str, rows: list[object]) -> None:
    """Emit one scalar on a monotonic PPO-step axis.

    A global counter for all tags shifts each line into a different x-axis
    range.  Preserve each source step per tag and only bump resumed/out-of-
    order points by one.
    """
    previous_step = -1
    for row in sorted(rows, key=lambda item: item.wall_time):
        source_step = int(getattr(row, "step", 0))
        step = max(source_step, previous_step + 1)
        writer.add_event(
            Event(
                wall_time=float(row.wall_time),
                step=step,
                summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(row.value))]),
            )
        )
        previous_step = step


def curate_object(object_name: str, source: Path, target: Path) -> bool:
    event_files = sorted(source.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return False
    event_file = event_files[-1]
    signature = {
        "format": 2,
        "source": str(event_file.resolve()),
        "size": event_file.stat().st_size,
        "mtime_ns": event_file.stat().st_mtime_ns,
    }
    meta = target / "source.json"
    if meta.is_file():
        try:
            if json.loads(meta.read_text()) == signature:
                return True
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("events.out.tfevents.*"):
        try:
            old.unlink()
        except OSError:
            pass
    acc = _read(event_file)
    if acc is None:
        return False
    available = set(acc.Tags().get("scalars", []))
    writer = EventFileWriter(str(target))
    try:
        for tag in TAGS:
            if tag in available:
                _emit(writer, tag, acc.Scalars(tag))
        # The ordered prefix is the operator-facing metric.  Reconstruct it
        # from the latest prefix values at each write-time sample.
        prefix_rows = {
            name: sorted(
                acc.Scalars(f"embedded/hard_prefix_{name}_fraction"),
                key=lambda item: item.wall_time,
            )
            for name in PREFIXES
            if f"embedded/hard_prefix_{name}_fraction" in available
        }
        times = sorted({row.wall_time for rows in prefix_rows.values() for row in rows})
        indices = {name: 0 for name in prefix_rows}
        latest = {name: 0.0 for name in PREFIXES}
        level_rows = []
        coverage_rows = []
        for wall_time in times:
            for name, rows in prefix_rows.items():
                while indices[name] < len(rows) and rows[indices[name]].wall_time <= wall_time:
                    latest[name] = float(rows[indices[name]].value)
                    indices[name] += 1
            level = 0
            for name in PREFIXES:
                if latest[name] > 0.0:
                    level += 1
                else:
                    break
            source_steps = [
                int(rows[max(indices[name] - 1, 0)].step)
                for name, rows in prefix_rows.items()
                if indices[name] > 0
            ]
            source_step = max(source_steps, default=0)
            level_rows.append((wall_time, source_step, level))
            coverage_rows.append(
                (
                    wall_time,
                    source_step,
                    min(
                        (latest[name] for name in PREFIXES[:level]),
                        default=0.0,
                    ),
                )
            )
        # Several env batches can write the same PPO step.  Keep the latest
        # value for that source step instead of incrementing a synthetic
        # counter into a second, misleading time axis.
        level_by_step = {}
        for wall_time, source_step, level in level_rows:
            level_by_step[int(source_step)] = (wall_time, level)
        for source_step, (wall_time, level) in sorted(level_by_step.items()):
            writer.add_event(
                Event(
                    wall_time=float(wall_time),
                    step=int(source_step),
                    summary=Summary(
                        value=[Summary.Value(tag="embedded/current_prefix_level", simple_value=float(level))]
                    ),
                )
            )
        coverage_by_step = {}
        for wall_time, source_step, coverage in coverage_rows:
            coverage_by_step[int(source_step)] = (wall_time, coverage)
        for source_step, (wall_time, coverage) in sorted(coverage_by_step.items()):
            writer.add_event(
                Event(
                    wall_time=float(wall_time),
                    step=int(source_step),
                    summary=Summary(
                        value=[
                            Summary.Value(
                                tag="embedded/current_prefix_coverage",
                                simple_value=float(coverage),
                            )
                        ]
                    ),
                )
            )
        writer.flush()
    finally:
        writer.close()
    meta.write_text(json.dumps(signature, indent=2) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        snapshot = json.loads(args.snapshot.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    for object_name, worker in snapshot.get("workers", {}).items():
        if worker.get("nominal_lineage") not in {"meshaware_v3", "meshaware_v4", "meshaware_v4b"}:
            continue
        if worker.get("stage") == "stopped_waiting_reconcile" and not worker.get("process_live"):
            continue
        event_file = worker.get("training_metrics", {}).get("event_file")
        if not event_file:
            continue
        curate_object(object_name, Path(event_file).resolve().parent, args.output / object_name)
    (args.output / "updated_at.txt").parent.mkdir(parents=True, exist_ok=True)
    (args.output / "updated_at.txt").write_text(f"{time.time():.6f}\n")


if __name__ == "__main__":
    main()
