#!/usr/bin/env python3
"""Export merged TensorBoard curves for embedded XHand PPO training."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_OBJECTS = (
    "sphere",
    "sphere_small",
    "cracker_large",
    "cracker",
    "pyramid",
    "cube",
)

SUMMARY_TAGS = (
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_noise_std",
    "Train/mean_reward",
    "reward/total_scaled",
    "reward/bilateral_contact",
    "reward/lift_height",
    "reward/hold_stability",
    "reward/disturbance_recovery",
    "reward/penetration",
    "reward/penetration_clear",
    "reward/force_closure",
    "reward/hard_gate_frontier",
    "reward/disturbance_direction",
    "reward/ablation_contract",
    "embedded/terminal_success_fraction",
    "embedded/force_closure_evaluated_fraction",
    "embedded/force_closure_reward_ready_fraction",
    "embedded/hard_prefix_stable_force_closure_fraction",
    "embedded/hard_prefix_disturbed_fraction",
    "embedded/hard_prefix_ablated_fraction",
    "embedded/hard_disturbance_direction_pass_fraction",
    "embedded/gate_bilateral_contact_continuity_fraction",
    "embedded/gate_lift_height_fraction",
    "embedded/gate_gravity_hold_fraction",
    "embedded/gate_physx_penetration_fraction",
    "embedded/gate_formal_force_closure_fraction",
    "embedded/gate_translation_disturbance_fraction",
    "embedded/gate_rotation_disturbance_fraction",
    "Perf/total_fps",
)


def load_merged_scalars(training_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Merge restarted TensorBoard event files, preferring the newest event per step."""
    merged: dict[str, dict[int, tuple[float, float]]] = {}
    event_paths = sorted(training_dir.glob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError(f"no TensorBoard events in {training_dir}")
    for event_path in event_paths:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            by_step = merged.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                previous = by_step.get(event.step)
                if previous is None or event.wall_time >= previous[0]:
                    by_step[event.step] = (event.wall_time, float(event.value))
    return {
        tag: [(step, value, wall_time) for step, (wall_time, value) in sorted(by_step.items())]
        for tag, by_step in merged.items()
    }


def load_event_scalars(event_path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Load exactly one retry event file without merging older lineages."""
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(event.step, float(event.value), event.wall_time) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _canonical_nominal_root(root: Path) -> Path | None:
    """Read the immutable nominal root expected by the active lineage."""
    marker = root / "EMBEDDED_RL_PIPELINE_ROOT.json"
    try:
        payload = json.loads(marker.read_text())
        value = payload.get("nominal_root")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return Path(str(value)).resolve() if value else None


def snapshot_event_paths(
    snapshot: Path,
    objects: tuple[str, ...],
    *,
    canonical_nominal_root: Path | None = None,
) -> dict[str, Path]:
    """Resolve current PPO events, rejecting stale nominal lineages.

    A supervisor handoff can leave the status JSON pointing at an old worker
    while the immutable root marker already targets a new nominal dataset.
    For current-only plots, those old events are not comparable and must be
    omitted until the new worker has emitted its first TensorBoard event.
    """
    payload = json.loads(snapshot.read_text())
    paths: dict[str, Path] = {}
    workers = payload.get("workers", {})
    for object_name in objects:
        worker = workers.get(object_name, {})
        if canonical_nominal_root is not None:
            nominal = worker.get("nominal_dataset")
            if not nominal:
                continue
            nominal_path = Path(str(nominal)).resolve()
            try:
                nominal_path.relative_to(canonical_nominal_root)
            except ValueError:
                # Explicitly exclude v3/v4b/v4c-final2 workers during an
                # append-only handoff.  Their reward scales and action
                # semantics are not evidence about the active v4c lineage.
                continue
        event_file = worker.get("training_metrics", {}).get("event_file")
        if event_file:
            path = Path(event_file).resolve()
            if path.is_file():
                paths[object_name] = path
    missing = [name for name in objects if name not in paths]
    if missing and canonical_nominal_root is None:
        raise FileNotFoundError(
            "live snapshot has no readable current PPO event for: " + ", ".join(missing)
        )
    return paths


def load_current_attempt_scalars(
    training_dir: Path,
) -> tuple[Path, dict[str, list[tuple[int, float, float]]]]:
    event_paths = list(training_dir.glob("events.out.tfevents.*"))
    if not event_paths:
        raise FileNotFoundError(f"no TensorBoard events in {training_dir}")
    event_path = max(event_paths, key=lambda path: path.stat().st_mtime)
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {
        tag: [(event.step, float(event.value), event.wall_time) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }
    return event_path, scalars


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    width = min(max(window, 1), values.size)
    kernel = np.ones(width, dtype=np.float64) / width
    averaged = np.convolve(values, kernel, mode="valid")
    prefix = np.full(width - 1, np.nan, dtype=np.float64)
    return np.concatenate((prefix, averaged))


def rebase_attempt_steps(
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
) -> dict[str, dict[str, list[tuple[int, float, float]]]]:
    """Rebase every selected retry to its own first logged PPO update.

    A formal retry can resume from a checkpoint at iteration 98, 275, or
    another non-zero step.  Plotting those raw step numbers beside a fresh
    retry makes a shorter run look artificially more advanced.  Keep the
    event values and wall times intact; only the display x-coordinate is
    shifted.  The raw values remain available in ``latest_metrics.json``.
    """

    rebased: dict[str, dict[str, list[tuple[int, float, float]]]] = {}
    for object_name, tags in data.items():
        # Some RSL-RL writers emit auxiliary scalars (for example a fixed
        # policy-noise value) at step 0 after a resumed checkpoint.  Those are
        # not PPO iteration anchors and made retry-relative plots retain raw
        # steps such as 700 or 160.  Rebase from a primary training curve and
        # only fall back to all tags when no primary curve exists.
        anchor_events = tags.get("Loss/value_function") or tags.get("Train/mean_reward")
        first_step = min(
            (event[0] for event in anchor_events),
            default=min(
                (event[0] for events in tags.values() for event in events),
                default=0,
            ),
        )
        rebased[object_name] = {
            tag: [(step - first_step, value, wall_time) for step, value, wall_time in events]
            for tag, events in tags.items()
        }
    return rebased


def annotate_lineage_labels(
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
    snapshot_payload: dict[str, object] | None,
) -> dict[str, dict[str, list[tuple[int, float, float]]]]:
    """Add the live lineage to plot labels so stale retries cannot look valid.

    The status snapshot can temporarily contain a worker that is still
    finishing an old v3 process while the root contract already targets v4b.
    Keeping the data visible is useful for diagnosis, but the legend must make
    that mismatch explicit instead of presenting it as current progress.
    """
    if not snapshot_payload:
        return data
    workers = snapshot_payload.get("workers", {})
    annotated: dict[str, dict[str, list[tuple[int, float, float]]]] = {}
    for object_name, tags in data.items():
        worker = workers.get(object_name, {}) if isinstance(workers, dict) else {}
        lineage = worker.get("nominal_lineage") or "unknown"
        mismatch = bool(worker.get("nominal_mismatch"))
        suffix = f" [{lineage}{' · MISMATCH' if mismatch else ''}]"
        annotated[f"{object_name}{suffix}"] = tags
    return annotated


def plot_tag(
    axis: plt.Axes,
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
    tag: str,
    title: str,
    smoothing: int,
    *,
    log_scale: bool = False,
    x_label: str = "PPO iteration",
) -> None:
    has_positive = False
    for object_name, tags in data.items():
        events = tags.get(tag, [])
        if not events:
            continue
        steps = np.asarray([event[0] for event in events], dtype=np.int64)
        values = np.asarray([event[1] for event in events], dtype=np.float64)
        has_positive = has_positive or bool(np.any(values > 0.0))
        (line,) = axis.plot(steps, values, alpha=0.14, linewidth=0.7)
        axis.plot(
            steps,
            rolling_mean(values, smoothing),
            color=line.get_color(),
            linewidth=1.8,
            label=object_name,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.grid(alpha=0.22)
    if tag.startswith("embedded/") and "fraction" in tag:
        # Fractions are gate coverage in [0, 1].  A log axis makes a rare
        # one-trajectory event look visually comparable to a stable 50% basin
        # and hides the fact that zeros dominate the current retries.
        axis.set_ylim(0.0, 1.0)
    if log_scale and has_positive:
        axis.set_yscale("log")


def save_figure(
    output_path: Path,
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
    panels: tuple[tuple[str, str, bool], ...],
    smoothing: int,
    title: str,
    *,
    nrows: int = 2,
    ncols: int = 2,
    x_label: str = "PPO iteration",
) -> None:
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, (tag, panel_title, log_scale) in zip(axes.flat, panels, strict=True):
        plot_tag(
            axis,
            data,
            tag,
            panel_title,
            smoothing,
            log_scale=log_scale,
            x_label=x_label,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def save_per_object_figure(
    output_path: Path,
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
    objects: tuple[str, ...],
    panels: tuple[tuple[str, str, bool], ...],
    smoothing: int,
    title: str,
    *,
    fraction_limits: bool = False,
    symlog_linthresh: float | None = None,
) -> None:
    """Write a small-multiple diagnostic with one row per object.

    Six overlaid lines can hide a small object's behavior behind a much larger
    reward scale.  The per-object view keeps the retry-relative x-axis while
    exposing the latest value and the historical maximum for each strict
    fraction.  It is diagnostic only and does not change training or receipt
    acceptance.
    """

    nrows = max(len(objects), 1)
    ncols = len(panels)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, object_name in enumerate(objects):
        tags = data.get(object_name, {})
        for column_index, (tag, panel_title, log_scale) in enumerate(panels):
            axis = axes[row_index, column_index]
            events = tags.get(tag, [])
            values = np.asarray([event[1] for event in events], dtype=np.float64)
            if events:
                steps = np.asarray([event[0] for event in events], dtype=np.int64)
                axis.plot(steps, values, color="0.65", alpha=0.28, linewidth=0.7)
                axis.plot(
                    steps,
                    rolling_mean(values, smoothing),
                    color="tab:blue",
                    linewidth=1.6,
                )
                axis.scatter([steps[-1]], [values[-1]], color="tab:red", s=12, zorder=4)
                if tag.startswith("embedded/") and "fraction" in tag:
                    peak = float(np.nanmax(values))
                    axis.axhline(peak, color="tab:orange", alpha=0.45, linewidth=0.8)
                    axis.text(
                        0.99,
                        0.92,
                        f"latest={values[-1]:.3g}  max={peak:.3g}",
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7,
                    )
            axis.set_title(panel_title if row_index == 0 else object_name)
            axis.set_xlabel("updates since retry start")
            axis.grid(alpha=0.22)
            if fraction_limits:
                axis.set_ylim(0.0, 1.0)
            if symlog_linthresh is not None and not log_scale:
                axis.set_yscale("symlog", linthresh=symlog_linthresh)
            elif log_scale and values.size and np.any(values > 0.0):
                axis.set_yscale("log")
            if column_index == 0:
                axis.set_ylabel(object_name)
    figure.suptitle(title, fontsize=15)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def latest_summary(
    data: dict[str, dict[str, list[tuple[int, float, float]]]],
    current_attempts: dict[str, tuple[Path, dict[str, list[tuple[int, float, float]]]]],
    smoothing: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for object_name, tags in data.items():
        current_path, current_tags = current_attempts[object_name]
        iteration_events = tags.get("Loss/value_function", [])
        current_iteration_events = current_tags.get("Loss/value_function", [])
        row: dict[str, object] = {
            "object": object_name,
            "latest_iteration": iteration_events[-1][0] if iteration_events else None,
            "current_attempt_event_file": str(current_path.resolve()),
            "current_attempt_start_iteration": (
                current_iteration_events[0][0] if current_iteration_events else None
            ),
            "current_attempt_latest_iteration": (
                current_iteration_events[-1][0] if current_iteration_events else None
            ),
        }
        if current_iteration_events:
            row["current_attempt_updates"] = max(
                current_iteration_events[-1][0] - current_iteration_events[0][0],
                0,
            )
        else:
            row["current_attempt_updates"] = None
        for tag in SUMMARY_TAGS:
            events = tags.get(tag, [])
            key = tag.replace("/", "__")
            if not events:
                row[key] = None
                row[f"{key}__mean_last_{smoothing}"] = None
                continue
            values = np.asarray([event[1] for event in events[-smoothing:]], dtype=np.float64)
            row[key] = float(events[-1][1])
            row[f"{key}__mean_last_{smoothing}"] = float(values.mean())
        hard_events = tags.get("embedded/terminal_success_fraction", [])
        row["lineage_hard_success_iterations"] = sum(value > 0.0 for _, value, _ in hard_events)
        row["lineage_max_hard_success_fraction"] = max(
            (value for _, value, _ in hard_events), default=0.0
        )
        current_hard_events = current_tags.get("embedded/terminal_success_fraction", [])
        row["current_attempt_hard_success_iterations"] = sum(
            value > 0.0 for _, value, _ in current_hard_events
        )
        row["current_attempt_max_hard_success_fraction"] = max(
            (value for _, value, _ in current_hard_events), default=0.0
        )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS))
    parser.add_argument("--smoothing", type=int, default=25)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="status_snapshot.json; selects the current retry event for every object",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="plot only the selected current retry, never the merged historical lineage",
    )
    parser.add_argument(
        "--absolute-steps",
        action="store_true",
        help="keep checkpoint-relative/raw TensorBoard steps on the x-axis",
    )
    parser.add_argument(
        "--per-object",
        action="store_true",
        help="also write small-multiple PPO/physical diagnostics with one row per object",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothing < 1:
        raise ValueError("smoothing must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or args.root / "visualizations" / f"training_curves_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    snapshot_payload = json.loads(args.snapshot.read_text()) if args.snapshot else None
    canonical_nominal_root = (
        _canonical_nominal_root(args.root) if args.current_only else None
    )
    selected_paths = (
        snapshot_event_paths(
            args.snapshot,
            tuple(args.objects),
            canonical_nominal_root=canonical_nominal_root,
        )
        if args.snapshot
        else {}
    )
    if args.current_only and not selected_paths:
        raise ValueError("--current-only requires at least one readable event in the canonical lineage")
    plot_objects = tuple(selected_paths) if args.current_only else tuple(args.objects)
    if selected_paths:
        data = {
            object_name: load_event_scalars(selected_paths[object_name])
            if args.current_only
            else load_merged_scalars(args.root / "runs" / object_name / "training")
            for object_name in plot_objects
        }
        current_attempts = {
            object_name: (selected_paths[object_name], load_event_scalars(selected_paths[object_name]))
            for object_name in plot_objects
        }
    else:
        data = {
            object_name: load_merged_scalars(args.root / "runs" / object_name / "training")
            for object_name in plot_objects
        }
        current_attempts = {
            object_name: load_current_attempt_scalars(
                args.root / "runs" / object_name / "training"
            )
            for object_name in plot_objects
        }
    # Current-retry plots are comparisons of progress within each retry, not
    # comparisons of checkpoint numbering.  Rebase only the display data;
    # latest_summary still reports the raw event steps for provenance.
    relative_steps = bool(args.current_only and not args.absolute_steps)
    plot_data = rebase_attempt_steps(data) if relative_steps else data
    plot_data_for_display = annotate_lineage_labels(plot_data, snapshot_payload)
    x_label = "Updates since current retry start" if relative_steps else "PPO iteration"
    title_suffix = " · current retry only" if args.current_only else " · merged lineage"
    save_figure(
        output / "ppo_training_curves.png",
        plot_data_for_display,
        (
            ("Loss/value_function", "Value-function loss", True),
            ("Loss/surrogate", "PPO surrogate loss", False),
            ("Loss/entropy", "Policy entropy", False),
            ("Policy/mean_noise_std", "Action noise std", False),
            ("Train/mean_reward", "Mean episode reward", False),
            ("reward/total_scaled", "Scaled training reward", False),
            ("reward/force_closure", "Force-closure shaping reward", False),
            ("reward/hard_gate_frontier", "Strict gate-frontier reward", False),
        ),
        args.smoothing,
        "XHand embedded-physics PPO training" + title_suffix,
        nrows=2,
        ncols=4,
        x_label=x_label,
    )
    if args.per_object:
        save_per_object_figure(
            output / "ppo_training_curves_per_object.png",
            plot_data,
            tuple(plot_objects),
            (
                ("Train/mean_reward", "Mean episode reward", False),
                ("reward/total_scaled", "Scaled training reward", False),
                ("Loss/value_function", "Value-function loss", True),
                ("Loss/surrogate", "PPO surrogate loss", False),
            ),
            args.smoothing,
            "XHand PPO diagnostics by object" + title_suffix,
            symlog_linthresh=0.01,
        )
    save_figure(
        output / "physical_training_curves.png",
        plot_data_for_display,
        (
            # All physical fractions use a linear [0, 1] scale.  This keeps a
            # rare one-trajectory event visibly distinct from a stable basin
            # and makes the first failed ordered prefix easy to compare.
            ("embedded/hard_prefix_contact_fraction", "Strict prefix: bilateral contact", False),
            ("embedded/hard_prefix_lifted_fraction", "Strict prefix: lifted", False),
            ("embedded/hard_prefix_clear_fraction", "Strict prefix: penetration-clear", False),
            (
                "embedded/hard_prefix_stable_force_closure_fraction",
                "Strict prefix: stable force closure",
                False,
            ),
            ("embedded/terminal_success_fraction", "Strict terminal success fraction", False),
            (
                "embedded/force_closure_evaluated_fraction",
                "Force-closure audit evaluated",
                False,
            ),
            (
                "embedded/gate_bilateral_contact_continuity_fraction",
                "Gate: bilateral contact continuity",
                False,
            ),
            ("embedded/gate_lift_height_fraction", "Gate: lift height", False),
            ("embedded/gate_gravity_hold_fraction", "Gate: gravity hold", False),
            (
                "embedded/gate_formal_force_closure_fraction",
                "Gate: formal force closure",
                False,
            ),
            (
                "embedded/gate_physx_penetration_fraction",
                "Gate: PhysX penetration clear",
                False,
            ),
            (
                "embedded/gate_translation_disturbance_fraction",
                "Gate: translation disturbance",
                False,
            ),
            (
                "embedded/gate_rotation_disturbance_fraction",
                "Gate: rotation disturbance",
                False,
            ),
            (
                "embedded/hard_prefix_disturbed_fraction",
                "Strict prefix: 12-direction disturbance",
                False,
            ),
            (
                "embedded/hard_prefix_ablated_fraction",
                "Strict prefix: single-hand ablations",
                False,
            ),
        ),
        args.smoothing,
        "Embedded physical signals and throughput" + title_suffix,
        nrows=5,
        ncols=3,
        x_label=x_label,
    )
    if args.per_object:
        save_per_object_figure(
            output / "physical_training_curves_per_object.png",
            plot_data,
            tuple(plot_objects),
            (
                (
                    "embedded/hard_prefix_contact_fraction",
                    "Strict prefix: bilateral contact",
                    False,
                ),
                ("embedded/hard_prefix_lifted_fraction", "Strict prefix: lifted", False),
                ("embedded/hard_prefix_clear_fraction", "Strict prefix: clear", False),
                (
                    "embedded/hard_prefix_stable_force_closure_fraction",
                    "Strict prefix: stable force closure",
                    False,
                ),
                (
                    "embedded/gate_formal_force_closure_fraction",
                    "Gate: formal force closure",
                    False,
                ),
                ("embedded/terminal_success_fraction", "Terminal success", False),
            ),
            args.smoothing,
            "Same-trajectory strict progress by object" + title_suffix,
            fraction_limits=True,
        )
    rows = latest_summary(data, current_attempts, args.smoothing)
    (output / "latest_metrics.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "root": str(args.root.resolve()),
                "smoothing_window": args.smoothing,
                "x_axis_mode": "retry_relative" if relative_steps else "raw_tensorboard_steps",
                "canonical_nominal_root": (
                    str(canonical_nominal_root) if canonical_nominal_root else None
                ),
                "omitted_objects": [
                    object_name for object_name in args.objects if object_name not in plot_objects
                ],
                "objects": rows,
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "latest_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
