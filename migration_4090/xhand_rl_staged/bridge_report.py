#!/usr/bin/env python3
"""Render Stage-2.5 TensorBoard curves and fixed-evaluation comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from migration_4090.xhand_rl_embedded.report_training_curves import (
    load_merged_scalars,
)
from migration_4090.xhand_rl_staged.micro_lift import summarize_micro_lift_reports


CURVE_TAGS = (
    "Loss/value_function",
    "Loss/surrogate",
    "bridge/controlled_fraction",
    "bridge/stable_ready_fraction",
    "bridge/overshoot_penalty",
    "bridge/angular_speed_rad_s",
    "bridge/contact_gate_progress",
    "reward/palm_center_alignment",
    "geometry/min_palm_center_cosine",
)


def _series(
    scalars: dict[str, list[tuple[int, float, float]]], tag: str
) -> tuple[list[int], list[float]]:
    events = scalars.get(tag, [])
    return [event[0] for event in events], [event[1] for event in events]


def _plot_training_curves(
    scalars: dict[str, list[tuple[int, float, float]]], output: Path
) -> None:
    panels = (
        ("Loss/value_function", "Critic value loss"),
        ("Loss/surrogate", "PPO surrogate loss"),
        ("bridge/controlled_fraction", "Controlled-lift fraction"),
        ("bridge/stable_ready_fraction", "Stable-ready fraction"),
        ("bridge/overshoot_penalty", "Overshoot proxy"),
        ("bridge/angular_speed_rad_s", "Object angular speed (rad/s)"),
        ("reward/palm_center_alignment", "Palm-center alignment reward"),
        ("geometry/min_palm_center_cosine", "Weak-hand palm cosine"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(19, 8), constrained_layout=True)
    for axis, (tag, title) in zip(axes.flat, panels, strict=True):
        steps, values = _series(scalars, tag)
        axis.plot(steps, values, marker="o", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("PPO iteration")
        axis.grid(alpha=0.3)
    figure.suptitle("BODex bimanual controlled-5mm bridge training", fontsize=15)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_evaluation_comparison(
    baseline: dict[str, object], trained: dict[str, object], output: Path
) -> None:
    metrics = (
        "target_retained_at_terminal",
        "physically_bounded_micro_lift",
        "controlled_micro_lift",
        "stage4_ready_micro_lift",
        "stage2_contact_pass",
        "sustained_micro_lift",
    )
    labels = ("retained", "bounded", "controlled", "stage4-ready", "contact", "sustained")
    baseline_values = [100.0 * baseline["overall_rates"][metric] for metric in metrics]
    trained_values = [100.0 * trained["overall_rates"][metric] for metric in metrics]
    x_values = list(range(len(metrics)))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    width = 0.36
    axes[0].bar([x - width / 2 for x in x_values], baseline_values, width, label="baseline")
    axes[0].bar([x + width / 2 for x in x_values], trained_values, width, label="trained")
    axes[0].set_xticks(x_values, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Pass rate (%)")
    axes[0].set_title("Balanced 128-episode evaluation")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    candidates = sorted(baseline["candidate_rates"], key=int)
    candidate_x = list(range(len(candidates)))
    base_candidate = [
        100.0 * baseline["candidate_rates"][candidate]["controlled_micro_lift"]
        for candidate in candidates
    ]
    trained_candidate = [
        100.0 * trained["candidate_rates"][candidate]["controlled_micro_lift"]
        for candidate in candidates
    ]
    axes[1].bar(
        [x - width / 2 for x in candidate_x], base_candidate, width, label="baseline"
    )
    axes[1].bar(
        [x + width / 2 for x in candidate_x], trained_candidate, width, label="trained"
    )
    axes[1].set_xticks(candidate_x, [f"Candidate {value}" for value in candidates])
    axes[1].set_ylabel("Controlled-lift rate (%)")
    axes[1].set_title("Diversity guard: per-candidate performance")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _with_failure_funnels(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("failure_funnels") is not None:
        return payload
    summary = summarize_micro_lift_reports(
        payload["reports"], target_height_m=float(payload["target_height_m"])
    )
    return {**payload, "failure_funnels": summary["failure_funnels"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--trained-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    scalars = load_merged_scalars(args.training_dir)
    baseline = _with_failure_funnels(json.loads(args.baseline_evaluation.read_text()))
    trained = _with_failure_funnels(json.loads(args.trained_evaluation.read_text()))
    _plot_training_curves(scalars, args.output / "bridge_training_curves.png")
    _plot_evaluation_comparison(
        baseline, trained, args.output / "bridge_evaluation_comparison.png"
    )

    with (args.output / "bridge_training_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tag", "step", "value", "wall_time"))
        for tag in CURVE_TAGS:
            for step, value, wall_time in scalars.get(tag, []):
                writer.writerow((tag, step, value, wall_time))

    metrics = sorted(baseline["overall_rates"])
    report = {
        "schema": "xhand_bodex_lift_bridge_report_v1",
        "training_dir": str(args.training_dir.resolve()),
        "baseline_evaluation": str(args.baseline_evaluation.resolve()),
        "trained_evaluation": str(args.trained_evaluation.resolve()),
        "overall_delta_percentage_points": {
            metric: 100.0
            * (trained["overall_rates"][metric] - baseline["overall_rates"][metric])
            for metric in metrics
        },
        "baseline_failure_funnels": baseline.get("failure_funnels"),
        "trained_failure_funnels": trained.get("failure_funnels"),
        "candidate_controlled_rates": {
            candidate: {
                "baseline": baseline["candidate_rates"][candidate][
                    "controlled_micro_lift"
                ],
                "trained": trained["candidate_rates"][candidate][
                    "controlled_micro_lift"
                ],
            }
            for candidate in sorted(baseline["candidate_rates"], key=int)
        },
    }
    (args.output / "bridge_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
