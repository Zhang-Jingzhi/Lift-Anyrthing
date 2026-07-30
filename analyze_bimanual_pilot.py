"""Audit and visualize a completed strict bimanual pilot dataset."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from generate_bimanual_pilot import DIRECTION_NAMES, transformed_points
from utils.hand_model import create_hand_model


BOOL_FIELDS = {
    "joint_limit_pass",
    "outer_penetration_pass",
    "inner_penetration_pass",
    "inter_hand_clearance_pass",
    "dual_contact_pass",
    "geometry_pass",
    "isaac_evaluated",
    "isaac_success",
    "realized_joint_limit_pass",
    "realized_object_penetration_pass",
    "realized_hand_clearance_pass",
    "realized_pose_pass",
    "strict_success",
}


def load_rows(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in BOOL_FIELDS:
            row[key] = row.get(key, "") == "True"
        for key, value in list(row.items()):
            if key in BOOL_FIELDS or key in {
                "object_name",
                "rejection_reasons",
            }:
                continue
            if value == "":
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def equal_axes(axis, groups):
    points = np.concatenate(groups, axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 0.01)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def make_dashboard(rows, output_path):
    strict = [row for row in rows if row["strict_success"]]
    funnel_keys = [
        ("Candidates", None),
        ("Joint limits", "joint_limit_pass"),
        ("Open pose", "outer_penetration_pass"),
        ("Hand clearance", "inter_hand_clearance_pass"),
        ("Dual proximity", "dual_contact_pass"),
        ("Geometry", "geometry_pass"),
        ("Isaac", "isaac_success"),
        ("Final strict", "strict_success"),
    ]
    funnel_values = [
        len(rows) if key is None else sum(row[key] for row in rows)
        for _, key in funnel_keys
    ]

    object_counts = Counter(row["object_name"] for row in strict)
    object_names = sorted({row["object_name"] for row in rows})
    reason_counts = Counter()
    for row in rows:
        reason_counts.update(
            reason
            for reason in row["rejection_reasons"].split(";")
            if reason
        )

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axis = axes[0, 0]
    axis.bar(
        [label for label, _ in funnel_keys],
        funnel_values,
        color="#3978b9",
    )
    axis.set_title("Candidate filtering funnel")
    axis.set_ylabel("Number of candidates")
    axis.tick_params(axis="x", rotation=35)
    for index, value in enumerate(funnel_values):
        axis.text(index, value, str(value), ha="center", va="bottom")

    axis = axes[0, 1]
    values = [object_counts[name] for name in object_names]
    short_names = [name.split("+", 1)[1] for name in object_names]
    axis.bar(short_names, values, color="#3a9d5d")
    axis.set_title("Strict retained samples per development object")
    axis.set_ylabel("Samples")
    axis.tick_params(axis="x", rotation=30)
    for index, value in enumerate(values):
        axis.text(index, value, str(value), ha="center", va="bottom")

    axis = axes[1, 0]
    reasons = reason_counts.most_common()
    if reasons:
        labels, values = zip(*reasons)
        axis.barh(labels, values, color="#d95f59")
        axis.invert_yaxis()
        for index, value in enumerate(values):
            axis.text(value, index, f" {value}", va="center")
    axis.set_title("Rejection reasons (non-exclusive)")
    axis.set_xlabel("Occurrences")

    axis = axes[1, 1]
    direction_values = [
        [
            row[f"displacement_{direction}_mm"]
            for row in strict
            if row[f"displacement_{direction}_mm"] is not None
        ]
        for direction in DIRECTION_NAMES
    ]
    if strict:
        axis.boxplot(direction_values, tick_labels=DIRECTION_NAMES)
    axis.axhline(20, color="#d62728", linestyle="--", label="20 mm limit")
    axis.set_title("Six-direction displacement of retained samples")
    axis.set_ylabel("Displacement (mm)")
    axis.legend()

    figure.suptitle("TRO-Grasp bimanual pilot data audit", fontsize=18)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_representatives(repo, dataset, output_path):
    samples_by_object = defaultdict(list)
    for sample in dataset["samples"]:
        samples_by_object[sample["object_name"]].append(sample)

    object_names = sorted(samples_by_object)
    hand = create_hand_model("allegro", torch.device("cpu"))
    rows = 2
    columns = 3
    figure = plt.figure(figsize=(16, 10))
    np.random.seed(20260730)
    for plot_index, object_name in enumerate(object_names):
        samples = sorted(
            samples_by_object[object_name],
            key=lambda sample: sample["metrics"][
                "max_direction_displacement_mm"
            ],
        )
        sample = samples[len(samples) // 2]
        dataset_name, mesh_name = object_name.split("+", 1)
        mesh_path = (
            repo
            / "data/data_urdf/object"
            / dataset_name
            / mesh_name
            / f"{mesh_name}.stl"
        )
        mesh = trimesh.load_mesh(mesh_path)
        object_points = mesh.sample(1600)
        left_points = transformed_points(hand, sample["left_q"])
        right_points = transformed_points(hand, sample["right_q"])

        axis = figure.add_subplot(
            rows,
            columns,
            plot_index + 1,
            projection="3d",
        )
        axis.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            s=2,
            alpha=0.38,
            color="#ef709d",
            label="object",
        )
        axis.scatter(
            left_points[:, 0],
            left_points[:, 1],
            left_points[:, 2],
            s=4,
            alpha=0.75,
            color="#2878b5",
            label="hand A",
        )
        axis.scatter(
            right_points[:, 0],
            right_points[:, 1],
            right_points[:, 2],
            s=4,
            alpha=0.75,
            color="#33a65c",
            label="hand B",
        )
        equal_axes(axis, [object_points, left_points, right_points])
        max_mm = sample["metrics"]["max_direction_displacement_mm"]
        pen_mm = max(
            sample["metrics"]["left_realized_penetration_mm"],
            sample["metrics"]["right_realized_penetration_mm"],
        )
        clearance_mm = sample["metrics"]["realized_hand_clearance_mm"]
        axis.set_title(
            f"{mesh_name}\n"
            f"max disturbance={max_mm:.2f} mm, "
            f"penetration={pen_mm:.2f} mm, "
            f"clearance={clearance_mm:.2f} mm",
            fontsize=10,
        )
        axis.set_axis_off()

    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle(
        "Representative strict bimanual samples "
        "(median disturbance per object)",
        fontsize=17,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_summary(rows, dataset, manifest):
    strict_rows = [row for row in rows if row["strict_success"]]
    samples = dataset["samples"]
    object_counts = Counter(sample["object_name"] for sample in samples)
    serialized = []
    finite = True
    shape_pass = True
    for sample in samples:
        pair = torch.cat([sample["left_q"], sample["right_q"]])
        finite = finite and bool(torch.isfinite(pair).all())
        shape_pass = shape_pass and tuple(pair.shape) == (44,)
        serialized.append(
            tuple(torch.round(pair * 1e5).to(torch.int64).tolist())
        )
    duplicate_count = len(serialized) - len(set(serialized))
    direction_maxima = [
        row["max_direction_displacement_mm"] for row in strict_rows
    ]
    realized_penetrations = [
        max(
            row["left_realized_penetration_mm"],
            row["right_realized_penetration_mm"],
        )
        for row in strict_rows
    ]
    clearances = [
        row["realized_hand_clearance_mm"] for row in strict_rows
    ]
    expected = set(manifest["train_objects"])
    observed = set(object_counts)
    return {
        "dataset_version": dataset["version"],
        "candidate_count": len(rows),
        "retained_count": len(samples),
        "retention_rate": len(samples) / max(len(rows), 1),
        "object_counts": dict(sorted(object_counts.items())),
        "expected_objects_present": expected == observed,
        "held_out_objects_absent": not (
            set(manifest["held_out_objects"]) & observed
        ),
        "all_q_finite": finite,
        "all_pair_shapes_44": shape_pass,
        "duplicate_count_at_1e-5": duplicate_count,
        "all_retained_rows_strict": all(
            row["isaac_success"]
            and row["realized_pose_pass"]
            and row["strict_success"]
            for row in strict_rows
        ),
        "max_retained_direction_displacement_mm": (
            max(direction_maxima) if direction_maxima else None
        ),
        "max_retained_realized_penetration_mm": (
            max(realized_penetrations)
            if realized_penetrations
            else None
        ),
        "min_retained_realized_hand_clearance_mm": (
            min(clearances) if clearances else None
        ),
    }


def write_report(path, summary, manifest):
    status_checks = [
        (
            "All configured development objects present",
            summary["expected_objects_present"],
        ),
        ("Held-out objects absent", summary["held_out_objects_absent"]),
        ("All 44-DoF pairs finite", summary["all_q_finite"]),
        ("All saved pair shapes are 44", summary["all_pair_shapes_44"]),
        ("All saved samples pass strict gates", summary["all_retained_rows_strict"]),
        (
            "No duplicate pair at 1e-5 precision",
            summary["duplicate_count_at_1e-5"] == 0,
        ),
    ]
    lines = [
        "# Bimanual pilot data audit",
        "",
        f"- Version: `{summary['dataset_version']}`",
        f"- Candidates: {summary['candidate_count']}",
        f"- Strict retained: {summary['retained_count']} "
        f"({summary['retention_rate']:.1%})",
        "",
        "## Integrity checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {label}"
        for label, passed in status_checks
    )
    lines.extend(
        [
            "",
            "## Object distribution",
            "",
            "| Object | Retained |",
            "|---|---:|",
        ]
    )
    for name, count in summary["object_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Strict retained extrema",
            "",
            "- Maximum six-direction displacement: "
            f"{summary['max_retained_direction_displacement_mm']:.3f} mm.",
            "- Maximum realized hand-object penetration: "
            f"{summary['max_retained_realized_penetration_mm']:.3f} mm.",
            "- Minimum realized inter-hand clearance: "
            f"{summary['min_retained_realized_hand_clearance_mm']:.3f} mm.",
            "",
            "Thresholds are "
            f"{manifest['disturbance_threshold_mm']} mm displacement, "
            f"{manifest['penetration_mm']} mm penetration, and "
            f">{manifest['clearance_mm']} mm hand clearance.",
            "",
            "## Interpretation boundary",
            "",
            "This is a validated pilot for data-pipeline development, not yet",
            "a hardware-faithful left/right Allegro dataset. Both actors use",
            "the repository's same left-hand URDF, gravity is disabled to match",
            "the legacy TRO-Grasp protocol, and held-out-object generalization",
            "has not yet been evaluated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/pilot_v4_realized_strict"
        ),
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    if not args.data_dir.is_absolute():
        args.data_dir = (args.repo / args.data_dir).resolve()

    rows = load_rows(args.data_dir / "sample_results.csv")
    dataset = torch.load(
        args.data_dir / "bimanual_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if len(dataset["samples"]) != sum(
        row["strict_success"] for row in rows
    ):
        raise RuntimeError("Dataset/CSV retained counts do not match")

    make_dashboard(rows, args.data_dir / "audit_dashboard.png")
    make_representatives(
        args.repo,
        dataset,
        args.data_dir / "representative_pairs.png",
    )
    summary = make_summary(rows, dataset, manifest)
    (args.data_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_report(args.data_dir / "DATA_AUDIT.md", summary, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
