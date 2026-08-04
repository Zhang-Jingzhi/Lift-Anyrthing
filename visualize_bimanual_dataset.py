"""Create stratified static and interactive QC assets for bimanual data."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.hand_model import create_hand_model


CATEGORY_SPECS = (
    (
        "most_stable",
        "Most stable",
        lambda sample: sample["metrics"]["max_direction_displacement_mm"],
        False,
    ),
    (
        "typical",
        "Typical",
        lambda sample: sample["metrics"]["max_direction_displacement_mm"],
        None,
    ),
    (
        "near_penetration_limit",
        "Highest penetration",
        lambda sample: max(
            sample["metrics"]["left_realized_penetration_mm"],
            sample["metrics"]["right_realized_penetration_mm"],
        ),
        True,
    ),
    (
        "near_clearance_limit",
        "Lowest hand clearance",
        lambda sample: sample["metrics"]["realized_hand_clearance_mm"],
        False,
    ),
)


def mesh_collection(mesh, color, alpha=0.9, max_faces=4500):
    faces = mesh.faces
    if len(faces) > max_faces:
        indices = np.linspace(
            0,
            len(faces) - 1,
            max_faces,
            dtype=np.int64,
        )
        faces = faces[indices]
    return Poly3DCollection(
        mesh.vertices[faces],
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        rasterized=True,
    )


def equal_limits(axis, bounds):
    lower, upper = bounds
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2, 0.04) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def choose_unique(samples, key, reverse):
    ordered = sorted(samples, key=key, reverse=reverse)
    return ordered


def select_samples(samples_by_object):
    selected = {}
    for object_name, samples in sorted(samples_by_object.items()):
        used = set()
        selections = []
        for category, label, key, reverse in CATEGORY_SPECS:
            if reverse is None:
                ordered = sorted(samples, key=key)
                center = len(ordered) // 2
                index_order = sorted(
                    range(len(ordered)),
                    key=lambda index: abs(index - center),
                )
                ordered = [ordered[index] for index in index_order]
            else:
                ordered = choose_unique(samples, key, reverse)
            sample = next(
                item
                for item in ordered
                if int(item["candidate_index"]) not in used
            )
            used.add(int(sample["candidate_index"]))
            selections.append((category, label, sample))
        selected[object_name] = selections
    return selected


def object_mesh(repo, object_name):
    dataset, name = object_name.split("+", 1)
    return trimesh.load_mesh(
        repo
        / "data/data_urdf/object"
        / dataset
        / name
        / f"{name}.stl",
        force="mesh",
    )


def render_gallery(repo, selected, output_path):
    hand = create_hand_model("allegro", torch.device("cpu"))
    np.random.seed(20260730)
    object_names = list(selected)
    figure, axes = plt.subplots(
        len(object_names),
        len(CATEGORY_SPECS),
        figsize=(18, 4.2 * len(object_names)),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    if len(object_names) == 1:
        axes = np.asarray([axes])

    for row_index, object_name in enumerate(object_names):
        mesh = object_mesh(repo, object_name)
        rendered = []
        bounds = [mesh.bounds]
        for _, _, sample in selected[object_name]:
            left = hand.get_trimesh_q(sample["left_q"])["visual"]
            right = hand.get_trimesh_q(sample["right_q"])["visual"]
            rendered.append(
                (
                    sample,
                    left,
                    right,
                    left.sample(1400),
                    right.sample(1400),
                )
            )
            bounds.extend((left.bounds, right.bounds))
        stacked = np.vstack(bounds)
        row_bounds = np.vstack(
            (stacked.min(axis=0), stacked.max(axis=0))
        )

        for column_index, (
            (category, label, _),
            (sample, left, right, left_points, right_points),
        ) in enumerate(zip(selected[object_name], rendered)):
            axis = axes[row_index, column_index]
            axis.add_collection3d(
                mesh_collection(
                    mesh,
                    color=(0.94, 0.44, 0.61),
                    alpha=0.70,
                    max_faces=6500,
                )
            )
            axis.scatter(
                left_points[:, 0],
                left_points[:, 1],
                left_points[:, 2],
                s=1.5,
                color=(0.18, 0.51, 0.82),
                alpha=0.85,
                rasterized=True,
            )
            axis.scatter(
                right_points[:, 0],
                right_points[:, 1],
                right_points[:, 2],
                s=1.5,
                color=(0.20, 0.67, 0.38),
                alpha=0.85,
                rasterized=True,
            )
            equal_limits(axis, row_bounds)
            axis.view_init(elev=24, azim=42)
            axis.set_axis_off()

            metrics = sample["metrics"]
            penetration = max(
                metrics["left_realized_penetration_mm"],
                metrics["right_realized_penetration_mm"],
            )
            axis.set_title(
                f"{label} | sample {sample['candidate_index']}\n"
                f"dist={metrics['max_direction_displacement_mm']:.2f} mm, "
                f"pen={penetration:.2f} mm, "
                f"clear={metrics['realized_hand_clearance_mm']:.2f} mm",
                fontsize=9,
            )
        axes[row_index, 0].text2D(
            -0.08,
            0.5,
            object_name.split("+", 1)[1],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=12,
            fontweight="bold",
        )

    figure.suptitle(
        "Strict bimanual dataset: stratified visual quality control\n"
        "pink=object, blue=hand A, green=hand B",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def selection_rows(selected):
    rows = []
    for object_name, selections in selected.items():
        for category, _, sample in selections:
            metrics = sample["metrics"]
            rows.append(
                {
                    "object_name": object_name,
                    "category": category,
                    "candidate_index": int(sample["candidate_index"]),
                    "max_direction_displacement_mm": metrics[
                        "max_direction_displacement_mm"
                    ],
                    "max_realized_penetration_mm": max(
                        metrics["left_realized_penetration_mm"],
                        metrics["right_realized_penetration_mm"],
                    ),
                    "realized_hand_clearance_mm": metrics[
                        "realized_hand_clearance_mm"
                    ],
                    "left_contact_point_count": metrics[
                        "left_contact_point_count"
                    ],
                    "right_contact_point_count": metrics[
                        "right_contact_point_count"
                    ],
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, audit_rows):
    worst_disturbance = max(
        rows,
        key=lambda row: float(row["max_direction_displacement_mm"]),
    )
    worst_penetration = max(
        rows,
        key=lambda row: float(row["max_realized_penetration_mm"]),
    )
    worst_clearance = min(
        rows,
        key=lambda row: float(row["realized_hand_clearance_mm"]),
    )
    lines = [
        "# Bimanual visual QC",
        "",
        "- 24 strict samples selected: four strata for each of six objects.",
        "- Strata: most stable, typical, highest retained penetration, and",
        "  lowest retained inter-hand clearance.",
        "- All displayed samples already passed the exact geometry and",
        "  six-direction Isaac gates.",
        f"- Excluded mesh-audit errors listed separately: {len(audit_rows)}.",
        "",
        "## Boundary cases in the selected gallery",
        "",
        "- Largest disturbance: "
        f"`{worst_disturbance['object_name']}` sample "
        f"{worst_disturbance['candidate_index']} — "
        f"{float(worst_disturbance['max_direction_displacement_mm']):.3f} mm.",
        "- Largest penetration: "
        f"`{worst_penetration['object_name']}` sample "
        f"{worst_penetration['candidate_index']} — "
        f"{float(worst_penetration['max_realized_penetration_mm']):.3f} mm.",
        "- Smallest hand clearance: "
        f"`{worst_clearance['object_name']}` sample "
        f"{worst_clearance['candidate_index']} — "
        f"{float(worst_clearance['realized_hand_clearance_mm']):.3f} mm.",
        "",
        "Thresholds are <20 mm disturbance, <=5 mm hand-object",
        "penetration, and >2 mm inter-hand clearance.",
        "",
        "The gallery is a deterministic screening aid. Interactive inspection",
        "is still recommended before using the data for a longer training run.",
    ]
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
    repo = args.repo.resolve()
    data_dir = (
        args.data_dir
        if args.data_dir.is_absolute()
        else repo / args.data_dir
    )
    output_dir = data_dir / "visual_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(
        data_dir / "bimanual_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    samples_by_object = defaultdict(list)
    for sample in dataset["samples"]:
        samples_by_object[sample["object_name"]].append(sample)
    selected = select_samples(samples_by_object)

    render_gallery(
        repo,
        selected,
        output_dir / "stratified_gallery.png",
    )
    rows = selection_rows(selected)
    write_csv(output_dir / "selected_samples.csv", rows)
    torch.save(
        [
            {
                "category": category,
                "sample": sample,
            }
            for selections in selected.values()
            for category, _, sample in selections
        ],
        output_dir / "selected_samples.pt",
    )

    with (data_dir / "sample_results.csv").open(newline="") as file:
        all_rows = list(csv.DictReader(file))
    audit_rows = [
        row
        for row in all_rows
        if row.get("realized_geometry_audit_error") == "True"
    ]
    if audit_rows:
        write_csv(output_dir / "excluded_audit_errors.csv", audit_rows)
    write_report(output_dir / "VISUAL_QC.md", rows, audit_rows)
    print(output_dir)
    print(f"selected={len(rows)} audit_errors={len(audit_rows)}")


if __name__ == "__main__":
    main()
