#!/usr/bin/env python3
"""Render three static views of the verified left/right grasp."""

from pathlib import Path
import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.hand_model import create_hand_model


def add_mesh(axis, mesh, color, alpha, max_faces=5000):
    faces = mesh.faces
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
        faces = faces[indices]
    axis.add_collection3d(
        Poly3DCollection(
            mesh.vertices[faces],
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            rasterized=True,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            REPO
            / "graph_exp/bimanual_data/final_v2_lr_g98_tabletop_lift/"
            "bimanual_dataset.pt"
        ),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pose-stage",
        choices=("final", "seed", "outer", "command"),
        default="final",
    )
    parser.add_argument(
        "--show-table",
        action="store_true",
        help="Render a support plane at the object's initial bottom face.",
    )
    parser.add_argument(
        "--annotate-contacts",
        action="store_true",
        help="Label every geometric contact sample in all three views.",
    )
    parser.add_argument(
        "--contact-csv",
        type=Path,
        help="Write contact labels, link names, coordinates and distances.",
    )
    args = parser.parse_args()
    repo = REPO
    dataset_path = args.dataset.resolve()
    dataset = torch.load(
        dataset_path,
        map_location="cpu",
        weights_only=False,
    )
    samples = dataset.get("samples", dataset)
    sample = samples[args.sample_index]
    manifest = dataset.get("manifest", {})
    contact_threshold = manifest.get("contact_mm", 5.0) / 1000.0
    left_hand = create_hand_model("allegro_left", torch.device("cpu"))
    right_hand = create_hand_model("allegro_right", torch.device("cpu"))
    pose_keys = {
        "final": ("left_q", "right_q"),
        "seed": ("left_q_seed", "right_q_seed"),
        "outer": ("left_q_outer", "right_q_outer"),
        "command": ("left_q_command", "right_q_command"),
    }
    left_key, right_key = pose_keys[args.pose_stage]
    left_pose = sample[left_key]
    right_pose = sample[right_key]
    left_mesh = left_hand.get_trimesh_q(left_pose)["visual"]
    right_mesh = right_hand.get_trimesh_q(right_pose)["visual"]
    dataset_name, mesh_name = sample["object_name"].split("+", 1)
    object_mesh = trimesh.load_mesh(
        repo
        / "data/data_urdf/object"
        / dataset_name
        / mesh_name
        / f"{mesh_name}.stl",
        force="mesh",
    )
    object_query = trimesh.proximity.ProximityQuery(object_mesh)

    def contact_records(hand, q, side, threshold=0.005):
        # Match the final geometric audit: use the stored dense per-link cloud
        # (512 samples per link) rather than the 512-point search cloud for the
        # entire hand.  The visualization therefore shows every audited
        # contact sample inside the configured surface-distance threshold.
        links, _ = hand.get_transformed_links_pc(
            q,
            links_pc=hand.links_pc_original,
        )
        records = []
        minimum = float("inf")
        for link_name, link_points in links.items():
            points = link_points.cpu().numpy()
            _, distances, _ = object_query.on_surface(points)
            minimum = min(minimum, float(distances.min()))
            for point, distance in zip(points[distances <= threshold], distances[distances <= threshold]):
                records.append(
                    {
                        "side": side,
                        "link_name": str(link_name),
                        "point": point,
                        "distance_mm": float(distance * 1000.0),
                    }
                )
        prefix = "L" if side == "left" else "R"
        records.sort(key=lambda record: (record["link_name"], record["distance_mm"]))
        for index, record in enumerate(records, start=1):
            record["label"] = f"{prefix}{index}"
        return records, minimum

    left_records, left_minimum = contact_records(
        left_hand, left_pose, "left", contact_threshold
    )
    right_records, right_minimum = contact_records(
        right_hand, right_pose, "right", contact_threshold
    )
    left_contacts = np.asarray([record["point"] for record in left_records]).reshape(-1, 3)
    right_contacts = np.asarray([record["point"] for record in right_records]).reshape(-1, 3)

    if args.contact_csv:
        args.contact_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.contact_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("label", "side", "link_name", "x_m", "y_m", "z_m", "distance_mm"),
            )
            writer.writeheader()
            for record in left_records + right_records:
                writer.writerow(
                    {
                        "label": record["label"],
                        "side": record["side"],
                        "link_name": record["link_name"],
                        "x_m": f"{record['point'][0]:.8f}",
                        "y_m": f"{record['point'][1]:.8f}",
                        "z_m": f"{record['point'][2]:.8f}",
                        "distance_mm": f"{record['distance_mm']:.4f}",
                    }
                )
    bounds = np.vstack(
        [object_mesh.bounds, left_mesh.bounds, right_mesh.bounds]
    )
    table_mesh = None
    if args.show_table:
        table_span = max(float((bounds.max(axis=0) - bounds.min(axis=0))[:2].max()) * 1.35, 0.5)
        table_thickness = 0.012
        table_mesh = trimesh.creation.box(
            extents=(table_span, table_span, table_thickness)
        )
        # In the initial/open view the object rests on the table.  In the
        # final object-relative view, move the table down by the measured
        # lift so the image does not falsely depict post-lift support.
        lift_offset = (
            float(sample.get("metrics", {}).get("lift_displacement_mm", 0.0))
            / 1000.0
            if args.pose_stage == "final"
            else 0.0
        )
        table_mesh.apply_translation(
            (
                0.0,
                0.0,
                float(object_mesh.bounds[0, 2])
                - lift_offset
                - table_thickness / 2.0,
            )
        )
        bounds = np.vstack([bounds, table_mesh.bounds])
    lower = bounds.min(axis=0)
    upper = bounds.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max((upper - lower).max() / 2.0, 0.05) * 1.08

    figure = plt.figure(figsize=(18, 6.4))
    for index, azimuth in enumerate((35, 155, 275), start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        add_mesh(axis, object_mesh, (0.94, 0.44, 0.61), 0.70, 6500)
        if table_mesh is not None:
            add_mesh(axis, table_mesh, (0.55, 0.57, 0.61), 0.32, 12)
        left_points = left_mesh.sample(4500)
        right_points = right_mesh.sample(4500)
        axis.scatter(
            left_points[:, 0],
            left_points[:, 1],
            left_points[:, 2],
            s=0.8,
            color=(0.18, 0.51, 0.82),
            alpha=0.85,
            rasterized=True,
        )
        axis.scatter(
            right_points[:, 0],
            right_points[:, 1],
            right_points[:, 2],
            s=0.8,
            color=(0.20, 0.67, 0.38),
            alpha=0.85,
            rasterized=True,
        )
        axis.scatter(
            left_contacts[:, 0],
            left_contacts[:, 1],
            left_contacts[:, 2],
            s=24,
            color="#ffd400",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            label="Left contacts",
        )
        axis.scatter(
            right_contacts[:, 0],
            right_contacts[:, 1],
            right_contacts[:, 2],
            s=24,
            color="#b517e8",
            edgecolor="black",
            linewidth=0.8,
            depthshade=False,
            marker="D",
            label="Right contacts",
        )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=25, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(f"View {index}")
        if args.annotate_contacts:
            # Project the 3-D points and draw the labels as 2-D overlays.  This
            # keeps every label above the dense hand point clouds.
            label_offsets = {
                "L1": (-34, 30),
                "L2": (-34, -30),
                "R1": (34, 34),
                "R2": (42, 0),
                "R3": (34, -34),
            }
            for record in left_records + right_records:
                point = record["point"]
                projected_x, projected_y, _ = proj3d.proj_transform(
                    point[0], point[1], point[2], axis.get_proj()
                )
                is_left = record["side"] == "left"
                annotation_color = "#b28a00" if is_left else "#8612a8"
                default_offset = (
                    (-38, 24 + 18 * ((int(record["label"][1:]) - 1) % 3))
                    if is_left
                    else (38, 24 + 18 * ((int(record["label"][1:]) - 1) % 3))
                )
                axis.annotate(
                    record["label"],
                    xy=(projected_x, projected_y),
                    xytext=label_offsets.get(record["label"], default_offset),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    zorder=1000,
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "white",
                        "edgecolor": annotation_color,
                        "alpha": 0.98,
                        "linewidth": 1.4,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "color": annotation_color,
                        "linewidth": 1.2,
                        "shrinkA": 2,
                        "shrinkB": 2,
                    },
                    annotation_clip=False,
                )
    metrics = sample["metrics"]
    physics_label = ""
    if "robot_friction" in manifest:
        physics_label = (
            f" | friction={manifest['robot_friction']:.2f}"
        )
    if manifest.get("finger_effort_limit_nm") is not None:
        physics_label += (
            f" | finger effort<="
            f"{manifest['finger_effort_limit_nm']:.2f} Nm"
        )
    figure.suptitle(
        f"{args.pose_stage.capitalize()} tabletop lateral Allegro left/right bimanual grasp | "
        f"lift={metrics.get('lift_displacement_mm', float('nan')):.2f} mm | "
        f"gravity={metrics['gravity_displacement_mm']:.2f} mm | "
        f"6-dir={metrics['max_direction_displacement_mm']:.2f} mm"
        f"{physics_label}\n"
        f"pink=object, blue=left, green=right, yellow circles=left contacts, "
        f"purple diamonds=right contacts (within "
        f"{contact_threshold*1000:.0f} mm | "
        f"L={len(left_contacts)} ({left_minimum*1000:.2f} mm), "
        f"R={len(right_contacts)} ({right_minimum*1000:.2f} mm))",
        fontsize=14,
        fontweight="bold",
    )
    if args.annotate_contacts:
        contact_descriptions = []
        for record in left_records + right_records:
            contact_descriptions.append(
                f"{record['label']}: {record['link_name']} "
                f"(surface distance {record['distance_mm']:.2f} mm)"
            )
        figure.text(
            0.5,
            0.025,
            "  |  ".join(contact_descriptions),
            ha="center",
            va="bottom",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f7f7f7",
                "edgecolor": "#777777",
                "alpha": 0.96,
            },
        )
    figure.subplots_adjust(top=0.80, bottom=0.12, left=0.01, right=0.99, wspace=0.02)
    output = args.output or dataset_path.with_name(
        "verified_lateral_grasp_three_views.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
