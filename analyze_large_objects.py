"""Rank TRO-Grasp objects by mesh size and prepare a bimanual object split."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


DEFAULT_DEVELOPMENT_OBJECTS = [
    "ycb+bleach_cleanser",
    "ycb+wood_block",
    "contactdb+piggy_bank",
    "ycb+power_drill",
    "contactdb+cube_large",
    "contactdb+cylinder_large",
]

DEFAULT_TEST_OBJECTS = [
    "ycb+pitcher_base",
    "ycb+cracker_box",
    "ycb+toy_airplane",
]


def mesh_path(data_root, object_name):
    dataset, name = object_name.split("+")
    return data_root / dataset / name / f"{name}.stl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/data_urdf/object"),
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path(
            "data/CMapDataset_filtered/split_train_validate_objects.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/object_selection"
        ),
    )
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text())
    selected_role = {
        name: "development" for name in DEFAULT_DEVELOPMENT_OBJECTS
    }
    selected_role.update({name: "held_out_test" for name in DEFAULT_TEST_OBJECTS})

    rows = []
    for source_split, object_names in split.items():
        for object_name in object_names:
            path = mesh_path(args.data_root, object_name)
            mesh = trimesh.load_mesh(path, force="mesh")
            extents_mm = np.asarray(mesh.extents) * 1000
            rows.append(
                {
                    "object_name": object_name,
                    "source_split": source_split,
                    "bimanual_role": selected_role.get(object_name, ""),
                    "x_mm": extents_mm[0],
                    "y_mm": extents_mm[1],
                    "z_mm": extents_mm[2],
                    "max_dimension_mm": extents_mm.max(),
                    "middle_dimension_mm": np.sort(extents_mm)[-2],
                    "diagonal_mm": np.linalg.norm(extents_mm),
                    "volume_cm3": abs(mesh.volume) * 1e6,
                    "watertight": bool(mesh.is_watertight),
                    "mesh_path": str(path.resolve()),
                }
            )
    rows.sort(
        key=lambda row: (
            row["max_dimension_mm"],
            row["middle_dimension_mm"],
        ),
        reverse=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "object_sizes.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = [row for row in rows if row["bimanual_role"]]
    split_path = args.output_dir / "bimanual_object_split.json"
    split_path.write_text(
        json.dumps(
            {
                "development": DEFAULT_DEVELOPMENT_OBJECTS,
                "held_out_test": DEFAULT_TEST_OBJECTS,
            },
            indent=2,
        )
        + "\n"
    )

    top = rows[:20][::-1]
    colors = [
        (
            "#3478bf"
            if row["bimanual_role"] == "development"
            else "#ee8a2d"
            if row["bimanual_role"] == "held_out_test"
            else "#aaaaaa"
        )
        for row in top
    ]
    figure, axis = plt.subplots(figsize=(11, 7.5), constrained_layout=True)
    axis.barh(
        [row["object_name"] for row in top],
        [row["max_dimension_mm"] for row in top],
        color=colors,
    )
    axis.axvline(150, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("Maximum mesh dimension (mm)")
    axis.set_title("TRO-Grasp large-object candidates")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(
        args.output_dir / "large_object_ranking.png",
        dpi=180,
        facecolor="white",
    )
    plt.close(figure)

    report = [
        "# Large-object selection for bimanual TRO-Grasp",
        "",
        "Objects are ranked by the maximum dimension of their original STL mesh.",
        "Blue objects form the development set; orange objects are reserved for",
        "bimanual held-out evaluation.",
        "",
        "## Development objects",
        "",
        "| Object | Dimensions (mm) | Max (mm) |",
        "|---|---:|---:|",
    ]
    for row in selected:
        if row["bimanual_role"] != "development":
            continue
        report.append(
            f"| `{row['object_name']}` | "
            f"{row['x_mm']:.1f} × {row['y_mm']:.1f} × "
            f"{row['z_mm']:.1f} | {row['max_dimension_mm']:.1f} |"
        )
    report.extend(
        [
            "",
            "## Held-out bimanual test objects",
            "",
            "| Object | Dimensions (mm) | Max (mm) |",
            "|---|---:|---:|",
        ]
    )
    for row in selected:
        if row["bimanual_role"] != "held_out_test":
            continue
        report.append(
            f"| `{row['object_name']}` | "
            f"{row['x_mm']:.1f} × {row['y_mm']:.1f} × "
            f"{row['z_mm']:.1f} | {row['max_dimension_mm']:.1f} |"
        )
    report.extend(
        [
            "",
            "![Large-object ranking](large_object_ranking.png)",
            "",
            "The repository's original train/validation split was designed for",
            "single-hand generalization. This new split is only for the bimanual",
            "extension: held-out objects must not be used when constructing paired",
            "or optimized bimanual training demonstrations.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(csv_path)
    print(split_path)
    print(args.output_dir / "large_object_ranking.png")
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
