import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.hand_model import create_hand_model


MODE_LABELS = {
    "shadowhand_unconditioned": "ShadowHand unconditioned",
    "allegro_conditioned": "AllegroHand conditioned",
}


def load_epoch(evaluation_root, epoch, mode):
    path = evaluation_root / f"epoch_{epoch:03d}" / mode / "vis.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = torch.load(path, map_location="cpu")
    samples = {}
    for entry in entries:
        for index in range(entry["predict_q"].shape[0]):
            key = (entry["object_name"], index)
            samples[key] = {
                "robot_name": entry["robot_name"],
                # Visualize the synthesized grasp pose itself. The rollout
                # success flag still comes from Isaac, while isaac_q is the
                # post-controller state and may be far from the object after a
                # failed rollout.
                "q": entry["predict_q"][index],
                "success": bool(entry["success"][index]),
            }
    return samples


def select_examples(epoch_samples, epochs):
    common_keys = set.intersection(
        *(set(epoch_samples[epoch]) for epoch in epochs)
    )
    records = []
    for key in sorted(common_keys):
        pattern = tuple(
            epoch_samples[epoch][key]["success"] for epoch in epochs
        )
        records.append((key, pattern))

    selected = []

    def pick(label, predicate, score):
        candidates = [
            (score(pattern), key, pattern)
            for key, pattern in records
            if key not in {item[1] for item in selected}
            and predicate(pattern)
        ]
        if candidates:
            _, key, pattern = max(candidates)
            selected.append((label, key, pattern))

    pick(
        "improves with training",
        lambda pattern: not pattern[0] and pattern[-1],
        lambda pattern: (
            sum(pattern[1:]),
            sum(pattern[-2:]),
            pattern,
        ),
    )
    pick(
        "consistently successful",
        lambda pattern: all(pattern),
        lambda pattern: pattern,
    )
    pick(
        "challenging / unstable",
        lambda pattern: not all(pattern),
        lambda pattern: (
            not pattern[-1],
            len(pattern) - sum(pattern),
            tuple(not value for value in pattern),
        ),
    )

    # Fill any missing category with the most informative remaining patterns.
    for key, pattern in sorted(
        records,
        key=lambda item: (
            len(set(item[1])),
            len(item[1]) - sum(item[1]),
        ),
        reverse=True,
    ):
        if len(selected) >= 3:
            break
        if key not in {item[1] for item in selected}:
            selected.append(("additional example", key, pattern))
    return selected


def mesh_collection(mesh, color, alpha, max_faces):
    faces = mesh.faces
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    polygons = mesh.vertices[faces]
    return Poly3DCollection(
        polygons,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        rasterized=True,
    )


def set_equal_limits(axis, bounds):
    lower, upper = bounds
    center = (lower + upper) / 2
    radius = max((upper - lower).max() / 2, 0.04) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def render_mode(repo, evaluation_root, output_dir, epochs, mode):
    epoch_samples = {
        epoch: load_epoch(evaluation_root, epoch, mode) for epoch in epochs
    }
    selected = select_examples(epoch_samples, epochs)
    if not selected:
        raise RuntimeError(f"No common samples found for {mode}")

    robot_name = next(iter(epoch_samples[epochs[0]].values()))["robot_name"]
    hand = create_hand_model(robot_name, torch.device("cpu"))
    object_cache = {}
    mesh_cache = {}

    for _, key, _ in selected:
        object_name, _ = key
        dataset, name = object_name.split("+")
        object_path = (
            repo / "data/data_urdf/object" / dataset / name / f"{name}.stl"
        )
        object_cache[object_name] = trimesh.load_mesh(
            object_path, force="mesh"
        )
        for epoch in epochs:
            q = epoch_samples[epoch][key]["q"]
            mesh_cache[(epoch, key)] = hand.get_trimesh_q(q)["visual"]

    figure, axes = plt.subplots(
        len(selected),
        len(epochs),
        figsize=(4.2 * len(epochs), 4.0 * len(selected)),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = np.asarray([axes])

    rows = []
    for row_index, (category, key, pattern) in enumerate(selected):
        object_name, sample_index = key
        object_mesh = object_cache[object_name]
        combined_bounds = np.vstack(
            [object_mesh.bounds]
            + [mesh_cache[(epoch, key)].bounds for epoch in epochs]
        )
        bounds = np.vstack(
            [combined_bounds.min(axis=0), combined_bounds.max(axis=0)]
        )
        for column_index, epoch in enumerate(epochs):
            axis = axes[row_index, column_index]
            robot_mesh = mesh_cache[(epoch, key)]
            success = epoch_samples[epoch][key]["success"]
            axis.add_collection3d(
                mesh_collection(
                    object_mesh,
                    color=(0.94, 0.52, 0.65),
                    alpha=0.78,
                    max_faces=65000,
                )
            )
            axis.add_collection3d(
                mesh_collection(
                    robot_mesh,
                    color=(0.32, 0.69, 0.95),
                    alpha=0.95,
                    max_faces=50000,
                )
            )
            set_equal_limits(axis, bounds)
            axis.view_init(elev=24, azim=42)
            axis.set_axis_off()
            axis.set_title(
                f"Epoch {epoch} | {'SUCCESS' if success else 'FAIL'}",
                color="#14833b" if success else "#c93434",
                fontsize=11,
                fontweight="bold",
            )
            rows.append(
                {
                    "mode": mode,
                    "category": category,
                    "object": object_name,
                    "sample_index": sample_index,
                    "epoch": epoch,
                    "success": success,
                }
            )
        axes[row_index, 0].text2D(
            -0.08,
            0.5,
            f"{category}\n{object_name}\nsample {sample_index}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=10,
        )

    figure.suptitle(
        f"{MODE_LABELS.get(mode, mode)}: checkpoint grasp-pose comparison",
        fontsize=16,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{mode}_checkpoint_comparison.png"
    figure.savefig(image_path, dpi=190, facecolor="white")
    plt.close(figure)

    selection_path = output_dir / f"{mode}_selections.csv"
    with selection_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "category",
                "object",
                "sample_index",
                "epoch",
                "success",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return image_path, selection_path, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path(
            "graph_exp/reproduction/train-multi-hand-rtx3070/evaluations"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/reproduction/train-multi-hand-rtx3070/"
            "checkpoint_visual_comparison"
        ),
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 30, 50, 60])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["shadowhand_unconditioned", "allegro_conditioned"],
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    evaluation_root = (
        args.evaluation_root
        if args.evaluation_root.is_absolute()
        else repo / args.evaluation_root
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo / args.output_dir
    )

    for mode in args.modes:
        image_path, selection_path, selected = render_mode(
            repo,
            evaluation_root,
            output_dir,
            args.epochs,
            mode,
        )
        print(image_path)
        print(selection_path)
        for category, key, pattern in selected:
            print(category, key, pattern)


if __name__ == "__main__":
    main()
