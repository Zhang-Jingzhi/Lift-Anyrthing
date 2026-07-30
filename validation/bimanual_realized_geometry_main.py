"""Exact realized-pose geometry audit in a short-lived subprocess."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hand_model import create_hand_model


def transformed_points(hand, q):
    links, _ = hand.get_transformed_links_pc(q)
    return torch.cat(list(links.values()), dim=0)


def exact_metrics(mesh, query, points, contact_threshold):
    points_numpy = points.detach().cpu().numpy()
    _, distances, _ = query.on_surface(points_numpy)
    inside = mesh.contains(points_numpy)
    penetration = float(distances[inside].max()) if inside.any() else 0.0
    return {
        "penetration_depth_mm": penetration * 1000.0,
        "min_surface_distance_mm": float(distances.min()) * 1000.0,
        "contact_point_count": int(
            np.count_nonzero(distances <= contact_threshold)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--left-q-file", type=Path, required=True)
    parser.add_argument("--right-q-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--contact-mm", type=float, default=5.0)
    args = parser.parse_args()

    left_q = torch.load(
        args.left_q_file,
        map_location="cpu",
        weights_only=False,
    )
    right_q = torch.load(
        args.right_q_file,
        map_location="cpu",
        weights_only=False,
    )
    if left_q.shape != right_q.shape or left_q.shape[-1] != 22:
        raise ValueError("Expected matching [N, 22] left/right tensors")

    dataset_name, mesh_name = args.object_name.split("+", 1)
    mesh_path = (
        args.repo
        / "data/data_urdf/object"
        / dataset_name
        / mesh_name
        / f"{mesh_name}.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    query = trimesh.proximity.ProximityQuery(mesh)
    hand = create_hand_model("allegro", torch.device("cpu"))
    results = []
    for left, right in zip(left_q, right_q):
        left_points = transformed_points(hand, left)
        right_points = transformed_points(hand, right)
        results.append(
            {
                "left": exact_metrics(
                    mesh,
                    query,
                    left_points,
                    args.contact_mm / 1000.0,
                ),
                "right": exact_metrics(
                    mesh,
                    query,
                    right_points,
                    args.contact_mm / 1000.0,
                ),
                "clearance_mm": float(
                    torch.cdist(left_points, right_points).min() * 1000
                ),
            }
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output_file)


if __name__ == "__main__":
    main()
