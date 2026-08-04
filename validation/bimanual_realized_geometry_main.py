"""Exact realized-pose geometry audit in a short-lived subprocess."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.hand_model import create_hand_model


def transformed_points(hand, q):
    # The filtered point cloud contains only 512 points for the whole hand;
    # some individual finger links consequently have as few as 7 samples.
    # That representation is useful for fast candidate search but is too
    # sparse for a strict 2 mm realized-contact audit.  Use the stored
    # 512-points-per-link cloud here (10,752 points per Allegro hand).
    links, _ = hand.get_transformed_links_pc(
        q,
        links_pc=hand.links_pc_original,
    )
    return links, torch.cat(list(links.values()), dim=0)


def nearest_point_distance_mm(left_points, right_points):
    """Return exact sampled-point clearance without an O(N*M) tensor."""
    left = left_points.detach().cpu().numpy()
    right = right_points.detach().cpu().numpy()
    if len(left) > len(right):
        left, right = right, left
    # Keep the audit predictable while the candidate generator is already
    # using the host cores; per-candidate audits are small enough that one
    # KD-tree worker is sufficient.
    distances, _ = cKDTree(right).query(left, k=1, workers=1)
    return float(distances.min()) * 1000.0


def query_surface_chunked(query, points, chunk_size=512):
    """Query a dense point cloud without trimesh allocating one huge batch."""
    closest_parts = []
    distance_parts = []
    triangle_parts = []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        closest, distances, triangle = query.on_surface(points[start:stop])
        closest_parts.append(closest)
        distance_parts.append(distances)
        triangle_parts.append(triangle)
    return (
        np.concatenate(closest_parts, axis=0),
        np.concatenate(distance_parts, axis=0),
        np.concatenate(triangle_parts, axis=0),
    )


def exact_metrics(mesh, query, links, points, contact_threshold):
    points_numpy = points.detach().cpu().numpy()
    closest, distances, triangle = query_surface_chunked(query, points_numpy)
    # Classify the side of the locally nearest oriented collision triangle.
    # This gives the penetration sign required at the strict 2 mm boundary
    # without launching tens of thousands of global ray casts.  The latter
    # are prohibitively slow on large COACD meshes and are undefined on the
    # original non-watertight ContactDB triangle soups.
    normals = mesh.face_normals[triangle]
    signed_side = np.einsum(
        "ij,ij->i",
        points_numpy - closest,
        normals,
    )
    inside = signed_side < 0.0
    penetration = float(distances[inside].max()) if inside.any() else 0.0
    contact_link_count = 0
    cursor = 0
    for link_points in links.values():
        # `points` is the ordered concatenation of `links.values()`, so reuse
        # the single dense nearest-surface query above instead of repeating
        # one costly mesh query for every link.
        link_count = len(link_points)
        link_distances = distances[cursor : cursor + link_count]
        cursor += link_count
        if float(link_distances.min()) <= contact_threshold:
            contact_link_count += 1
    return {
        "penetration_depth_mm": penetration * 1000.0,
        "min_surface_distance_mm": float(distances.min()) * 1000.0,
        "contact_point_count": int(
            np.count_nonzero(distances <= contact_threshold)
        ),
        "contact_link_count": contact_link_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--left-q-file", type=Path, required=True)
    parser.add_argument("--right-q-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--contact-mm", type=float, default=5.0)
    parser.add_argument(
        "--audit-collision-mesh",
        action="store_true",
        help=(
            "Audit the combined COACD mesh instead of the visual surface. "
            "Disabled by default because overlapping convex components "
            "contain internal faces that do not define object inside/outside."
        ),
    )
    parser.add_argument("--left-robot-name", default="allegro_left")
    parser.add_argument("--right-robot-name", default="allegro_right")
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
    visual_mesh_path = (
        args.repo
        / "data/data_urdf/object"
        / dataset_name
        / mesh_name
        / f"{mesh_name}.stl"
    )
    collision_mesh_path = (
        args.repo
        / "data/data_urdf/object"
        / dataset_name
        / mesh_name
        / "coacd_allinone.obj"
    )
    mesh_path = (
        collision_mesh_path
        if args.audit_collision_mesh and collision_mesh_path.is_file()
        else visual_mesh_path
    )
    mesh = trimesh.load_mesh(mesh_path, process=True)
    query = trimesh.proximity.ProximityQuery(mesh)
    left_hand = create_hand_model(
        args.left_robot_name,
        torch.device("cpu"),
    )
    right_hand = create_hand_model(
        args.right_robot_name,
        torch.device("cpu"),
    )
    results = []
    for left, right in zip(left_q, right_q):
        left_links, left_points = transformed_points(left_hand, left)
        right_links, right_points = transformed_points(right_hand, right)
        results.append(
            {
                "left": exact_metrics(
                    mesh,
                    query,
                    left_links,
                    left_points,
                    args.contact_mm / 1000.0,
                ),
                "right": exact_metrics(
                    mesh,
                    query,
                    right_links,
                    right_points,
                    args.contact_mm / 1000.0,
                ),
                "clearance_mm": nearest_point_distance_mm(
                    left_points,
                    right_points,
                ),
            }
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output_file)


if __name__ == "__main__":
    main()
