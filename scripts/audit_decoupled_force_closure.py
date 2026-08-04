#!/usr/bin/env python3
"""Audit per-hand force-closure quality for bimanual grasp samples.

This is a lightweight BiDexGrasp-style audit.  It builds a friction-pyramid
grasp matrix independently for the left and right hand from dense realized
contacts, then solves non-negative least-squares problems for twelve unit
wrench directions.  It is a geometric/contact-quality metric; final samples
must still pass gravity, effort-limited lift, and disturbance simulation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.optimize import nnls

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hand_model import create_hand_model


def transformed_dense_links(hand, q):
    links, _ = hand.get_transformed_links_pc(
        q,
        links_pc=hand.links_pc_original,
    )
    return {
        name: points.detach().cpu().numpy()
        for name, points in links.items()
    }


def tangent_basis(normal):
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    tangent_a = np.cross(normal, helper)
    tangent_a /= np.linalg.norm(tangent_a) + 1e-12
    tangent_b = np.cross(normal, tangent_a)
    tangent_b /= np.linalg.norm(tangent_b) + 1e-12
    return tangent_a, tangent_b


def contact_set(mesh, query, links, threshold_m, points_per_link):
    contacts = []
    link_rows = []
    for link_name, points in links.items():
        closest, distance, triangle = query.on_surface(points)
        indices = np.flatnonzero(distance <= threshold_m)
        if len(indices) == 0:
            continue
        indices = indices[np.argsort(distance[indices])[:points_per_link]]
        link_rows.append(
            {
                "link_name": link_name,
                "point_count": int(len(indices)),
                "min_distance_mm": float(distance[indices].min() * 1000),
            }
        )
        for index in indices:
            # trimesh face normals point out of the object.  The force that
            # an external finger applies to the object points inward.
            inward = -np.asarray(mesh.face_normals[triangle[index]])
            inward /= np.linalg.norm(inward) + 1e-12
            contacts.append((closest[index], inward, link_name))
    return contacts, link_rows


def grasp_matrix(contacts, center, radius, friction):
    columns = []
    for point, normal, _ in contacts:
        tangent_a, tangent_b = tangent_basis(normal)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            force = normal + friction * (sx * tangent_a + sy * tangent_b)
            force /= np.linalg.norm(force) + 1e-12
            torque = np.cross(point - center, force) / max(radius, 1e-6)
            columns.append(np.concatenate([force, torque]))
    if not columns:
        return np.zeros((6, 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def wrench_coverage(matrix):
    targets = np.concatenate([np.eye(6), -np.eye(6)], axis=0)
    residuals = []
    coefficients = []
    if matrix.shape[1] == 0:
        return [1.0] * len(targets), [0.0] * len(targets)
    for target in targets:
        solution, residual = nnls(matrix, target, maxiter=10000)
        residuals.append(float(residual))
        coefficients.append(float(solution.sum()))
    return residuals, coefficients


def audit_hand(mesh, query, hand, q, args):
    links = transformed_dense_links(hand, q)
    contacts, contact_links = contact_set(
        mesh,
        query,
        links,
        args.contact_mm / 1000.0,
        args.points_per_link,
    )
    center = np.asarray(mesh.bounding_box.centroid)
    radius = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]) * 0.5)
    matrix = grasp_matrix(contacts, center, radius, args.friction)
    residuals, coefficients = wrench_coverage(matrix)
    return {
        "contact_count": len(contacts),
        "contact_link_count": len(contact_links),
        "contact_links": contact_links,
        "wrench_residuals": residuals,
        "mean_wrench_residual": float(np.mean(residuals)),
        "max_wrench_residual": float(np.max(residuals)),
        "min_wrench_residual": float(np.min(residuals)),
        "mean_force_coefficient_sum": float(np.mean(coefficients)),
        "force_closure_proxy_pass": bool(
            len(contact_links) >= args.min_contact_links
            and (
                (
                    args.residual_statistic == "max"
                    and np.max(residuals) <= args.max_wrench_residual
                )
                or (
                    args.residual_statistic == "mean"
                    and np.mean(residuals) <= args.max_wrench_residual
                )
            )
        ),
    }


def load_samples(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "samples" in payload:
            return payload["samples"]
        if "retained" in payload:
            return payload["retained"]
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--contact-mm", type=float, default=2.0)
    parser.add_argument("--points-per-link", type=int, default=8)
    parser.add_argument("--min-contact-links", type=int, default=2)
    parser.add_argument("--max-wrench-residual", type=float, default=0.35)
    parser.add_argument(
        "--residual-statistic",
        choices=("max", "mean"),
        default="max",
        help=(
            "Use mean for cooperative large-object grasps, where requiring "
            "each hand to independently span every 6-D wrench would "
            "contradict the bimanual-required objective."
        ),
    )
    parser.add_argument("--left-robot-name", default="allegro_left")
    parser.add_argument("--right-robot-name", default="allegro_right")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    left_hand = create_hand_model(args.left_robot_name, torch.device("cpu"))
    right_hand = create_hand_model(args.right_robot_name, torch.device("cpu"))
    meshes = {}
    queries = {}
    rows = []
    for index, sample in enumerate(samples):
        object_name = sample["object_name"]
        if object_name not in meshes:
            dataset_name, mesh_name = object_name.split("+", 1)
            mesh_path = (
                args.repo
                / "data/data_urdf/object"
                / dataset_name
                / mesh_name
                / f"{mesh_name}.stl"
            )
            meshes[object_name] = trimesh.load_mesh(mesh_path)
            queries[object_name] = trimesh.proximity.ProximityQuery(
                meshes[object_name]
            )
        mesh = meshes[object_name]
        query = queries[object_name]
        left = audit_hand(mesh, query, left_hand, sample["left_q"], args)
        right = audit_hand(mesh, query, right_hand, sample["right_q"], args)
        rows.append(
            {
                "sample_index": index,
                "object_name": object_name,
                "left": left,
                "right": right,
                "decoupled_force_closure_pass": bool(
                    left["force_closure_proxy_pass"]
                    and right["force_closure_proxy_pass"]
                ),
            }
        )

    output = {
        "method": "bidexgrasp_inspired_decoupled_wrench_audit_v1",
        "source_dataset": str(args.dataset),
        "friction": args.friction,
        "contact_mm": args.contact_mm,
        "points_per_link": args.points_per_link,
        "min_contact_links": args.min_contact_links,
        "max_wrench_residual": args.max_wrench_residual,
        "residual_statistic": args.residual_statistic,
        "num_samples": len(rows),
        "num_pass": sum(row["decoupled_force_closure_pass"] for row in rows),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
