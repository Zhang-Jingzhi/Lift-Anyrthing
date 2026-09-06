#!/usr/bin/env python3
"""Measure how far each BODex candidate's hands already sit inside the object.

A candidate whose reset state overlaps the object gets a depenetration impulse
from PhysX on the first physics steps, which launches the object before the
policy has done anything.  Measured 2026-09-06 on a captured episode: the object
sits still for three steps and then jumps to 1.65 m/s.  Everything downstream --
contact depth, lift control, terminal stability -- inherits that.

This is pure geometry on the real visual meshes, no simulator: forward
kinematics from the URDF, then a signed distance from every hand-link surface
sample to the object mesh.  Positive means inside the object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from yourdfpy import URDF

HAND_LINK_HINTS = ("hand", "thumb", "index", "mid", "ring", "pinky")
SAMPLES_PER_LINK = 400


def hand_link_names(robot: URDF) -> list[str]:
    names = []
    for node in robot.scene.graph.nodes_geometry:
        low = node.lower()
        if any(h in low for h in HAND_LINK_HINTS):
            names.append(node)
    return names


def object_mesh(sample: dict) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(sample["object_mesh_path"], force="mesh", process=False)
    pose = np.asarray(sample["object_pose_world"], dtype=np.float64)
    transform = np.eye(4)
    if pose.shape == (4, 4):
        transform = pose
    else:
        transform[:3, 3] = pose[:3]
        if pose.shape == (7,):
            w, x, y, z = pose[3:7]
            transform[:3, :3] = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]
    mesh.apply_transform(transform)
    return mesh


def measure(robot: URDF, links: list[str], obj: trimesh.Trimesh, joint_values: dict) -> dict:
    robot.update_cfg(
        np.asarray([joint_values.get(n, 0.0) for n in robot.actuated_joint_names])
    )
    worst = 0.0
    worst_link = ""
    clearance = float("inf")
    overlapping = []
    for link in links:
        transform, geometry_name = robot.scene.graph.get(link)
        geometry = robot.scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        points, _ = trimesh.sample.sample_surface(geometry, SAMPLES_PER_LINK)
        points = trimesh.transform_points(points, np.asarray(transform, dtype=np.float64))
        signed = trimesh.proximity.signed_distance(obj, points)  # >0 inside
        depth = float(signed.max())
        clearance = min(clearance, float(-signed.max()))
        if depth > 0.0:
            overlapping.append((link, depth))
            if depth > worst:
                worst, worst_link = depth, link
    overlapping.sort(key=lambda item: -item[1])
    return {
        "maximum_penetration_m": worst,
        "deepest_link": worst_link,
        "minimum_clearance_m": clearance,
        "overlapping_link_count": len(overlapping),
        "overlapping_links": [
            {"link": name, "penetration_m": round(depth, 6)} for name, depth in overlapping[:6]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-urdf", type=Path, required=True)
    ap.add_argument("--bank", type=Path, action="append", required=True)
    ap.add_argument("--states", default="pregrasp_full_body_q,full_body_q")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    robot = URDF.load(str(args.robot_urdf))
    links = hand_link_names(robot)
    rows = []
    for bank_path in args.bank:
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        for sample in bank["samples"]:
            obj = object_mesh(sample)
            names = list(sample["joint_names"])
            row = {"bank": bank_path.name, "candidate_id": sample["candidate_id"]}
            for state in args.states.split(","):
                values = dict(zip(names, np.asarray(sample[state], dtype=np.float64).tolist()))
                row[state] = measure(robot, links, obj, values)
            rows.append(row)
            pre = row["pregrasp_full_body_q"]
            grasp = row["full_body_q"]
            print(
                f"{sample['candidate_id'][:44]:46s} "
                f"pregrasp {pre['maximum_penetration_m'] * 1000:7.2f}mm "
                f"({pre['overlapping_link_count']:2d} links)  "
                f"grasp {grasp['maximum_penetration_m'] * 1000:7.2f}mm "
                f"({grasp['overlapping_link_count']:2d} links)"
            )
    args.output.write_text(
        json.dumps(
            {
                "schema": "xhand_bodex_candidate_object_overlap_v1",
                "robot_urdf": str(args.robot_urdf),
                "surface_samples_per_link": SAMPLES_PER_LINK,
                "hand_link_count": len(links),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    clean = [r for r in rows if r["pregrasp_full_body_q"]["maximum_penetration_m"] <= 0.0]
    print(f"\n{len(clean)}/{len(rows)} candidates have no pregrasp overlap")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
