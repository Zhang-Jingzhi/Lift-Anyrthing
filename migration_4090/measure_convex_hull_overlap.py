#!/usr/bin/env python3
"""Measure the grasp against the collision geometry the simulator actually uses.

measure_bank_overlap.py samples the visual meshes, which is what BODex optimises
against, and reports every zero_overlap4 grasp pose as clear: 0.34 to 13.47 mm.
PhysX reports a median of 8.2 mm of hand-object penetration on the same poses.

The robot USD names its collision approximation `convexHull`, and the manifest
calls the URDF `robot_visual_urdf`.  A convex hull of a curved finger link is
strictly larger than the link, so a grasp that just touches the exact mesh
interpenetrates the hull.  This measures the same poses against per-link convex
hulls to see whether that gap accounts for the penetration PhysX reports.

Same sampling and signed-distance machinery as measure_bank_overlap, so the two
numbers are comparable directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from yourdfpy import URDF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_bank_overlap import SAMPLES_PER_LINK, hand_link_names, object_mesh  # noqa: E402


def link_geometry(robot: URDF, link: str, *, hull: bool):
    transform, geometry_name = robot.scene.graph.get(link)
    geometry = robot.scene.geometry.get(geometry_name)
    if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
        return None, None
    return (geometry.convex_hull if hull else geometry), np.asarray(transform)


def worst_penetration(robot: URDF, links, obj, joint_values: dict, *, hull: bool) -> float:
    """Deepest penetration in millimetres; negative means everything is clear."""
    robot.update_cfg(joint_values)
    deepest = -np.inf
    for link in links:
        geometry, transform = link_geometry(robot, link, hull=hull)
        if geometry is None:
            continue
        points, _ = trimesh.sample.sample_surface(geometry, SAMPLES_PER_LINK)
        points = trimesh.transform_points(points, transform)
        signed = trimesh.proximity.signed_distance(obj, points)  # > 0 is inside
        deepest = max(deepest, float(np.max(signed)))
    return deepest * 1000.0


def hull_bulge(robot: URDF, links) -> float:
    """How far each link's convex hull stands off its own surface, in millimetres."""
    worst = 0.0
    for link in links:
        geometry, _ = link_geometry(robot, link, hull=False)
        if geometry is None:
            continue
        hull = geometry.convex_hull
        points, _ = trimesh.sample.sample_surface(hull, SAMPLES_PER_LINK)
        distance = trimesh.proximity.signed_distance(geometry, points)
        worst = max(worst, float(np.max(-distance)))
    return worst * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    args = parser.parse_args()

    bank = torch.load(args.bodex_bank, map_location="cpu", weights_only=False)
    robot = URDF.load(str(args.urdf))
    links = hand_link_names(robot)

    print("每个手部连杆的凸包相对自身表面最多外扩 %.2f mm" % hull_bulge(robot, links))
    print()
    print("%-6s %26s %26s" % ("cand", "精确网格穿透 (mm)", "凸包穿透 (mm)"))
    print("%-6s %26s %26s" % ("", "BODex 优化所用", "PhysX 实际所用"))
    for index, sample in enumerate(bank["samples"]):
        obj = object_mesh(sample)
        names = list(sample["joint_names"])
        source = sample.get("controller_grasp_full_body_q", sample.get("full_body_q"))
        values = dict(zip(names, [float(v) for v in source]))
        exact = worst_penetration(robot, links, obj, values, hull=False)
        hull = worst_penetration(robot, links, obj, values, hull=True)
        print("%-6d %23.2f mm %23.2f mm" % (index, exact, hull))


if __name__ == "__main__":
    main()
