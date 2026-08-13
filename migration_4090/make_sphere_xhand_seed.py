#!/usr/bin/env python3
"""Derive a minimal Tianji/XHand IK seed for the scaled sphere asset."""

import copy
from pathlib import Path

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "migration_4090/results/xhand_compact_piggy_small_x0p50_source_v1/"
    "baseline__contactdb__piggy_bank_formal_large_random_v1_000.pt"
)
MESH = ROOT / (
    "data/data_urdf/object/contactdb/sphere_large_xhand_290mm_v1/"
    "coacd_allinone.obj"
)
OUTPUT = ROOT / "migration_4090/results/xhand_sphere_290mm_seed_v1.pt"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    sample = copy.deepcopy(payload["samples"][0])
    mesh = trimesh.load(MESH, force="mesh", process=False)
    table_top = 0.71
    object_z = table_top - float(mesh.bounds[0, 2])
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [0.50, 0.0, object_z]
    # Start with opposed wrists just outside a 290 mm sphere and slightly
    # below its geometric centre.  The upward-palm generator solves the exact
    # dual-arm IK and adds the outward pre-grasp trajectory.
    target_z = object_z + float(mesh.bounds[:, 2].mean()) - 0.02
    targets = np.asarray(
        [[0.50, 0.205, target_z], [0.50, -0.205, target_z]],
        dtype=np.float32,
    )
    sample.update({
        "object_name": "contactdb+sphere_large_xhand_290mm_v1",
        "object_mesh_path": str(MESH),
        "object_pose_world": pose.tolist(),
        "object_transform": {
            "source_extents_m": np.asarray(mesh.extents).tolist(),
            "grasp_scale_xyz": [1.0, 1.0, 1.0],
            "scale_mode": "uniform",
            "target_y_extent_m": float(mesh.extents[1]),
        },
        "target_tcp_positions_world": targets.tolist(),
        "geometry_pass": False,
        "isaaclab_physical_validated": False,
    })
    torch.save({"schema": "xhand_sphere_seed_v1", "samples": [sample]}, OUTPUT)
    print(OUTPUT)
    print("sphere_extents_m", np.asarray(mesh.extents).tolist())
    print("object_pose_z_m", object_z)
    print("target_tcp_positions_world", targets.tolist())


if __name__ == "__main__":
    main()
