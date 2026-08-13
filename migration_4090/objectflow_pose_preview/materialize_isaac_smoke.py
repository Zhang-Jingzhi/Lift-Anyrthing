#!/usr/bin/env python3
"""Materialize one candidate bank entry for the Isaac XHand validator.

The adapter accepts both the older ObjectFlow schema and the compact XHand
candidate-bank schema.  It copies the selected mesh into a run-local asset
directory and writes a small, self-contained ``sample.pt`` without changing
the source bank or mesh.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    return trimesh.util.concatenate(tuple(mesh.geometry.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--asset-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(args.bank, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    if not 0 <= args.index < len(samples):
        raise IndexError(f"candidate index {args.index} outside bank of {len(samples)}")
    source = samples[args.index]
    source_mesh = Path(source["object_mesh_path"])
    mesh = load_mesh(source_mesh)

    # ObjectFlow samples store a preview transform for the scanned mesh.  The
    # compact bank already stores a canonical mesh and world pose; applying a
    # second transform there would move the object twice.
    if "object_pose_robot_preview" in source:
        transform = np.asarray(source["object_pose_robot_preview"], dtype=float)
        mesh.apply_transform(transform)
        center = mesh.bounding_box.centroid
        mesh.apply_translation(-center)
        object_pose_world = [
            [1, 0, 0, float(center[0])],
            [0, 1, 0, float(center[1])],
            [0, 0, 1, float(center[2])],
            [0, 0, 0, 1],
        ]
        object_name = source.get(
            "episode", "objectflow"
        ) + f"_frame_{int(source.get('frame_index', args.index)):05d}"
        source_schema = source.get("schema", "objectflow")
    else:
        center = np.asarray(source["object_pose_world"], dtype=float)[:3, 3]
        object_pose_world = source["object_pose_world"]
        object_name = source.get("object_name", f"candidate_{args.index:04d}")
        source_schema = source.get("schema", "xhand_compact_candidate_bank_v2")

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.asset_dir / "object_world_aligned.obj"
    mesh.export(mesh_path)

    full_q = source.get("full_body_ik_q", source.get("full_body_q"))
    if full_q is None:
        raise KeyError("candidate has neither full_body_ik_q nor full_body_q")
    joint_names = list(source.get("joint_names", []))
    if not joint_names:
        raise KeyError("candidate is missing joint_names")

    sample = {
        "method": source.get("method", "objectflow_kinematic_preview"),
        "object_name": object_name,
        "object_mesh_path": str(mesh_path.resolve()),
        "object_pose_world": object_pose_world,
        "target_tcp_positions_world": source["target_tcp_positions_world"],
        "achieved_tcp_positions_world": source["achieved_tcp_positions_world"],
        "full_body_q": full_q,
        "joint_names": joint_names,
        "ik_position_error_m": source.get("ik_position_error_m", []),
        "fixed_waist": True,
        "geometry_pass": False,
        "isaaclab_physical_validated": False,
        "isaac_gym_physical_validated": False,
        "source_pose": source.get(
            "object_pose_source_opencv", source.get("object_pose_world")
        ),
        "source_schema": source_schema,
    }
    for key in (
        "physical_parameters",
        "pregrasp_full_body_q",
        "pregrasp_target_tcp_positions_world",
        "pregrasp_achieved_tcp_positions_world",
    ):
        if key in source:
            sample[key] = source[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "xhand_fullbody_grasp_pose_v1", "samples": [sample]}, args.output)
    print(
        json.dumps(
            {
                "dataset": str(args.output.resolve()),
                "sample": object_name,
                "mesh": str(mesh_path.resolve()),
                "center_world": center.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
