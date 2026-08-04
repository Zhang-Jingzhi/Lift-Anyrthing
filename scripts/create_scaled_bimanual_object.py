#!/usr/bin/env python3
"""Create a physically scaled object asset and matching source-pose entry."""

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--dataset", default="contactdb")
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--scale", nargs=3, type=float, required=True)
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--output-vis", type=Path, required=True)
    args = parser.parse_args()

    source = (
        args.repo
        / "data/data_urdf/object"
        / args.dataset
        / args.source_object
    )
    target = (
        args.repo
        / "data/data_urdf/object"
        / args.dataset
        / args.target_object
    )
    target.mkdir(parents=True, exist_ok=True)
    scale = np.asarray(args.scale, dtype=np.float64)

    visual = trimesh.load_mesh(
        source / f"{args.source_object}.stl", force="mesh", process=False
    )
    visual.vertices *= scale
    visual.export(target / f"{args.target_object}.stl")

    collision = trimesh.load_mesh(
        source / "coacd_allinone.obj", force="mesh", process=False
    )
    collision.vertices *= scale
    collision.export(target / "coacd_allinone.obj")

    source_entries = torch.load(
        args.source_vis, map_location="cpu", weights_only=False
    )
    old_name = f"{args.dataset}+{args.source_object}"
    new_name = f"{args.dataset}+{args.target_object}"
    original = next(entry for entry in source_entries if entry["object_name"] == old_name)
    replacement = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in original.items()
    }
    replacement["object_name"] = new_name
    output_entries = [
        entry for entry in source_entries if entry["object_name"] != new_name
    ]
    output_entries.append(replacement)
    args.output_vis.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_entries, args.output_vis)

    print(f"created {new_name}: extents_mm={(visual.extents * 1000).round(2)}")
    print(args.output_vis)


if __name__ == "__main__":
    main()
