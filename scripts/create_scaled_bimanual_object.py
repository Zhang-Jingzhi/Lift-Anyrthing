#!/usr/bin/env python3
"""Create a physically scaled object asset and matching source-pose entry."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

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
    output_files = [
        target / f"{args.target_object}.stl",
        target / "coacd_allinone.obj",
        target / "coacd_decomposed_object_one_link.urdf",
        args.output_vis,
    ]
    existing = [path for path in output_files if path.exists()]
    if target.exists() or existing:
        raise RuntimeError(
            "Refusing to overwrite scaled-object output: "
            + ", ".join(str(path) for path in ([target] + existing))
        )
    target.mkdir(parents=True, exist_ok=False)
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

    visual_name = escape(f"{args.target_object}.stl")
    urdf = f"""<robot name="root">
  <link name="link_original">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{visual_name}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="coacd_allinone.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""
    (target / "coacd_decomposed_object_one_link.urdf").write_text(urdf)

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
