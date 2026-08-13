#!/usr/bin/env python3
"""Expose each convex CoACD component as its own URDF collision shape."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import trimesh


URDF_NAME = "coacd_decomposed_object_multicollision_v1.urdf"
PARTS_NAME = "coacd_parts_multicollision_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir", type=Path, required=True)
    args = parser.parse_args()
    object_dir = args.object_dir.resolve()
    source = object_dir / "coacd_allinone.obj"
    visual = object_dir / f"{object_dir.name}.stl"
    output_urdf = object_dir / URDF_NAME
    output_parts = object_dir / PARTS_NAME
    if output_urdf.exists() or output_parts.exists():
        raise FileExistsError(
            f"refusing to overwrite {output_urdf} or {output_parts}"
        )
    if not source.is_file() or not visual.is_file():
        raise FileNotFoundError(f"missing source assets in {object_dir}")

    mesh = trimesh.load_mesh(source, process=False, force="mesh")
    source_components = list(mesh.split(only_watertight=False))
    if not source_components:
        raise RuntimeError(f"no connected components found in {source}")
    repaired_indices = [
        index
        for index, part in enumerate(source_components)
        if not part.is_convex
    ]
    # CoACD output is intended to be convex, but its triangulation can fail
    # trimesh's numerical convexity predicate on a small number of parts.
    # Hull only those individual parts; never hull the complete object.
    components = [
        part.convex_hull if index in repaired_indices else part
        for index, part in enumerate(source_components)
    ]
    if any(not part.is_convex for part in components):
        raise RuntimeError("component-level convex repair failed")

    staging = Path(tempfile.mkdtemp(prefix=f".{PARTS_NAME}.", dir=object_dir))
    try:
        for index, part in enumerate(components):
            part.export(staging / f"part_{index:04d}.obj")
        (staging / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "tro_grasp_coacd_multicollision_v1",
                    "source": source.name,
                    "component_count": len(components),
                    "all_components_convex": True,
                    "component_convex_repairs": repaired_indices,
                    "source_component_volumes_m3": [
                        float(part.volume) for part in source_components
                    ],
                    "component_volumes_m3": [
                        float(part.volume) for part in components
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        os.rename(staging, output_parts)
    except Exception:
        raise

    collision_xml = "\n".join(
        "    <collision>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        "      <geometry><mesh filename=\""
        f"{PARTS_NAME}/part_{index:04d}.obj"
        "\" scale=\"1 1 1\"/></geometry>\n"
        "    </collision>"
        for index in range(len(components))
    )
    urdf = (
        "<robot name=\"root\">\n"
        "  <link name=\"link_original\">\n"
        "    <visual>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        f"      <geometry><mesh filename=\"{visual.name}\" "
        "scale=\"1 1 1\"/></geometry>\n"
        "    </visual>\n"
        f"{collision_xml}\n"
        "  </link>\n"
        "</robot>\n"
    )
    # Exclusive creation preserves the user's no-overwrite requirement.
    with output_urdf.open("x") as handle:
        handle.write(urdf)
    print(
        f"object={object_dir.name} convex_components={len(components)} "
        f"urdf={output_urdf}"
    )


if __name__ == "__main__":
    main()
