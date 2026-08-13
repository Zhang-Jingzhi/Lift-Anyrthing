#!/usr/bin/env python3
"""Create a versioned V-HACD asset sourced from the exact visual surface.

High-v1 decomposes ``coacd_allinone.obj``.  For some ContactDB objects that
mesh is already an outward-shifted union of convex parts, so decomposing it a
second time can leave the simulated contact boundary several millimetres away
from the visual surface used by the strict realized-geometry audit.  High-v2
keeps the same Isaac V-HACD quality parameters but feeds the exact visual STL
to V-HACD.  All files are versioned and creation refuses to overwrite data.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MESH_NAME = "visual_surface_vhacd_high_v2.stl"
URDF_NAME = "vhacd_visual_high_v2_object_one_link.urdf"
MANIFEST_NAME = "vhacd_visual_high_v2_asset_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir", type=Path, required=True)
    args = parser.parse_args()
    object_dir = args.object_dir.resolve()
    visual = object_dir / f"{object_dir.name}.stl"
    target_mesh = object_dir / MESH_NAME
    target_urdf = object_dir / URDF_NAME
    target_manifest = object_dir / MANIFEST_NAME
    existing = [
        path for path in (target_mesh, target_urdf, target_manifest) if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite high-v2 assets: {existing}")
    if not visual.is_file():
        raise FileNotFoundError(visual)

    with visual.open("rb") as source_handle, target_mesh.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    urdf = (
        "<robot name=\"root\">\n"
        "  <link name=\"link_original\">\n"
        "    <visual>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        f"      <geometry><mesh filename=\"{visual.name}\" scale=\"1 1 1\"/></geometry>\n"
        "    </visual>\n"
        "    <collision>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        f"      <geometry><mesh filename=\"{MESH_NAME}\" scale=\"1 1 1\"/></geometry>\n"
        "    </collision>\n"
        "  </link>\n"
        "</robot>\n"
    )
    with target_urdf.open("x") as handle:
        handle.write(urdf)
    manifest = {
        "schema": "tro_grasp_vhacd_visual_high_v2_asset",
        "source_visual_mesh": visual.name,
        "source_visual_mesh_sha256": sha256(visual),
        "collision_mesh": target_mesh.name,
        "collision_mesh_sha256": sha256(target_mesh),
        "urdf": target_urdf.name,
        "vhacd_resolution": 1_000_000,
        "vhacd_max_convex_hulls": 128,
        "vhacd_max_vertices": 64,
    }
    with target_manifest.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
