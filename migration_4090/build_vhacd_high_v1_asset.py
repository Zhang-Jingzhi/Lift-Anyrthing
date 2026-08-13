#!/usr/bin/env python3
"""Create protocol-unique mesh/URDF names for high-quality V-HACD caches."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MESH_NAME = "coacd_allinone_vhacd_high_v1.obj"
URDF_NAME = "vhacd_high_v1_object_one_link.urdf"
MANIFEST_NAME = "vhacd_high_v1_asset_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir", type=Path, required=True)
    args = parser.parse_args()
    object_dir = args.object_dir.resolve()
    source = object_dir / "coacd_allinone.obj"
    visual = object_dir / f"{object_dir.name}.stl"
    target_mesh = object_dir / MESH_NAME
    target_urdf = object_dir / URDF_NAME
    target_manifest = object_dir / MANIFEST_NAME
    if target_mesh.exists() or target_urdf.exists() or target_manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite high-v1 assets in {object_dir}"
        )
    if not source.is_file() or not visual.is_file():
        raise FileNotFoundError(f"missing source assets in {object_dir}")
    with source.open("rb") as source_handle, target_mesh.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    urdf = (
        "<robot name=\"root\">\n"
        "  <link name=\"link_original\">\n"
        "    <visual>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        f"      <geometry><mesh filename=\"{visual.name}\" "
        "scale=\"1 1 1\"/></geometry>\n"
        "    </visual>\n"
        "    <collision>\n"
        "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
        f"      <geometry><mesh filename=\"{MESH_NAME}\" "
        "scale=\"1 1 1\"/></geometry>\n"
        "    </collision>\n"
        "  </link>\n"
        "</robot>\n"
    )
    with target_urdf.open("x") as handle:
        handle.write(urdf)
    manifest = {
        "schema": "tro_grasp_vhacd_high_v1_asset",
        "source_mesh": source.name,
        "source_mesh_sha256": sha256(source),
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
    print(f"object={object_dir.name} mesh={target_mesh} urdf={target_urdf}")


if __name__ == "__main__":
    main()
