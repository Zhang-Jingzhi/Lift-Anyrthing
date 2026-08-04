#!/usr/bin/env python3
"""Build a mechanically mirrored Allegro right-hand asset.

The repository ships an Allegro left hand only.  This script reflects the
physical hand through the palm-local x-z plane (y -> -y) while keeping the six
virtual world-pose joints unchanged.  Keeping the virtual joints unchanged
means that the first six generalized coordinates retain their usual world
xyz/rpy meaning for both hands.

For a reflection matrix M = diag(1, -1, 1):

* points and prismatic axes transform as M v;
* rotations transform as M R M;
* revolute axes transform as -M a (angular vectors are pseudovectors);
* inertia tensors transform as M I M;
* triangle winding is reversed after reflecting mesh vertices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from lxml import etree


MIRROR = np.diag([1.0, -1.0, 1.0])


def parse_vector(value: str | None, default: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        return np.zeros(3) if default is None else default.copy()
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {value!r}")
    return result


def format_vector(value: np.ndarray) -> str:
    cleaned = np.where(np.abs(value) < 5e-13, 0.0, value)
    return " ".join(f"{float(component):.17g}" for component in cleaned)


def mirror_origin(origin: etree._Element | None) -> None:
    if origin is None:
        return
    xyz = parse_vector(origin.get("xyz"))
    rpy = parse_vector(origin.get("rpy"))
    # URDF's fixed-axis roll-pitch-yaw convention gives the exact identity
    # M R(r, p, y) M = R(-r, p, -y) for M = diag(1, -1, 1).
    mirrored_rpy = np.asarray([-rpy[0], rpy[1], -rpy[2]])
    if origin.get("xyz") is not None:
        origin.set("xyz", format_vector(MIRROR @ xyz))
    if origin.get("rpy") is not None:
        origin.set("rpy", format_vector(mirrored_rpy))


def inertia_matrix(element: etree._Element) -> np.ndarray:
    return np.array(
        [
            [float(element.get("ixx")), float(element.get("ixy")), float(element.get("ixz"))],
            [float(element.get("ixy")), float(element.get("iyy")), float(element.get("iyz"))],
            [float(element.get("ixz")), float(element.get("iyz")), float(element.get("izz"))],
        ],
        dtype=np.float64,
    )


def set_inertia_matrix(element: etree._Element, matrix: np.ndarray) -> None:
    for key, value in {
        "ixx": matrix[0, 0],
        "ixy": matrix[0, 1],
        "ixz": matrix[0, 2],
        "iyy": matrix[1, 1],
        "iyz": matrix[1, 2],
        "izz": matrix[2, 2],
    }.items():
        element.set(key, f"{float(value):.17g}")


def right_name(name: str) -> str:
    return name.replace("_left", "_right").replace("left_", "right_")


def mirror_obj(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for raw_line in source.read_text(errors="strict").splitlines(keepends=True):
        newline = "\n" if raw_line.endswith("\n") else ""
        line = raw_line.rstrip("\r\n")
        if line.startswith("v ") or line.startswith("vn "):
            prefix, *tokens = line.split()
            if len(tokens) < 3:
                raise ValueError(f"Malformed OBJ vector in {source}: {line}")
            vector = np.asarray([float(tokens[0]), float(tokens[1]), float(tokens[2])])
            vector = MIRROR @ vector
            tokens[:3] = [f"{component:.9g}" for component in vector]
            line = " ".join([prefix, *tokens])
        elif line.startswith("f "):
            prefix, *tokens = line.split()
            line = " ".join([prefix, *reversed(tokens)])
        elif line.startswith("mtllib "):
            prefix, material = line.split(maxsplit=1)
            line = f"{prefix} {right_name(material)}"
        elif line.startswith("o "):
            line = right_name(line)
        output.append(line + newline)
    destination.write_text("".join(output))


def mirror_point_cloud(source: Path, destination: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    mirrored = {}
    for group_name, group in payload.items():
        mirrored[group_name] = {}
        for link_name, points in group.items():
            points = points.clone()
            points[..., 1] *= -1
            mirrored[group_name][link_name] = points
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mirrored, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(repo: Path) -> dict:
    left_dir = repo / "data/data_urdf/robot/allegro"
    right_dir = repo / "data/data_urdf/robot/allegro_right"
    source_urdf = left_dir / "allegro_hand_left_extended.urdf"
    target_urdf = right_dir / "allegro_hand_right_extended.urdf"
    parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = etree.parse(str(source_urdf), parser)
    root = tree.getroot()
    root.set("name", "allegro_right")

    referenced_meshes: set[str] = set()
    for link in root.findall("link"):
        for container_name in ("inertial", "visual", "collision"):
            for container in link.findall(container_name):
                mirror_origin(container.find("origin"))
                if container_name == "inertial":
                    inertia = container.find("inertia")
                    if inertia is not None:
                        set_inertia_matrix(inertia, MIRROR @ inertia_matrix(inertia) @ MIRROR)
                mesh = container.find("geometry/mesh")
                if mesh is not None:
                    filename = mesh.get("filename")
                    if filename is None:
                        raise ValueError(f"Mesh without filename in link {link.get('name')}")
                    mirrored_filename = right_name(filename)
                    mesh.set("filename", mirrored_filename)
                    referenced_meshes.add(filename)

    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name.startswith("virtual_"):
            continue
        mirror_origin(joint.find("origin"))
        axis = joint.find("axis")
        if axis is not None:
            vector = parse_vector(axis.get("xyz"))
            if joint.get("type") in {"revolute", "continuous"}:
                vector = -MIRROR @ vector
            elif joint.get("type") == "prismatic":
                vector = MIRROR @ vector
            axis.set("xyz", format_vector(vector))

    right_dir.mkdir(parents=True, exist_ok=True)
    tree.write(
        str(target_urdf),
        pretty_print=True,
        # pytorch_kinematics passes this file to lxml as a Unicode string;
        # lxml rejects Unicode strings that contain an encoding declaration.
        xml_declaration=False,
        encoding="utf-8",
    )

    for relative_name in sorted(referenced_meshes):
        source_mesh = left_dir / relative_name
        target_relative = Path(right_name(relative_name))
        target_mesh = right_dir / target_relative
        if source_mesh.suffix.lower() != ".obj":
            raise ValueError(f"Only OBJ mirroring is implemented: {source_mesh}")
        mirror_obj(source_mesh, target_mesh)
        source_mtl = source_mesh.with_suffix(".mtl")
        if source_mtl.is_file():
            target_mtl = target_mesh.with_suffix(".mtl")
            target_mtl.parent.mkdir(parents=True, exist_ok=True)
            target_mtl.write_text(source_mtl.read_text())

    shutil.copy2(left_dir / "LICENSE", right_dir / "LICENSE")
    left_pc = repo / "data/PointCloud/robot/allegro.pt"
    right_pc = repo / "data/PointCloud/robot/allegro_right.pt"
    left_alias_pc = repo / "data/PointCloud/robot/allegro_left.pt"
    mirror_point_cloud(left_pc, right_pc)
    shutil.copy2(left_pc, left_alias_pc)

    manifest = {
        "asset": "allegro_right",
        "source": str(source_urdf.relative_to(repo)),
        "output": str(target_urdf.relative_to(repo)),
        "mirror_plane": "palm-local y=0 (y -> -y)",
        "virtual_world_pose_joints_mirrored": False,
        "mesh_count": len(referenced_meshes),
        "source_sha256": sha256(source_urdf),
        "output_sha256": sha256(target_urdf),
        "point_cloud": str(right_pc.relative_to(repo)),
    }
    (right_dir / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    manifest = build(args.repo.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
