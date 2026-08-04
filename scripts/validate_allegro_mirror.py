#!/usr/bin/env python3
"""Validate that Allegro right is the exact y-reflection of Allegro left."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import trimesh
from lxml import etree
from scipy.spatial import cKDTree
from urdf_parser_py.urdf import URDF


MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])


def load_mesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    return mesh_or_scene


def origin_matrix(container: etree._Element) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    origin = container.find("origin")
    result = np.eye(4)
    if origin is None:
        return result
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def link_geometry_vertices(
    urdf_dir: Path,
    link: etree._Element,
    container_name: str,
) -> np.ndarray:
    vertices = []
    for container in link.findall(container_name):
        geometry = container.find("geometry")
        if geometry is None or len(geometry) != 1:
            continue
        element = geometry[0]
        if element.tag == "mesh":
            mesh = load_mesh(
                trimesh.load(urdf_dir / element.get("filename"), process=False)
            ).copy()
            scale = np.fromstring(element.get("scale", "1 1 1"), sep=" ")
            mesh.vertices *= scale
        elif element.tag == "box":
            mesh = trimesh.creation.box(
                extents=np.fromstring(element.get("size"), sep=" ")
            )
        elif element.tag == "sphere":
            mesh = trimesh.creation.icosphere(radius=float(element.get("radius")))
        elif element.tag == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=float(element.get("radius")),
                height=float(element.get("length")),
            )
        else:
            raise ValueError(f"Unsupported URDF geometry: {element.tag}")
        mesh.apply_transform(origin_matrix(container))
        vertices.append(mesh.vertices)
    return np.concatenate(vertices, axis=0) if vertices else np.empty((0, 3))


def hausdorff(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        return float("inf")
    return max(
        float(cKDTree(left).query(right, k=1)[0].max(initial=0.0)),
        float(cKDTree(right).query(left, k=1)[0].max(initial=0.0)),
    )


def inertia(element: etree._Element) -> np.ndarray:
    return np.array(
        [
            [float(element.get("ixx")), float(element.get("ixy")), float(element.get("ixz"))],
            [float(element.get("ixy")), float(element.get("iyy")), float(element.get("iyz"))],
            [float(element.get("ixz")), float(element.get("iyz")), float(element.get("izz"))],
        ]
    )


def validate(repo: Path, samples: int, tolerance: float) -> dict:
    left_path = repo / "data/data_urdf/robot/allegro/allegro_hand_left_extended.urdf"
    right_path = repo / "data/data_urdf/robot/allegro_right/allegro_hand_right_extended.urdf"
    left_root = etree.parse(str(left_path)).getroot()
    right_root = etree.parse(str(right_path)).getroot()

    left_model = URDF.from_xml_file(str(left_path))
    right_model = URDF.from_xml_file(str(right_path))
    structural = {
        "links_equal": len(left_model.links) == len(right_model.links),
        "joints_equal": len(left_model.joints) == len(right_model.joints),
        "joint_names_equal": [j.name for j in left_model.joints] == [j.name for j in right_model.joints],
    }

    left_xml_joints = {joint.get("name"): joint for joint in left_root.findall("joint")}
    right_xml_joints = {joint.get("name"): joint for joint in right_root.findall("joint")}
    joint_limit_error = 0.0
    for name, left_joint in left_xml_joints.items():
        right_joint = right_xml_joints[name]
        for key in ("lower", "upper", "effort", "velocity"):
            left_value = left_joint.find("limit")
            right_value = right_joint.find("limit")
            if left_value is not None and right_value is not None and left_value.get(key) is not None:
                joint_limit_error = max(
                    joint_limit_error,
                    abs(float(left_value.get(key)) - float(right_value.get(key))),
                )

    left_links = {link.get("name"): link for link in left_root.findall("link")}
    right_links = {link.get("name"): link for link in right_root.findall("link")}
    visual_mesh_error = 0.0
    collision_mesh_error = 0.0
    inertia_error = 0.0
    inertial_origin_error = 0.0
    mass_error = 0.0
    for name, left_link in left_links.items():
        right_link = right_links[name]
        for container_name in ("visual", "collision"):
            left_vertices = link_geometry_vertices(
                left_path.parent, left_link, container_name
            )
            right_vertices = link_geometry_vertices(
                right_path.parent, right_link, container_name
            )
            if len(left_vertices):
                reflected = left_vertices.copy()
                reflected[:, 1] *= -1
                error = hausdorff(reflected, right_vertices)
                if container_name == "visual":
                    visual_mesh_error = max(visual_mesh_error, error)
                else:
                    collision_mesh_error = max(collision_mesh_error, error)
        left_inertia = left_link.find("inertial/inertia")
        right_inertia = right_link.find("inertial/inertia")
        if left_inertia is not None and right_inertia is not None:
            expected = MIRROR[:3, :3] @ inertia(left_inertia) @ MIRROR[:3, :3]
            inertia_error = max(inertia_error, float(np.abs(expected - inertia(right_inertia)).max()))
            left_inertial = left_link.find("inertial")
            right_inertial = right_link.find("inertial")
            inertial_origin_error = max(
                inertial_origin_error,
                float(
                    np.abs(
                        MIRROR @ origin_matrix(left_inertial) @ MIRROR
                        - origin_matrix(right_inertial)
                    ).max()
                ),
            )
            mass_error = max(
                mass_error,
                abs(
                    float(left_inertial.find("mass").get("value"))
                    - float(right_inertial.find("mass").get("value"))
                ),
            )

    left_chain = pk.build_chain_from_urdf(left_path.read_text()).to(dtype=torch.float64)
    right_chain = pk.build_chain_from_urdf(right_path.read_text()).to(dtype=torch.float64)
    joint_names_equal = left_chain.get_joint_parameter_names() == right_chain.get_joint_parameter_names()
    lower, upper = left_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=torch.float64)
    upper = torch.as_tensor(upper, dtype=torch.float64)
    generator = torch.Generator().manual_seed(20260802)
    kinematic_error = 0.0
    for _ in range(samples):
        q = lower + torch.rand(len(lower), generator=generator, dtype=torch.float64) * (upper - lower)
        q[:6] = 0.0
        left_fk = left_chain.forward_kinematics(q)
        right_fk = right_chain.forward_kinematics(q)
        for name in left_fk:
            left_matrix = left_fk[name].get_matrix()[0].detach().cpu().numpy()
            right_matrix = right_fk[name].get_matrix()[0].detach().cpu().numpy()
            expected = MIRROR @ left_matrix @ MIRROR
            kinematic_error = max(kinematic_error, float(np.abs(expected - right_matrix).max()))

    left_pc = torch.load(repo / "data/PointCloud/robot/allegro.pt", map_location="cpu", weights_only=False)
    right_pc = torch.load(repo / "data/PointCloud/robot/allegro_right.pt", map_location="cpu", weights_only=False)
    point_cloud_error = 0.0
    point_cloud_structure_equal = left_pc.keys() == right_pc.keys()
    for group_name, left_group in left_pc.items():
        right_group = right_pc[group_name]
        point_cloud_structure_equal &= left_group.keys() == right_group.keys()
        for link_name, left_points in left_group.items():
            expected = left_points.clone()
            expected[..., 1] *= -1
            point_cloud_error = max(
                point_cloud_error,
                float((expected - right_group[link_name]).abs().max()),
            )

    checks = {
        **structural,
        "kinematic_joint_names_equal": joint_names_equal,
        "point_cloud_structure_equal": point_cloud_structure_equal,
        "joint_limit_max_error": joint_limit_error,
        "kinematic_mirror_max_error": kinematic_error,
        "visual_mesh_hausdorff_max_m": visual_mesh_error,
        "collision_mesh_hausdorff_max_m": collision_mesh_error,
        "inertia_mirror_max_error": inertia_error,
        "inertial_origin_mirror_max_error": inertial_origin_error,
        "mass_max_error_kg": mass_error,
        "point_cloud_mirror_max_error_m": point_cloud_error,
    }
    passed = (
        all(value for key, value in checks.items() if isinstance(value, bool))
        and joint_limit_error <= tolerance
        and kinematic_error <= tolerance
        and visual_mesh_error <= tolerance
        and collision_mesh_error <= tolerance
        and inertia_error <= tolerance
        and inertial_origin_error <= tolerance
        and mass_error <= tolerance
        and point_cloud_error <= tolerance
    )
    return {
        "passed": passed,
        "mirror_plane": "palm-local y=0",
        "random_fk_samples": samples,
        "tolerance": tolerance,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = validate(repo, args.samples, args.tolerance)
    report_path = repo / "data/data_urdf/robot/allegro_right/VALIDATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
