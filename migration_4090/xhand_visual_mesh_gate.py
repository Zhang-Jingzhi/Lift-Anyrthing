#!/usr/bin/env python3
"""Exact visual-mesh penetration gate for Tianji dual-XHand samples."""

import copy
from pathlib import Path

import numpy as np
import torch
import trimesh
from yourdfpy import URDF


URDF_PATH = Path(
    "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/"
    "xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"
)


def _xyzw_matrix(quaternion):
    x, y, z, w = map(float, quaternion)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-16:
        return np.eye(3)
    s = 2.0 / norm
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ]
    )


def materialize_actual_closure_state(sample, strict_report):
    """Return the pose that PhysX actually reached, not its controller target."""
    result = copy.deepcopy(sample)
    both = strict_report["runs"]["both"]
    actual = both["actual_dof_positions_after_closure"]
    values = dict(zip(result["joint_names"], result["full_body_q"]))
    missing = sorted(set(actual) - set(values))
    if missing:
        raise RuntimeError(f"actual PhysX joints absent from sample: {missing}")
    result["commanded_full_body_q"] = copy.deepcopy(result["full_body_q"])
    values.update({name: float(value) for name, value in actual.items()})
    result["full_body_q"] = [float(values[name]) for name in result["joint_names"]]

    object_pose = np.asarray(result["object_pose_world"], dtype=np.float64).copy()
    closure_pose = both.get("object_root_pose_after_closure_world")
    if closure_pose is not None and len(closure_pose) == 7:
        object_pose[:3, :3] = _xyzw_matrix(closure_pose[3:7])
        object_pose[:3, 3] = np.asarray(closure_pose[:3], dtype=np.float64)
    else:
        object_pose[:3, 3] = np.asarray(
            both["object_root_position_after_closure_world"], dtype=np.float64
        )
    result["commanded_object_pose_world"] = copy.deepcopy(result["object_pose_world"])
    result["object_pose_world"] = object_pose.tolist()
    result["actual_closure_state_source"] = {
        "strict_report": strict_report.get("output"),
        "max_commanded_actual_dof_error_rad": float(
            both["max_dof_position_error_after_closure"]
        ),
    }
    if "isaac_robot_root_alignment_offset_xyz" in both:
        result["isaac_robot_root_alignment_offset_xyz"] = list(
            map(float, both["isaac_robot_root_alignment_offset_xyz"])
        )
    return result


def _hand_surface_points(robot, side):
    points = []
    for node in robot.scene.graph.nodes_geometry:
        transform, geom_name = robot.scene.graph.get(node)
        geometry = robot.scene.geometry[geom_name]
        tag = f"{node} {geom_name}".lower()
        if (
            not isinstance(geometry, trimesh.Trimesh)
            or len(geometry.faces) == 0
            or f"{side}_hand" not in tag
        ):
            continue
        vertices = np.asarray(geometry.vertices)
        centers = np.asarray(geometry.triangles_center)
        local = np.concatenate((vertices, centers), axis=0)
        points.append(trimesh.transform_points(local, transform))
    if not points:
        raise RuntimeError(f"no {side} XHand visual meshes found in {URDF_PATH}")
    return np.concatenate(points, axis=0)


def evaluate_visual_mesh_penetration(
    sample, max_depth_mm=2.0, max_inside_fraction=0.01
):
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    robot.update_cfg(
        np.asarray([values[name] for name in robot.actuated_joint_names], dtype=np.float64)
    )
    obj = trimesh.load_mesh(sample["object_mesh_path"], force="mesh", process=False)
    obj.apply_transform(np.asarray(sample["object_pose_world"], dtype=np.float64))
    if not obj.is_watertight:
        raise RuntimeError(f"object mesh is not watertight: {sample['object_mesh_path']}")

    sides = {}
    # Isaac applies a fixed robot-root alignment after loading the URDF.
    # Candidate poses and the object are stored in world coordinates, so apply
    # the same translation to URDF mesh points before testing containment.
    root_alignment = np.asarray(sample.get("isaac_robot_root_alignment_offset_xyz", [0.0027, 0.00042, 0.0]), dtype=np.float64)
    for side in ("left", "right"):
        points = _hand_surface_points(robot, side) + root_alignment
        inside = obj.contains(points)
        inside_points = points[inside]
        if len(inside_points):
            _, distances, _ = trimesh.proximity.closest_point(obj, inside_points)
            max_depth = float(np.max(distances) * 1000.0)
            p95_depth = float(np.percentile(distances, 95) * 1000.0)
        else:
            max_depth = 0.0
            p95_depth = 0.0
        fraction = float(np.mean(inside))
        sides[side] = {
            "surface_point_count": int(len(points)),
            "inside_point_count": int(np.sum(inside)),
            "inside_fraction": fraction,
            "max_penetration_depth_mm": max_depth,
            "p95_penetration_depth_mm": p95_depth,
            "pass": bool(
                max_depth <= float(max_depth_mm)
                and fraction <= float(max_inside_fraction)
            ),
        }
    return {
        "schema": "xhand_visual_mesh_penetration_v1",
        "urdf": str(URDF_PATH),
        "object_mesh": str(sample["object_mesh_path"]),
        "max_allowed_depth_mm": float(max_depth_mm),
        "max_allowed_inside_fraction": float(max_inside_fraction),
        "sides": sides,
        "pass": bool(all(row["pass"] for row in sides.values())),
    }


def load_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False)
