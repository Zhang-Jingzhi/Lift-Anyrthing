#!/usr/bin/env python3
"""Run the pinned upstream BODex solver on paired dual-XHand surface seeds.

This is the production boundary between mesh-side paired initialization and
the validated bank exporter.  Both palms and all configured contacts are
optimized in one BODex problem.  A grasp is emitted only when BODex marks it
successful and a collision-aware, two-palm lift IK can be solved from that
same grasp.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterable

import numpy as np
import torch
import yaml

from .bodex_adapter import (
    configure_joint_bimanual_seed,
    validate_bodex_result_shapes,
)
from .build_robot_config import select_contact_point_sphere_index
from .contracts import sha256_file, verify_bodex_checkout
from .paired_surface import PairedSurfaceSeed


PALM_LINKS = ("left_hand_ee_link", "right_hand_ee_link")
CONTACT_MESH_LINKS_BY_SIDE = {
    side: tuple(
        f"{side}_hand_{suffix}"
        for suffix in (
            "index_rota_link2",
            "mid_link2",
            "ring_link2",
            "pinky_link2",
            "thumb_rota_link2",
        )
    )
    for side in ("left", "right")
}
CONTACT_POINT_LINKS_BY_SIDE = {
    side: tuple(
        f"{side}_hand_{suffix}"
        for suffix in (
            "index_rota_tip",
            "mid_tip",
            "ring_tip",
            "pinky_tip",
            "thumb_rota_tip",
        )
    )
    for side in ("left", "right")
}
ARM_JOINT_NAMES = tuple(
    f"{side}_j{index}" for side in ("left", "right") for index in range(1, 8)
)
SELF_COLLISION_SPHERE_OVERRIDE_SCHEMA = (
    "xhand_bodex_self_collision_sphere_overrides_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _quaternion_matrix_wxyz(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quaternion), dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_pair_to_world(
    pair: PairedSurfaceSeed,
    *,
    translation_xyz: Iterable[float],
    quaternion_wxyz: Iterable[float],
) -> PairedSurfaceSeed:
    """Transform one object-local paired seed into BODex world coordinates."""

    translation = np.asarray(tuple(translation_xyz), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must contain three finite xyz values")
    rotation = _quaternion_matrix_wxyz(quaternion_wxyz)

    def point(values: Iterable[float]) -> tuple[float, float, float]:
        transformed = rotation @ np.asarray(tuple(values), dtype=np.float64) + translation
        return tuple(float(value) for value in transformed)

    def rotation_6d(values: Iterable[float]) -> tuple[float, ...]:
        raw = np.asarray(tuple(values), dtype=np.float64)
        if raw.shape != (6,):
            raise ValueError("rotation representation must have six values")
        transformed = np.concatenate((rotation @ raw[:3], rotation @ raw[3:]))
        return tuple(float(value) for value in transformed)

    return replace(
        pair,
        left_position_m=point(pair.left_position_m),
        right_position_m=point(pair.right_position_m),
        left_rotation_6d=rotation_6d(pair.left_rotation_6d),
        right_rotation_6d=rotation_6d(pair.right_rotation_6d),
    )


def load_paired_seeds(path: str | Path) -> tuple[dict[str, Any], list[PairedSurfaceSeed]]:
    payload = json.loads(Path(path).read_text())
    _require(
        payload.get("schema")
        in {
            "xhand_bodex_paired_surface_seeds_v1",
            "xhand_bodex_paired_surface_seeds_v2",
        },
        "wrong paired-seed schema",
    )
    rows = payload.get("seeds")
    _require(isinstance(rows, list) and rows, "paired-seed file is empty")
    seeds = []
    for row in rows:
        _require(isinstance(row, dict), "paired seed must be a mapping")
        seeds.append(
            PairedSurfaceSeed(
                left_position_m=tuple(float(value) for value in row["left_position_m"]),
                right_position_m=tuple(float(value) for value in row["right_position_m"]),
                left_rotation_6d=tuple(float(value) for value in row["left_rotation_6d"]),
                right_rotation_6d=tuple(float(value) for value in row["right_rotation_6d"]),
                opposition_cosine=float(row["opposition_cosine"]),
                span_m=float(row["span_m"]),
                source_surface_indices=tuple(
                    int(value) for value in row["source_surface_indices"]
                ),
                lateral_fraction=float(row.get("lateral_fraction", 0.0)),
                side_normal_min_component=float(
                    row.get("side_normal_min_component", 0.0)
                ),
                x_offset_m=float(row.get("x_offset_m", 0.0)),
                z_offset_m=float(row.get("z_offset_m", 0.0)),
            )
        )
    _require(int(payload.get("count", len(seeds))) == len(seeds), "paired-seed count mismatch")
    return payload, seeds


def pair_window_indices(
    total_count: int,
    *,
    start: int = 0,
    limit: int | None = None,
) -> list[int]:
    """Return stable source indices for one non-overlapping solver shard."""

    if total_count <= 0:
        raise ValueError("paired-seed collection must be non-empty")
    if start < 0:
        raise ValueError("pair-start must be non-negative")
    if start >= total_count:
        raise ValueError(
            f"pair-start {start} is outside paired-seed collection of size {total_count}"
        )
    if limit is not None and limit <= 0:
        raise ValueError("max-pairs must be positive")
    stop = total_count if limit is None else min(total_count, start + limit)
    return list(range(start, stop))


def load_robot_seed_data(
    robot_config_path: str | Path,
    *,
    tensor_args: Any | None = None,
) -> tuple[list[str], list[float], dict[str, list[dict[str, Any]]]]:
    payload = yaml.safe_load(Path(robot_config_path).read_text())
    kinematics = payload.get("robot_cfg", {}).get("kinematics", {})
    cspace = kinematics.get("cspace", {})
    names = list(cspace.get("joint_names", []))
    default_values = cspace.get("retract_config", cspace.get("default_joint_position", []))
    default = [float(value) for value in default_values]
    _require(len(names) == 38, f"XHand BODex config must expose 38 joints, got {len(names)}")
    _require(len(default) == len(names), "robot default posture does not match joint names")
    _require(len(set(names)) == len(names), "robot config contains duplicate joint names")
    if tensor_args is not None:
        # CudaRobotGenerator reindexes cspace tensors to URDF topological
        # order. Seeder q and emitted joint_names must use that same internal
        # order; using the YAML order silently assigns finger values to the
        # wrong joints on XHand.
        from curobo.types.robot import RobotConfig

        robot_config = RobotConfig.from_dict(payload["robot_cfg"], tensor_args)
        cspace = robot_config.kinematics.cspace
        names = list(cspace.joint_names)
        default = [float(value) for value in cspace.retract_config.detach().cpu().tolist()]
    spheres = kinematics.get("collision_spheres", {})
    _require(isinstance(spheres, dict), "robot config lacks collision spheres")
    return names, default, spheres


def load_ik_initial_pose(
    path: str | Path,
    *,
    joint_names: list[str],
    fallback_q: list[float],
) -> tuple[list[float], dict[str, Any]]:
    """Load a single named posture for BODex's supported ``ik_init_q`` hook.

    This is an IK warm start, not a grasp candidate bank: unspecified joints
    retain the robot-config retract values and every emitted sample must still
    pass the joint BODex optimization and lift checks.
    """

    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    _require(
        payload.get("schema") == "xhand_bodex_bimanual_ik_seed_v1",
        "wrong IK seed schema",
    )
    positions = payload.get("joint_positions")
    _require(isinstance(positions, dict) and positions, "IK seed has no joint positions")
    unknown = sorted(set(positions) - set(joint_names))
    _require(not unknown, f"IK seed contains unknown joints: {unknown}")
    seeded = list(fallback_q)
    index = {name: i for i, name in enumerate(joint_names)}
    for name, value in positions.items():
        numeric = float(value)
        _require(math.isfinite(numeric), f"IK seed joint is non-finite: {name}")
        seeded[index[name]] = numeric
    return seeded, {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "schema": payload["schema"],
        "seeded_joint_count": len(positions),
        "role": "single_collision_aware_ik_warm_start_not_output_data",
        "note": payload.get("note"),
    }


def load_coordinate_seed_replay(
    path: str | Path,
    *,
    object_name: str,
    robot_config_path: str | Path,
    joint_names: list[str],
) -> tuple[dict[int, list[list[float]]], dict[str, Any]]:
    """Load previously generated joint-BODex coordinate seeds for exact replay."""

    replay_path = Path(path).resolve()
    payload = json.loads(replay_path.read_text())
    _require(
        payload.get("schema") == "xhand_bodex_rejected_solver_diagnostics_v1",
        "coordinate-seed replay has the wrong schema",
    )
    _require(payload.get("object") == object_name, "coordinate-seed replay object mismatch")
    _require(
        payload.get("robot_config_sha256") == sha256_file(robot_config_path),
        "coordinate-seed replay robot config mismatch",
    )
    seeds_by_pair: dict[int, list[list[float]]] = {}
    for row in payload.get("rows", []):
        _require(row.get("joint_names") == joint_names, "coordinate-seed joint order mismatch")
        pair_index = int(row.get("pair_index", -1))
        seeds = row.get("coordinate_seed_q")
        _require(pair_index >= 0 and isinstance(seeds, list) and seeds, "invalid replay row")
        normalized: list[list[float]] = []
        for seed in seeds:
            values = [float(value) for value in seed]
            _require(len(values) == len(joint_names), "coordinate seed replay has wrong width")
            _require(all(math.isfinite(value) for value in values), "non-finite coordinate seed")
            normalized.append(values)
        seeds_by_pair[pair_index] = normalized
    _require(seeds_by_pair, "coordinate-seed replay contains no rows")
    return seeds_by_pair, {
        "schema": "xhand_bodex_coordinate_seed_replay_v1",
        "path": str(replay_path),
        "sha256": sha256_file(replay_path),
        "source_schema": payload["schema"],
        "joint_bimanual_seed_source": True,
        "manual_pose_source": False,
    }


def load_self_collision_sphere_overrides(
    path: str | Path,
    *,
    robot_config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load mesh-validated, config-hash-locked collision-sphere exceptions."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    _require(
        payload.get("schema") == SELF_COLLISION_SPHERE_OVERRIDE_SCHEMA,
        "wrong self-collision sphere override schema",
    )
    robot_hash = sha256_file(robot_config_path)
    _require(
        payload.get("robot_config_sha256") == robot_hash,
        "self-collision sphere overrides target a different robot config",
    )
    rows = payload.get("overrides")
    _require(isinstance(rows, list) and rows, "self-collision sphere override file is empty")
    normalized = []
    seen: set[tuple[tuple[str, int], tuple[str, int]]] = set()
    for row in rows:
        _require(isinstance(row, dict), "self-collision sphere override must be a mapping")
        endpoints = []
        for key in ("first", "second"):
            endpoint = row.get(key)
            _require(isinstance(endpoint, dict), f"override endpoint is missing: {key}")
            link = endpoint.get("link")
            sphere_index = endpoint.get("sphere_index")
            _require(isinstance(link, str) and link, "override link must be a non-empty string")
            _require(
                isinstance(sphere_index, int) and sphere_index >= 0,
                "override sphere index must be non-negative",
            )
            endpoints.append((link, sphere_index))
        pair_key = tuple(sorted(endpoints))
        _require(pair_key not in seen, "duplicate self-collision sphere override")
        seen.add(pair_key)
        validation = row.get("mesh_validation")
        _require(isinstance(validation, dict), "override lacks real-mesh validation")
        _require(
            validation.get("surface_intersection") is False,
            "override is not validated as a collision-sphere false positive",
        )
        _require(
            float(validation.get("minimum_clearance_m", 0.0)) > 0.0,
            "override real-mesh clearance must be positive",
        )
        normalized.append(
            {
                "first": {"link": endpoints[0][0], "sphere_index": endpoints[0][1]},
                "second": {"link": endpoints[1][0], "sphere_index": endpoints[1][1]},
                "reason": row.get("reason"),
                "mesh_validation": validation,
            }
        )
    link_rows = payload.get("link_overrides", [])
    _require(isinstance(link_rows, list), "link overrides must be a list")
    normalized_links = []
    seen_links: set[tuple[str, str]] = set()
    for row in link_rows:
        _require(isinstance(row, dict), "self-collision link override must be a mapping")
        first_link = row.get("first_link")
        second_link = row.get("second_link")
        _require(
            isinstance(first_link, str) and first_link,
            "first override link must be a non-empty string",
        )
        _require(
            isinstance(second_link, str) and second_link,
            "second override link must be a non-empty string",
        )
        pair_key = tuple(sorted((first_link, second_link)))
        _require(pair_key not in seen_links, "duplicate self-collision link override")
        seen_links.add(pair_key)
        validation = row.get("mesh_validation")
        _require(isinstance(validation, dict), "link override lacks real-mesh validation")
        _require(
            validation.get("surface_intersection") is False,
            "link override is not validated as a collision-sphere false positive",
        )
        _require(
            float(validation.get("minimum_clearance_m", 0.0)) > 0.0,
            "link override real-mesh clearance must be positive",
        )
        normalized_links.append(
            {
                "first_link": first_link,
                "second_link": second_link,
                "reason": row.get("reason"),
                "mesh_validation": validation,
                "requires_postsolve_mesh_validation": True,
            }
        )
    return payload, normalized, normalized_links


def apply_self_collision_link_overrides(
    robot_config: Any,
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove a validated link pair while retaining all other self checks."""

    model_config = robot_config.kinematics
    kinematics = model_config.kinematics_config
    self_collision = model_config.self_collision_config
    resolved = []
    for row in overrides:
        first_indices = kinematics.get_sphere_index_from_link_name(row["first_link"])
        second_indices = kinematics.get_sphere_index_from_link_name(row["second_link"])
        first_set = set(int(value) for value in first_indices.detach().cpu().tolist())
        second_set = set(int(value) for value in second_indices.detach().cpu().tolist())
        if self_collision.experimental_kernel:
            entry_count = int(self_collision.thread_max)
            checked = self_collision.thread_location[:entry_count].reshape(-1, 2)
            first_tensor = torch.as_tensor(
                sorted(first_set), device=checked.device, dtype=checked.dtype
            )
            second_tensor = torch.as_tensor(
                sorted(second_set), device=checked.device, dtype=checked.dtype
            )
            matching = (
                torch.isin(checked[:, 0], first_tensor)
                & torch.isin(checked[:, 1], second_tensor)
            ) | (
                torch.isin(checked[:, 0], second_tensor)
                & torch.isin(checked[:, 1], first_tensor)
            )
            removed_count = int(matching.sum().item())
            _require(removed_count > 0, "override link pair is not checked")
            filtered = checked[~matching].reshape(-1)
            self_collision.thread_location.fill_(-1)
            self_collision.thread_location[: filtered.numel()] = filtered
            self_collision.thread_max = int(filtered.numel())
            backend = "experimental_thread_locations"
        else:
            matrix = self_collision.collision_matrix
            first_tensor = torch.as_tensor(sorted(first_set), device=matrix.device)
            second_tensor = torch.as_tensor(sorted(second_set), device=matrix.device)
            submatrix = matrix[first_tensor[:, None], second_tensor[None, :]]
            removed_count = int((submatrix > 0).sum().item())
            _require(removed_count > 0, "override link pair is not checked")
            matrix[first_tensor[:, None], second_tensor[None, :]] = 0
            matrix[second_tensor[:, None], first_tensor[None, :]] = 0
            backend = "fallback_collision_matrix"
        resolved.append(
            {
                **row,
                "first_global_sphere_indices": sorted(first_set),
                "second_global_sphere_indices": sorted(second_set),
                "removed_checked_sphere_pair_count": removed_count,
                "collision_backend": backend,
                "scope": "one_exact_link_pair_with_postsolve_mesh_validation",
            }
        )
    return resolved


def apply_self_collision_sphere_overrides(
    robot_config: Any,
    overrides: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove only validated sphere pairs from cuRobo's collision checks."""

    model_config = robot_config.kinematics
    kinematics = model_config.kinematics_config
    self_collision = model_config.self_collision_config
    resolved = []
    for row in overrides:
        global_indices = []
        for endpoint in (row["first"], row["second"]):
            link_indices = kinematics.get_sphere_index_from_link_name(endpoint["link"])
            local_index = int(endpoint["sphere_index"])
            _require(
                local_index < int(link_indices.numel()),
                f"override sphere does not exist: {endpoint['link']}/{local_index}",
            )
            global_indices.append(int(link_indices[local_index].item()))
        first, second = global_indices
        if self_collision.experimental_kernel:
            entry_count = int(self_collision.thread_max)
            checked = self_collision.thread_location[:entry_count].reshape(-1, 2)
            matching = ((checked[:, 0] == first) & (checked[:, 1] == second)) | (
                (checked[:, 0] == second) & (checked[:, 1] == first)
            )
            _require(bool(matching.any().item()), "override sphere pair is not checked")
            filtered = checked[~matching].reshape(-1)
            self_collision.thread_location.fill_(-1)
            self_collision.thread_location[: filtered.numel()] = filtered
            self_collision.thread_max = int(filtered.numel())
            backend = "experimental_thread_locations"
        else:
            matrix = self_collision.collision_matrix
            _require(
                bool((matrix[first, second] > 0).item())
                or bool((matrix[second, first] > 0).item()),
                "override sphere pair is not checked",
            )
            matrix[first, second] = 0
            matrix[second, first] = 0
            backend = "fallback_collision_matrix"
        resolved.append(
            {
                **row,
                "global_sphere_indices": global_indices,
                "collision_backend": backend,
                "scope": "one_exact_sphere_pair_only",
            }
        )
    return resolved


def exclude_nonphysical_contact_points_from_self_collision(
    robot_config: Any,
    contact_point_links: Iterable[str],
) -> dict[str, Any]:
    """Remove marker-only spheres from self collision, retaining physical checks."""

    model_config = robot_config.kinematics
    kinematics = model_config.kinematics_config
    self_collision = model_config.self_collision_config
    links = list(contact_point_links)
    excluded: set[int] = set()
    for link in links:
        indices = kinematics.get_sphere_index_from_link_name(link)
        _require(indices.numel() > 0, f"contact-point marker is not loaded: {link}")
        excluded.update(int(value) for value in indices.detach().cpu().tolist())
    excluded_tensor: torch.Tensor
    if self_collision.experimental_kernel:
        entry_count = int(self_collision.thread_max)
        checked = self_collision.thread_location[:entry_count].reshape(-1, 2)
        excluded_tensor = torch.as_tensor(
            sorted(excluded), device=checked.device, dtype=checked.dtype
        )
        matching = torch.isin(checked, excluded_tensor).any(dim=1)
        removed_count = int(matching.sum().item())
        filtered = checked[~matching].reshape(-1)
        self_collision.thread_location.fill_(-1)
        self_collision.thread_location[: filtered.numel()] = filtered
        self_collision.thread_max = int(filtered.numel())
        backend = "experimental_thread_locations"
    else:
        matrix = self_collision.collision_matrix
        excluded_tensor = torch.as_tensor(sorted(excluded), device=matrix.device)
        before = int(torch.triu(matrix, diagonal=1).sum().item())
        matrix[excluded_tensor, :] = 0
        matrix[:, excluded_tensor] = 0
        after = int(torch.triu(matrix, diagonal=1).sum().item())
        removed_count = before - after
        backend = "fallback_collision_matrix"
    _require(removed_count > 0, "contact-point markers had no self-collision checks")
    return {
        "contact_point_links": links,
        "global_sphere_indices": sorted(excluded),
        "removed_checked_sphere_pair_count": removed_count,
        "collision_backend": backend,
        "scope": "nonphysical_measurement_points_only",
        "physical_self_collision_remains_enabled": True,
    }


def contact_point_names(
    collision_spheres: dict[str, list[dict[str, Any]]],
    links: Iterable[str],
) -> list[str]:
    """Select point-marker spheres without treating marker meshes as contact."""

    names = []
    for link in links:
        spheres = collision_spheres.get(link)
        _require(isinstance(spheres, list) and spheres, f"contact link has no spheres: {link}")
        selected = select_contact_point_sphere_index(spheres)
        names.append(f"{link}/{selected}")
    return names


def base_manip_config(
    *,
    robot_config_path: str | Path,
    initial_q: list[float],
    seed_num: int,
    translation_jitter_m: float,
    rotation_jitter_deg: float,
    max_ge_stage: int = 0,
) -> dict[str, Any]:
    if seed_num <= 0:
        raise ValueError("seed_num must be positive")
    if min(translation_jitter_m, rotation_jitter_deg) < 0.0:
        raise ValueError("jitter bounds must be non-negative")
    if max_ge_stage not in (0, 1, 2):
        raise ValueError("max-ge-stage must be one of 0, 1, or 2")
    resolved_robot_config = Path(robot_config_path).resolve()
    return {
        "robot_file": str(resolved_robot_config),
        "robot_file_with_arm": str(resolved_robot_config),
        "base_cfg_file": "base_grasp.yml",
        "particle_file": "particle_grasp_debug.yml",
        "gradient_file": "gradient_grasp_fc.yml",
        "seed_num": int(seed_num),
        "seeder_cfg": {
            "obj_sample": {
                "num": 128,
                "inflate": 0.1,
                "convex_hull": True,
                "collision_free": True,
            },
            "ik_init_q": None,
            "load_path": None,
            "skip_transfer": False,
            "t": None,
            "r": None,
            "q": list(initial_q),
            "jitter_angle": [
                [-rotation_jitter_deg] * 3,
                [rotation_jitter_deg] * 3,
            ],
            "jitter_dist": [
                [-translation_jitter_m] * 3,
                [translation_jitter_m] * 3,
            ],
        },
        "grasp_contact_strategy": {
            "contact_points_name": [],
            "opt_progress": [0.0, 0.6, 0.8],
            "distance": [0.01, 0.01, 0.0],
            "contact_query_mode": [-1, 0, 0],
            "save_qpos": [False, True],
            "max_ge_stage": int(max_ge_stage),
        },
        "grasp_cfg": {
            "task_dict": {
                "f": [0, 0, 1],
                "p": [0, 0, 0],
                "t": [0, 0, 0],
                "gamma": 180,
            },
            "ge_param": {
                "type": "qp",
                "miu_coef": [0.1, 0.0],
                "solver_type": "batch_reluqp",
                "k_lower": 0.2,
                "pressure_constraints": [],
                "enable_density": False,
                "solve_interval": 5,
            },
        },
    }


def _object_pose_matrix(position: Iterable[float], quaternion: Iterable[float]) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_matrix_wxyz(quaternion)
    matrix[:3, 3] = np.asarray(tuple(position), dtype=np.float64)
    return matrix.tolist()


def _world_model(
    *,
    object_name: str,
    mesh_path: Path,
    object_urdf_path: Path,
    position: list[float],
    quaternion: list[float],
) -> dict[str, Any]:
    return {
        "mesh": {
            object_name: {
                "pose": [*position, *quaternion],
                "scale": [1.0, 1.0, 1.0],
                "file_path": str(mesh_path.resolve()),
                "urdf_path": str(object_urdf_path.resolve()),
            }
        }
    }


def materialize_single_mesh_urdf(mesh_path: Path, output_path: Path) -> Path:
    """Create the one-link URDF required by BODex's convex-mesh loader."""

    mesh = html.escape(str(mesh_path.resolve()), quote=True)
    content = (
        "<?xml version=\"1.0\"?>\n"
        "<robot name=\"xhand_bodex_object\">\n"
        "  <link name=\"object\">\n"
        "    <visual>\n"
        "      <geometry>\n"
        f"        <mesh filename=\"{mesh}\" scale=\"1 1 1\"/>\n"
        "      </geometry>\n"
        "    </visual>\n"
        "  </link>\n"
        "</robot>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        _require(output_path.read_text() == content, f"object URDF conflicts with existing file: {output_path}")
    else:
        output_path.write_text(content)
    return output_path


def materialize_fixed_base_gradient_config(
    bodex_root: Path,
    output_path: Path,
    *,
    retain_best: bool = False,
    joint_names: list[str] | None = None,
    arm_seed_regularization: float = 0.0,
) -> Path:
    """Derive BODex's grasp config for a fixed-base full-body robot.

    The upstream three-value ``base_scale`` is meaningful only when the robot
    has a seven-DoF floating root.  The fixed-base optimizer uses its scalar
    line-search scale.  Upstream's ``retain_best=False`` remains the default:
    BODex changes its grasp objective at scheduled contact stages, so costs
    from an early stage are not comparable to costs after the switch.
    """

    source = (
        bodex_root
        / "src/curobo/content/configs/task/gradient_grasp_fc.yml"
    )
    config = yaml.safe_load(source.read_text())
    config["lbfgs"]["base_scale"] = None
    config["lbfgs"]["retain_best"] = bool(retain_best)
    if arm_seed_regularization < 0.0:
        raise ValueError("arm seed regularization must be non-negative")
    if arm_seed_regularization > 0.0:
        _require(joint_names is not None, "joint names are required for arm regularization")
        config["cost"]["bound_cfg"]["null_space_weight"] = [
            float(arm_seed_regularization) if "_hand_" not in name else 0.0
            for name in joint_names
        ]
    content = yaml.safe_dump(config, sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        _require(
            output_path.read_text() == content,
            f"fixed-base gradient config conflicts with existing file: {output_path}",
        )
    else:
        output_path.write_text(content)
    return output_path


def joint_group_line_search_multipliers(
    joint_names: list[str],
    *,
    arm_multiplier: float,
    hand_multiplier: float,
) -> list[float]:
    """Return the fixed-base BODex line-search multiplier for every DoF.

    Upstream BODex uses separate floating-root and qpose scales. XHand has no
    floating root, so cuRobo otherwise applies one scalar to all 38 joints and
    lets the 14 arm joints move as aggressively as the 24 finger joints. The
    explicit canonical-name checks below prevent a reordered or changed robot
    ABI from silently receiving the wrong scale.
    """

    if not math.isfinite(arm_multiplier) or arm_multiplier <= 0.0:
        raise ValueError("arm line-search multiplier must be finite and positive")
    if not math.isfinite(hand_multiplier) or hand_multiplier <= 0.0:
        raise ValueError("hand line-search multiplier must be finite and positive")
    _require(len(joint_names) == 38, f"expected 38 XHand joints, got {len(joint_names)}")
    _require(len(set(joint_names)) == len(joint_names), "joint names contain duplicates")

    arm_names = set(ARM_JOINT_NAMES)
    observed_arm_names = {name for name in joint_names if name in arm_names}
    hand_names = {
        name
        for name in joint_names
        if name.startswith("left_hand_") or name.startswith("right_hand_")
    }
    unknown = sorted(set(joint_names) - observed_arm_names - hand_names)
    _require(not unknown, f"cannot classify XHand optimizer joints: {unknown}")
    _require(
        observed_arm_names == arm_names,
        f"XHand arm joint ABI mismatch: missing={sorted(arm_names - observed_arm_names)}",
    )
    _require(len(hand_names) == 24, f"expected 24 XHand hand joints, got {len(hand_names)}")
    return [
        float(arm_multiplier) if name in arm_names else float(hand_multiplier)
        for name in joint_names
    ]


def install_grouped_line_search_scales(
    grasp_config: Any,
    *,
    joint_names: list[str],
    arm_multiplier: float,
    hand_multiplier: float,
) -> dict[str, Any]:
    """Install 14-arm/24-hand scales in BODex's existing LBFGS optimizer.

    This changes only cuRobo's line-search step tensor. The objective,
    gradients, contact schedule, force-closure computation and success gates
    remain the pinned upstream BODex implementation.
    """

    multipliers = joint_group_line_search_multipliers(
        joint_names,
        arm_multiplier=arm_multiplier,
        hand_multiplier=hand_multiplier,
    )
    optimizer = grasp_config.solver.newton_optimizer
    _require(
        not optimizer.cu_opt_init,
        "grouped line-search scales must be installed before optimizer warmup",
    )
    original = optimizer.line_scale.detach().clone()
    _require(original.ndim == 3, "unexpected BODex line-scale tensor rank")
    _require(original.shape[0] == 1, "unexpected BODex line-scale batch dimension")
    _require(
        original.shape[-1] == len(joint_names),
        "BODex line-scale width does not match the 38D joint ABI",
    )
    # Fixed-base cuRobo should expose one scalar per line-search particle.
    # Refuse an already grouped tensor instead of multiplying it twice.
    reference = original[..., :1].expand_as(original)
    _require(
        bool(torch.allclose(original, reference)),
        "fixed-base BODex line scale is already non-uniform across joints",
    )
    multiplier_tensor = torch.as_tensor(
        multipliers,
        device=original.device,
        dtype=original.dtype,
    ).view(1, 1, -1)
    with torch.no_grad():
        optimizer.line_scale.copy_(original * multiplier_tensor)
        optimizer.alpha_list.copy_(
            optimizer.line_scale.repeat(optimizer.n_problems, 1, 1)
        )
        optimizer.zero_alpha_list = optimizer.alpha_list[:, :, 0:1].contiguous()

    arm_indices = [
        index for index, name in enumerate(joint_names) if name in ARM_JOINT_NAMES
    ]
    hand_indices = [
        index for index, name in enumerate(joint_names) if name not in ARM_JOINT_NAMES
    ]
    return {
        "schema": "xhand_bodex_grouped_line_search_v1",
        "arm_multiplier": float(arm_multiplier),
        "hand_multiplier": float(hand_multiplier),
        "arm_joint_indices": arm_indices,
        "hand_joint_indices": hand_indices,
        "base_line_scale_by_particle": original[0, :, 0].detach().cpu().tolist(),
        "effective_arm_scale_by_particle": optimizer.line_scale[
            0, :, arm_indices[0]
        ].detach().cpu().tolist(),
        "effective_hand_scale_by_particle": optimizer.line_scale[
            0, :, hand_indices[0]
        ].detach().cpu().tolist(),
        "note": "per-DoF adapter preserving the pinned BODex LBFGS objective and updates",
    }


def install_mesh_ge_query_refresh(
    grasp_config: Any,
    *,
    max_ge_stage: int,
) -> dict[str, Any]:
    """Keep BODex's mesh distance defined when GE remains active after stage 0.

    The pinned upstream ``GraspCost.forward`` refreshes GJK contact data every
    five calls in mesh-query mode, but stores every refreshed value except the
    local ``raw_dist`` tensor.  Upstream's default ``max_ge_stage=0`` never
    reads that tensor on cached calls.  Raising ``max_ge_stage`` therefore
    triggers ``UnboundLocalError`` on the first cached mesh-query call.

    For the non-default GE schedule, force the existing upstream GJK query to
    refresh on every cost call.  This changes neither the contact geometry nor
    the grasp-energy implementation; it only avoids the invalid cache branch.
    """

    if max_ge_stage not in (0, 1, 2):
        raise ValueError("max-ge-stage must be one of 0, 1, or 2")
    if max_ge_stage == 0:
        return {
            "schema": "xhand_bodex_mesh_ge_query_refresh_v1",
            "enabled": False,
            "reason": "pinned upstream max_ge_stage=0 does not read cached raw_dist",
        }

    rollouts = [getattr(grasp_config, "rollout_fn", None)]
    solver = getattr(grasp_config, "solver", None)
    rollouts.append(getattr(solver, "rollout_fn", None))
    costs: list[Any] = []
    seen: set[int] = set()
    for rollout in rollouts:
        cost = getattr(rollout, "grasp_cost", None)
        if cost is None or id(cost) in seen:
            continue
        seen.add(id(cost))
        costs.append(cost)
    _require(costs, "BODex grasp cost is unavailable for mesh GE query refresh")

    for cost in costs:
        _require(
            not getattr(cost, "_xhand_mesh_ge_query_refresh_installed", False),
            "mesh GE query refresh is already installed",
        )
        original_forward = cost.forward

        def refreshed_forward(
            self: Any,
            *forward_args: Any,
            _original_forward: Any = original_forward,
            **forward_kwargs: Any,
        ) -> Any:
            self.count = 0
            return _original_forward(*forward_args, **forward_kwargs)

        cost.forward = MethodType(refreshed_forward, cost)
        cost._xhand_mesh_ge_query_refresh_installed = True

    return {
        "schema": "xhand_bodex_mesh_ge_query_refresh_v1",
        "enabled": True,
        "max_ge_stage": int(max_ge_stage),
        "patched_cost_instance_count": len(costs),
        "upstream_refresh_interval_calls": 5,
        "effective_refresh_interval_calls": 1,
        "scope": "runtime adapter around pinned BODex GraspCost.forward",
        "source_checkout_modified": False,
    }


def install_reluqp_device_alignment(
    grasp_config: Any,
    *,
    tensor_args: Any,
) -> dict[str, Any]:
    """Keep upstream BODex's lazily-created ReLU-QP buffers on the run device.

    The pinned ``BATCHED_RELUQP.init_problem`` constructs its private
    ``_RELUQP`` without forwarding ``tensor_args``.  ``_RELUQP`` therefore
    uses cuRobo's default ``cuda:0`` even when the surrounding grasp solver is
    explicitly running on another GPU.  The failure is delayed until the
    first QP solve, where ``cuda:0`` work buffers are multiplied by the
    requested-device constraint matrix.

    Wrap the existing instance method and relocate only the newly-created QP
    tensors after every problem initialization.  The pinned BODex source,
    objective, constraints, and solver iterations remain unchanged.
    """

    target_device = torch.device(tensor_args.device)
    rollouts = [getattr(grasp_config, "rollout_fn", None)]
    solver = getattr(grasp_config, "solver", None)
    rollouts.append(getattr(solver, "rollout_fn", None))
    qpsolvers: list[Any] = []
    seen: set[int] = set()
    for rollout in rollouts:
        cost = getattr(rollout, "grasp_cost", None)
        energy = getattr(cost, "GraspEnergy", None)
        qpsolver = getattr(energy, "qpsolver", None)
        if qpsolver is None or id(qpsolver) in seen:
            continue
        seen.add(id(qpsolver))
        qpsolvers.append(qpsolver)
    _require(qpsolvers, "BODex ReLU-QP solver is unavailable for device alignment")

    for qpsolver in qpsolvers:
        _require(
            not getattr(qpsolver, "_xhand_device_alignment_installed", False),
            "ReLU-QP device alignment is already installed",
        )
        original_init_problem = qpsolver.init_problem

        def aligned_init_problem(
            self: Any,
            G_matrix: torch.Tensor,
            l_matrix: torch.Tensor,
            h_matrix: torch.Tensor,
            _original_init_problem: Any = original_init_problem,
        ) -> None:
            _original_init_problem(G_matrix, l_matrix, h_matrix)
            private_solver = getattr(self, "solver", None)
            _require(private_solver is not None, "ReLU-QP private solver was not created")
            for name, value in tuple(vars(private_solver).items()):
                if torch.is_tensor(value):
                    setattr(private_solver, name, value.to(device=target_device))
            private_solver.tensor_args = tensor_args
            for name in ("G_matrix", "l_matrix", "h_matrix"):
                value = getattr(self, name, None)
                _require(
                    torch.is_tensor(value) and value.device == target_device,
                    f"ReLU-QP {name} is not on {target_device}",
                )

        qpsolver.init_problem = MethodType(aligned_init_problem, qpsolver)
        qpsolver._xhand_device_alignment_installed = True

    return {
        "schema": "xhand_bodex_reluqp_device_alignment_v1",
        "enabled": True,
        "target_device": str(target_device),
        "patched_qpsolver_count": len(qpsolvers),
        "relocation_timing": "after_each_lazy_problem_initialization_before_solve",
        "scope": "runtime adapter around pinned BODex BATCHED_RELUQP.init_problem",
        "source_checkout_modified": False,
    }


def install_multiseed_seed_ik(
    grasp_config: Any,
    *,
    robot_config_path: str | Path,
    tensor_args: Any,
    num_seeds: int,
    use_particle_opt: bool,
    grad_iters: int | None,
    self_collision_sphere_overrides: list[dict[str, Any]] | None = None,
    self_collision_link_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replace BODex's one-seed arm IK with the same cuRobo IK in batch mode.

    The pinned BODex loader hard-codes ``num_seeds=1`` for the optional arm
    IK used by its heuristic grasp seeder.  That is fragile for a fixed-base
    14-DoF bimanual robot.  This adapter leaves the BODex grasp optimizer and
    collision world unchanged, but lets its seed IK search several joint-space
    initializations in parallel.
    """

    if num_seeds <= 0:
        raise ValueError("IK seed count must be positive")
    from curobo.types.robot import RobotConfig
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    robot_payload = yaml.safe_load(Path(robot_config_path).read_text())["robot_cfg"]
    kinematics = robot_payload["kinematics"]
    kinematics["link_names"] = list(PALM_LINKS)
    kinematics["ee_link"] = PALM_LINKS[0]
    ik_robot_config = RobotConfig.from_dict(robot_payload, tensor_args)
    applied_link_overrides = []
    if self_collision_link_overrides:
        applied_link_overrides = apply_self_collision_link_overrides(
            ik_robot_config,
            self_collision_link_overrides,
        )
    applied_overrides = []
    if self_collision_sphere_overrides:
        applied_overrides = apply_self_collision_sphere_overrides(
            ik_robot_config,
            self_collision_sphere_overrides,
        )
    ik_config = IKSolverConfig.load_from_robot_config(
        ik_robot_config,
        None,
        world_coll_checker=grasp_config.world_coll_checker,
        tensor_args=tensor_args,
        position_threshold=0.001,
        rotation_threshold=0.05,
        gradient_file="gradient_ik.yml",
        use_cuda_graph=False,
        use_particle_opt=bool(use_particle_opt),
        num_seeds=int(num_seeds),
        grad_iters=grad_iters,
    )
    ik_solver = IKSolver(ik_config)
    grasp_config.ik_solver = ik_solver
    grasp_config.q_sample_gen.ik_solver = ik_solver
    grasp_config.q_sample_gen.replace_ind = (
        ik_solver.kinematics.kinematics_config.get_replace_index(
            grasp_config.q_sample_gen.full_robot_model.kinematics_config
        )
    )
    return {
        "backend": "curobo.IKSolver",
        "num_seeds": int(num_seeds),
        "use_particle_opt": bool(use_particle_opt),
        "grad_iters": grad_iters,
        "position_threshold_m": 0.001,
        "rotation_threshold": 0.05,
        "shared_bodex_world_collision_checker": True,
        "self_collision_sphere_overrides": applied_overrides,
        "self_collision_link_overrides": applied_link_overrides,
    }


def install_per_candidate_hand_ik_seeds(
    grasp_config: Any,
    *,
    base_q: list[float] | None,
    joint_names: list[str],
    candidate_count: int,
    jitter_rad: float,
    seed: int,
    tensor_args: Any,
) -> dict[str, Any]:
    """Give each BODex candidate a deterministic, bounded hand posture seed."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if jitter_rad < 0.0:
        raise ValueError("hand IK jitter must be non-negative")
    if base_q is None:
        return {
            "enabled": False,
            "reason": "no explicit IK initial pose",
            "candidate_count": candidate_count,
        }
    base = np.asarray(base_q, dtype=np.float64)
    _require(base.shape == (len(joint_names),), "IK initial pose has wrong shape")
    seeds = np.repeat(base[None, :], candidate_count, axis=0)
    hand_indices = [index for index, name in enumerate(joint_names) if "_hand_" in name]
    if candidate_count > 1 and jitter_rad > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(
            -float(jitter_rad),
            float(jitter_rad),
            size=(candidate_count - 1, len(hand_indices)),
        )
        seeds[1:, hand_indices] += noise
    limits = grasp_config.robot_config.kinematics.get_joint_limits().position
    lower = limits[0].detach().cpu().numpy()
    upper = limits[1].detach().cpu().numpy()
    seeds = np.clip(seeds, lower[None, :], upper[None, :])
    grasp_config.q_sample_gen.ik_init = tensor_args.to_device(seeds).view(
        1, candidate_count, -1
    )
    return {
        "enabled": True,
        "candidate_count": int(candidate_count),
        "jitter_rad": float(jitter_rad),
        "seed": int(seed),
        "first_candidate_unjittered": True,
        "jittered_joint_count": len(hand_indices),
        "arm_joints_jittered": False,
        "clipped_to_robot_joint_limits": True,
    }


def _lift_with_two_palm_ik(
    grasp_solver: Any,
    grasp_q: torch.Tensor,
    joint_names: list[str],
    *,
    lift_height_m: float,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Hold both palm orientations while raising both targets together."""

    from curobo.types.math import Pose

    ik_solver = grasp_solver.ik_solver
    if ik_solver is None:
        return None, {"success": False, "reason": "BODex IK solver is unavailable"}
    state = grasp_solver.fk(grasp_q.unsqueeze(0))
    poses = state.link_poses or {}
    if any(link not in poses for link in PALM_LINKS):
        return None, {"success": False, "reason": "palm FK poses are unavailable"}
    targets: dict[str, Pose] = {}
    for link in PALM_LINKS:
        source = poses[link]
        position = source.position.clone()
        position[..., 2] += float(lift_height_m)
        targets[link] = Pose(position, source.quaternion.clone())
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    missing = [name for name in ik_solver.joint_names if name not in name_to_index]
    if missing:
        return None, {"success": False, "reason": f"lift IK joints missing from grasp: {missing}"}
    seed = torch.stack([grasp_q[name_to_index[name]] for name in ik_solver.joint_names])
    result = ik_solver.solve_batch(
        targets[PALM_LINKS[0]],
        seed_config=seed.view(1, 1, -1),
        return_seeds=1,
        num_seeds=1,
        use_nn_seed=False,
        link_poses=targets,
    )
    success = bool(result.success.reshape(-1)[0].item())
    metadata = {
        "success": success,
        "lift_height_m": float(lift_height_m),
        "position_error_m": result.position_error.detach().cpu().reshape(-1).tolist(),
        "rotation_error": result.rotation_error.detach().cpu().reshape(-1).tolist(),
        "joint_names": list(ik_solver.joint_names),
    }
    if not success:
        return None, metadata
    solved = result.solution.reshape(-1, result.solution.shape[-1])[0]
    lift_q = grasp_q.clone()
    for name, value in zip(ik_solver.joint_names, solved):
        lift_q[name_to_index[name]] = value
    return lift_q, metadata


def _finite_vector(tensor: torch.Tensor, expected: int, label: str) -> list[float]:
    values = tensor.detach().cpu().reshape(-1)
    _require(values.numel() == expected, f"{label} has {values.numel()} values, expected {expected}")
    _require(bool(torch.isfinite(values).all().item()), f"{label} contains non-finite values")
    return [float(value) for value in values.tolist()]


def collision_pair_diagnostics(
    robot_spheres: torch.Tensor,
    *,
    sphere_link_indices: torch.Tensor,
    link_index_to_name: dict[int, str],
    checked_pairs: torch.Tensor,
    sphere_offsets: torch.Tensor,
    maximum_rows: int = 16,
) -> list[dict[str, Any]]:
    """Identify the checked sphere pairs that actually penetrate.

    cuRobo's self-collision constraint returns only the maximum penetration
    for each candidate.  Keeping the checked sphere indices in this diagnostic
    makes it possible to distinguish a real robot collision from a fitted
    collision-sphere false positive without disabling self collision globally.
    """

    if maximum_rows <= 0:
        raise ValueError("maximum_rows must be positive")
    spheres = robot_spheres.detach().cpu()
    _require(spheres.ndim >= 3 and spheres.shape[-1] == 4, "invalid robot sphere tensor")
    sphere_count = int(spheres.shape[-2])
    links = sphere_link_indices.detach().cpu().reshape(-1).long()
    offsets = sphere_offsets.detach().cpu().reshape(-1)
    pairs = checked_pairs.detach().cpu().reshape(-1, 2).long()
    _require(links.numel() == sphere_count, "sphere-to-link map has wrong size")
    _require(offsets.numel() == sphere_count, "self-collision offsets have wrong size")
    _require(
        pairs.numel() == 0
        or bool(((pairs >= 0) & (pairs < sphere_count)).all().item()),
        "checked self-collision pair contains an invalid sphere index",
    )
    candidates = spheres.reshape(-1, sphere_count, 4)
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        for first_tensor, second_tensor in pairs:
            first = int(first_tensor.item())
            second = int(second_tensor.item())
            first_sphere = candidate[first]
            second_sphere = candidate[second]
            center_distance = float(
                torch.linalg.vector_norm(first_sphere[:3] - second_sphere[:3]).item()
            )
            collision_distance = float(
                first_sphere[3].item()
                + second_sphere[3].item()
                + offsets[first].item()
                + offsets[second].item()
            )
            penetration = collision_distance - center_distance
            if penetration <= 0.0:
                continue
            first_link_index = int(links[first].item())
            second_link_index = int(links[second].item())
            first_local_index = int((links[:first] == first_link_index).sum().item())
            second_local_index = int((links[:second] == second_link_index).sum().item())
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "sphere_indices": [first, second],
                    "link_sphere_indices": [first_local_index, second_local_index],
                    "link_indices": [first_link_index, second_link_index],
                    "link_names": [
                        link_index_to_name.get(first_link_index, f"link_index_{first_link_index}"),
                        link_index_to_name.get(second_link_index, f"link_index_{second_link_index}"),
                    ],
                    "center_distance_m": center_distance,
                    "collision_distance_m": collision_distance,
                    "penetration_m": penetration,
                }
            )
    rows.sort(key=lambda row: row["penetration_m"], reverse=True)
    return rows[:maximum_rows]


def _contiguous_constraint_q(q: torch.Tensor) -> torch.Tensor:
    """Normalize a BODex candidate batch for cuRobo's Warp constraints."""

    _require(q.ndim == 2 and q.shape[-1] == 38, "constraint diagnostics require [N, 38] q")
    _require(bool(torch.isfinite(q).all().item()), "constraint diagnostics received non-finite q")
    return q.detach().contiguous()


def finite_trajectory_mask(trajectories: torch.Tensor) -> torch.Tensor:
    """Return one validity bit per BODex seed without aborting the whole pair."""

    _require(
        trajectories.ndim == 3 and trajectories.shape[-1] == 38,
        "BODex trajectories must have shape [N, T, 38]",
    )
    return torch.isfinite(trajectories).all(dim=-1).all(dim=-1)


def json_safe_tensor_values(values: torch.Tensor) -> Any:
    """Convert diagnostic tensors to strict JSON, replacing NaN/Inf with null."""

    def safe(value: Any) -> Any:
        if isinstance(value, list):
            return [safe(item) for item in value]
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    return safe(values.detach().cpu().tolist())


def _constraint_state_terms(
    grasp_solver: Any,
    q: torch.Tensor,
) -> tuple[Any, Any, dict[str, torch.Tensor]]:
    q = _contiguous_constraint_q(q)
    rollout = grasp_solver.rollout_fn
    state = rollout.dynamics_model.forward(rollout.start_state, q.unsqueeze(1))
    terms: dict[str, torch.Tensor] = {
        "joint_bounds": rollout.bound_constraint.forward(state.state_seq),
    }
    if rollout.primitive_collision_constraint.enabled:
        terms["world_collision"] = rollout.primitive_collision_constraint.forward(
            state.robot_spheres,
            env_query_idx=rollout._goal_buffer.batch_world_idx,
            opt_progress=1.0,
        )
    if rollout.robot_self_collision_constraint.enabled:
        terms["self_collision"] = rollout.robot_self_collision_constraint.forward(
            state.robot_spheres
        )
    return rollout, state, terms


def _constraint_term_per_candidate(
    value: torch.Tensor,
    candidate_count: int,
) -> torch.Tensor:
    _require(value.numel() % candidate_count == 0, "constraint term has wrong batch size")
    return value.detach().reshape(candidate_count, -1).max(dim=-1).values


def bisect_signed_contact_boundary(
    outside_q: torch.Tensor,
    inside_q: torch.Tensor,
    *,
    signed_distance_fn: Callable[[torch.Tensor], torch.Tensor],
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Bisect a signed BODex contact constraint without relaxing success."""

    if iterations <= 0:
        raise ValueError("contact projection iterations must be positive")
    outside_q = _contiguous_constraint_q(outside_q)
    inside_q = _contiguous_constraint_q(inside_q)
    _require(outside_q.shape == inside_q.shape, "contact projection endpoints differ")
    candidate_count = outside_q.shape[0]

    outside_signed = signed_distance_fn(outside_q).detach().reshape(-1)
    inside_signed = signed_distance_fn(inside_q).detach().reshape(-1)
    _require(
        outside_signed.numel() == candidate_count
        and inside_signed.numel() == candidate_count,
        "signed contact evaluator has wrong batch size",
    )
    _require(
        bool(torch.isfinite(outside_signed).all().item())
        and bool(torch.isfinite(inside_signed).all().item()),
        "signed contact evaluator returned non-finite values",
    )
    bracketed = (outside_signed < 0.0) & (inside_signed > 0.0)
    low_q = outside_q.clone()
    high_q = inside_q.clone()
    low_signed = outside_signed.clone()
    high_signed = inside_signed.clone()
    prefer_outside = outside_signed.abs() <= inside_signed.abs()
    best_q = torch.where(prefer_outside[:, None], outside_q, inside_q).clone()
    best_signed = torch.where(prefer_outside, outside_signed, inside_signed).clone()

    for _ in range(iterations):
        mid_q = 0.5 * (low_q + high_q)
        mid_signed = signed_distance_fn(mid_q).detach().reshape(-1)
        _require(
            mid_signed.numel() == candidate_count
            and bool(torch.isfinite(mid_signed).all().item()),
            "signed contact evaluator failed during bisection",
        )
        better = bracketed & (mid_signed.abs() < best_signed.abs())
        best_q = torch.where(better[:, None], mid_q, best_q)
        best_signed = torch.where(better, mid_signed, best_signed)
        move_low = bracketed & (mid_signed <= 0.0)
        move_high = bracketed & (mid_signed > 0.0)
        low_q = torch.where(move_low[:, None], mid_q, low_q)
        low_signed = torch.where(move_low, mid_signed, low_signed)
        high_q = torch.where(move_high[:, None], mid_q, high_q)
        high_signed = torch.where(move_high, mid_signed, high_signed)

    return best_q, low_q, high_q, {
        "iterations": int(iterations),
        "bracketed": bracketed.detach().cpu().tolist(),
        "outside_signed_contact_m": outside_signed.detach().cpu().tolist(),
        "inside_signed_contact_m": inside_signed.detach().cpu().tolist(),
        "best_signed_contact_m": best_signed.detach().cpu().tolist(),
        "low_signed_contact_m": low_signed.detach().cpu().tolist(),
        "high_signed_contact_m": high_signed.detach().cpu().tolist(),
    }


def _evaluate_bodex_candidates(
    grasp_solver: Any,
    q: torch.Tensor,
    *,
    grasp_threshold: float,
    distance_threshold: float,
) -> tuple[Any, torch.Tensor]:
    """Re-evaluate candidates with BODex's original constraint/convergence code."""

    q = _contiguous_constraint_q(q)
    rollout, state, _ = _constraint_state_terms(grasp_solver, q)
    metrics = rollout.constraint_fn(state)
    metrics.state = state
    metrics = rollout.convergence_fn(state, metrics)
    candidate_count = q.shape[0]
    feasible = metrics.feasible.detach().reshape(candidate_count, -1).all(dim=-1)
    grasp_error = metrics.grasp_error.detach().reshape(candidate_count, -1).max(dim=-1).values
    distance_error = metrics.dist_error.detach().reshape(candidate_count, -1).max(dim=-1).values
    strict_success = (
        feasible
        & (grasp_error <= float(grasp_threshold))
        & (distance_error <= float(distance_threshold))
    )
    return metrics, strict_success


def _conservative_contact_boundary_success(
    grasp_solver: Any,
    q: torch.Tensor,
    metrics: Any,
    *,
    outside_tolerance_m: float,
    grasp_threshold: float,
    distance_threshold: float,
) -> torch.Tensor:
    """Accept only a numerically tiny outside residual and never penetration."""

    if outside_tolerance_m < 0.0:
        raise ValueError("outside contact tolerance must be non-negative")
    q = _contiguous_constraint_q(q)
    candidate_count = q.shape[0]
    _, _, terms = _constraint_state_terms(grasp_solver, q)
    world = _constraint_term_per_candidate(terms["world_collision"], candidate_count)
    constraint_success = (world <= 0.0) & (world >= -float(outside_tolerance_m))
    for name in ("joint_bounds", "self_collision"):
        if name in terms:
            constraint_success &= (
                _constraint_term_per_candidate(terms[name], candidate_count) == 0.0
            )
    grasp_error = metrics.grasp_error.detach().reshape(candidate_count, -1).max(dim=-1).values
    distance_error = metrics.dist_error.detach().reshape(candidate_count, -1).max(dim=-1).values
    return (
        constraint_success
        & (grasp_error <= float(grasp_threshold))
        & (distance_error <= float(distance_threshold))
    )


def evaluate_curriculum_precontact_candidates(
    grasp_solver: Any,
    q: torch.Tensor,
    *,
    grasp_error: torch.Tensor,
    distance_error: torch.Tensor,
    maximum_gap_m: float,
    grasp_threshold: float,
    distance_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Accept safe joint BODex near-contact states for curriculum stages 1--3.

    BODex's original feasibility gate requires exact contact, which is too
    strict for the first curriculum stage.  This profile keeps the original
    combined two-hand grasp/distance objectives, but permits a small *outside*
    gap.  Positive penetration and any joint/self-collision residual remain
    forbidden.
    """

    if not math.isfinite(maximum_gap_m) or maximum_gap_m <= 0.0:
        raise ValueError("precontact maximum gap must be finite and positive")
    q = _contiguous_constraint_q(q)
    candidate_count = q.shape[0]
    _, _, terms = _constraint_state_terms(grasp_solver, q)
    _require("world_collision" in terms, "BODex world contact constraint is disabled")
    world_signed = _constraint_term_per_candidate(
        terms["world_collision"], candidate_count
    )

    def exact_zero(name: str) -> torch.Tensor:
        value = terms.get(name)
        if value is None:
            return torch.ones(candidate_count, dtype=torch.bool, device=q.device)
        return value.detach().reshape(candidate_count, -1).eq(0.0).all(dim=-1)

    joint_zero = exact_zero("joint_bounds")
    self_zero = exact_zero("self_collision")
    no_positive_penetration = world_signed <= 0.0
    world_gap = torch.clamp(-world_signed, min=0.0)
    grasp_error = grasp_error.detach().reshape(candidate_count, -1).max(dim=-1).values
    distance_error = distance_error.detach().reshape(candidate_count, -1)[:, -1]
    objective_success = (
        (grasp_error <= float(grasp_threshold))
        & (distance_error <= float(distance_threshold))
    )
    accepted = (
        no_positive_penetration
        & (world_gap <= float(maximum_gap_m))
        & joint_zero
        & self_zero
        & objective_success
    )
    diagnostics = {
        "schema": "xhand_bodex_curriculum_precontact_acceptance_v1",
        "maximum_gap_m": float(maximum_gap_m),
        "grasp_threshold": float(grasp_threshold),
        "distance_threshold": float(distance_threshold),
        "world_signed_contact_m": world_signed.detach().cpu().tolist(),
        "world_gap_m": world_gap.detach().cpu().tolist(),
        "positive_world_penetration": (~no_positive_penetration).detach().cpu().tolist(),
        "joint_bounds_exact_zero": joint_zero.detach().cpu().tolist(),
        "self_collision_exact_zero": self_zero.detach().cpu().tolist(),
        "grasp_error_max": grasp_error.detach().cpu().tolist(),
        "distance_error_max": distance_error.detach().cpu().tolist(),
        "bodex_objective_success": objective_success.detach().cpu().tolist(),
        "accepted": accepted.detach().cpu().tolist(),
        "accepted_count": int(accepted.sum().item()),
        "criterion": (
            "joint bimanual BODex grasp/distance thresholds; world gap in [0, maximum_gap]; "
            "no positive penetration; exact zero joint-bound and self-collision residuals"
        ),
    }
    return accepted, diagnostics


def project_bodex_contact_boundary(
    grasp_solver: Any,
    outside_q: torch.Tensor,
    inside_q: torch.Tensor,
    *,
    iterations: int,
    maximum_inside_contact_m: float,
    outside_tolerance_m: float,
    grasp_threshold: float,
    distance_threshold: float,
) -> tuple[torch.Tensor, Any, torch.Tensor, dict[str, Any]]:
    """Project bracketed candidates and accept only strict BODex re-evaluation."""

    if maximum_inside_contact_m <= 0.0:
        raise ValueError("maximum inside contact must be positive")
    if outside_tolerance_m < 0.0:
        raise ValueError("outside contact tolerance must be non-negative")
    outside_q = _contiguous_constraint_q(outside_q)
    inside_q = _contiguous_constraint_q(inside_q)
    candidate_count = outside_q.shape[0]
    _, _, outside_terms = _constraint_state_terms(grasp_solver, outside_q)
    _, _, inside_terms = _constraint_state_terms(grasp_solver, inside_q)
    _require("world_collision" in outside_terms, "BODex world contact constraint is disabled")
    outside_world = _constraint_term_per_candidate(
        outside_terms["world_collision"], candidate_count
    )
    inside_world = _constraint_term_per_candidate(
        inside_terms["world_collision"], candidate_count
    )

    def endpoint_clear(terms: dict[str, torch.Tensor]) -> torch.Tensor:
        clear = torch.ones(candidate_count, device=outside_q.device, dtype=torch.bool)
        for name in ("joint_bounds", "self_collision"):
            if name in terms:
                clear &= _constraint_term_per_candidate(terms[name], candidate_count) == 0.0
        return clear

    eligible = (
        endpoint_clear(outside_terms)
        & endpoint_clear(inside_terms)
        & (outside_world < 0.0)
        & (inside_world > 0.0)
        & (inside_world <= float(maximum_inside_contact_m))
    )
    search_inside = torch.where(eligible[:, None], inside_q, outside_q)

    def signed_distance(candidate_q: torch.Tensor) -> torch.Tensor:
        _, _, terms = _constraint_state_terms(grasp_solver, candidate_q)
        return _constraint_term_per_candidate(
            terms["world_collision"], candidate_q.shape[0]
        )

    best_q, low_q, high_q, bisection = bisect_signed_contact_boundary(
        outside_q,
        search_inside,
        signed_distance_fn=signed_distance,
        iterations=iterations,
    )
    option_success_rows = []
    option_strict_success_rows = []
    option_grasp_error_rows = []
    option_distance_error_rows = []
    for option_q in (best_q, low_q, high_q):
        option_metrics, strict_candidate_success = _evaluate_bodex_candidates(
            grasp_solver,
            option_q,
            grasp_threshold=grasp_threshold,
            distance_threshold=distance_threshold,
        )
        if outside_tolerance_m > 0.0:
            candidate_success = _conservative_contact_boundary_success(
                grasp_solver,
                option_q,
                option_metrics,
                outside_tolerance_m=outside_tolerance_m,
                grasp_threshold=grasp_threshold,
                distance_threshold=distance_threshold,
            )
        else:
            candidate_success = strict_candidate_success
        option_success_rows.append(candidate_success)
        option_strict_success_rows.append(strict_candidate_success)
        option_grasp_error_rows.append(
            option_metrics.grasp_error.detach()
            .reshape(candidate_count, -1)
            .max(dim=-1)
            .values
        )
        option_distance_error_rows.append(
            option_metrics.dist_error.detach()
            .reshape(candidate_count, -1)
            .max(dim=-1)
            .values
        )
    option_success = torch.stack(option_success_rows, dim=0)
    selected_q = best_q.clone()
    selected_option = torch.full(
        (candidate_count,), -1, device=outside_q.device, dtype=torch.long
    )
    for option_index, option_q in enumerate((best_q, low_q, high_q)):
        choose = eligible & (selected_option < 0) & option_success[option_index]
        selected_q = torch.where(choose[:, None], option_q, selected_q)
        selected_option = torch.where(
            choose,
            torch.full_like(selected_option, option_index),
            selected_option,
        )
    selected_metrics, strict_success = _evaluate_bodex_candidates(
        grasp_solver,
        selected_q,
        grasp_threshold=grasp_threshold,
        distance_threshold=distance_threshold,
    )
    if outside_tolerance_m > 0.0:
        strict_success = _conservative_contact_boundary_success(
            grasp_solver,
            selected_q,
            selected_metrics,
            outside_tolerance_m=outside_tolerance_m,
            grasp_threshold=grasp_threshold,
            distance_threshold=distance_threshold,
        )
    strict_success &= eligible & (selected_option >= 0)
    bisection.update(
        {
            "schema": "xhand_bodex_contact_boundary_projection_v2",
            "enabled": True,
            "maximum_inside_contact_m": float(maximum_inside_contact_m),
            "outside_contact_tolerance_m": float(outside_tolerance_m),
            "positive_penetration_allowed": False,
            "eligible": eligible.detach().cpu().tolist(),
            "selected_option": selected_option.detach().cpu().tolist(),
            "option_boundary_success": option_success.detach().cpu().tolist(),
            "option_upstream_strict_success": torch.stack(
                option_strict_success_rows, dim=0
            ).detach().cpu().tolist(),
            "option_grasp_error_max": torch.stack(
                option_grasp_error_rows, dim=0
            ).detach().cpu().tolist(),
            "option_distance_error_max": torch.stack(
                option_distance_error_rows, dim=0
            ).detach().cpu().tolist(),
            "strict_success": strict_success.detach().cpu().tolist(),
            "strict_success_count": int(strict_success.sum().item()),
            "success_criterion": (
                "one-sided numerical contact boundary plus exact joint/self constraints and "
                "original BODex grasp_error/distance_error thresholds"
            ),
            "tolerance_relaxation": bool(outside_tolerance_m > 0.0),
        }
    )
    return selected_q, selected_metrics, strict_success, bisection


def _constraint_breakdown(grasp_solver: Any, q: torch.Tensor) -> dict[str, Any]:
    """Report which BODex feasibility constraint rejected final candidates."""

    q = _contiguous_constraint_q(q)
    rollout, state, terms = _constraint_state_terms(grasp_solver, q)
    robot_model = rollout.dynamics_model.robot_model
    kinematics = robot_model.kinematics_config
    link_index_to_name = {
        int(index): name for name, index in kinematics.link_name_to_idx_map.items()
    }
    diagnostics = {
        name: {
            "max": float(value.detach().max().item()),
            "min": float(value.detach().min().item()),
            "mean": float(value.detach().mean().item()),
            "nonzero": int((value.detach() != 0.0).sum().item()),
            "positive": int((value.detach() > 0.0).sum().item()),
            "negative": int((value.detach() < 0.0).sum().item()),
        }
        for name, value in terms.items()
    }
    combined_constraint = torch.stack(list(terms.values()), dim=0).sum(dim=0)
    diagnostics["combined_feasibility"] = {
        "max": float(combined_constraint.detach().max().item()),
        "min": float(combined_constraint.detach().min().item()),
        "feasible": int((combined_constraint.detach() == 0.0).sum().item()),
        "candidate_count": int(combined_constraint.numel()),
        "criterion": "BODex/curobo requires the summed constraint to equal exactly zero",
        "world_esdf_note": (
            "negative means all robot spheres remain outside the allowed object contact surface; "
            "zero means at least one allowed contact reaches its target surface without penetration"
        ),
    }
    if "world_collision" in diagnostics:
        query_buffer = rollout.primitive_collision_constraint._collision_query_buffer
        collision_buffer = query_buffer.mesh_collision_buffer
        if collision_buffer is None:
            collision_buffer = query_buffer.primitive_collision_buffer
        if collision_buffer is not None:
            per_sphere = collision_buffer.distance_buffer.detach().cpu().reshape(
                -1, state.robot_spheres.shape[-2]
            )
            sphere_links = kinematics.link_sphere_idx_map.detach().cpu().reshape(-1).long()
            sphere_rows = state.robot_spheres.detach().cpu().reshape(
                -1, state.robot_spheres.shape[-2], 4
            )
            closest_rows = []
            for candidate_index, values in enumerate(per_sphere):
                sphere_index = int(torch.argmax(values).item())
                link_index = int(sphere_links[sphere_index].item())
                local_index = int(
                    (sphere_links[:sphere_index] == link_index).sum().item()
                )
                closest_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "sphere_index": sphere_index,
                        "link_name": link_index_to_name.get(
                            link_index, f"link_index_{link_index}"
                        ),
                        "link_sphere_index": local_index,
                        "signed_contact_esdf_m": float(values[sphere_index].item()),
                        "center_world_m": sphere_rows[candidate_index, sphere_index, :3].tolist(),
                        "radius_m": float(
                            sphere_rows[candidate_index, sphere_index, 3].item()
                        ),
                    }
                )
            diagnostics["world_collision"]["closest_spheres"] = closest_rows
    if "self_collision" in diagnostics:
        self_collision = (
            rollout.robot_self_collision_constraint.self_collision_kin_config
        )
        if self_collision.experimental_kernel:
            pair_entry_count = int(self_collision.thread_max)
            checked_pairs = self_collision.thread_location[:pair_entry_count].reshape(-1, 2)
            pair_source = "experimental_thread_locations"
        else:
            # When the fitted full-body model exceeds the experimental
            # kernel's 16,384-pair capacity, cuRobo falls back to its complete
            # NxN collision matrix.  The partially filled thread-location
            # buffer must not be used for diagnostics in that mode.
            collision_matrix = self_collision.collision_matrix
            checked_pairs = torch.nonzero(
                torch.triu(collision_matrix, diagonal=1), as_tuple=False
            )
            pair_source = "fallback_collision_matrix"
        diagnostics["self_collision"]["checked_pair_count"] = int(
            checked_pairs.shape[0]
        )
        diagnostics["self_collision"]["checked_pair_source"] = pair_source
        diagnostics["self_collision"]["colliding_pairs"] = (
            collision_pair_diagnostics(
                state.robot_spheres,
                sphere_link_indices=kinematics.link_sphere_idx_map,
                link_index_to_name=link_index_to_name,
                checked_pairs=checked_pairs,
                sphere_offsets=self_collision.offset,
            )
        )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--paired-seeds", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds-per-pair", type=int, default=8)
    parser.add_argument(
        "--pair-start",
        type=int,
        default=0,
        help="zero-based source-pair offset for non-overlapping solver shards",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--translation-jitter-m", type=float, default=0.008)
    parser.add_argument("--rotation-jitter-deg", type=float, default=8.0)
    parser.add_argument("--lift-height-m", type=float, default=0.08)
    parser.add_argument("--grasp-threshold", type=float, default=0.1)
    parser.add_argument("--distance-threshold", type=float, default=0.02)
    parser.add_argument(
        "--curriculum-seed-stage",
        type=int,
        choices=range(1, 7),
        default=6,
        help=(
            "curriculum profile produced by this solver run; stages 1--3 accept safe "
            "joint BODex precontact states, while stages 4--6 require strict contact "
            "and collision-aware lift IK"
        ),
    )
    parser.add_argument(
        "--stage1-precontact-max-gap-m",
        type=float,
        default=0.006,
        help="maximum outside world-contact gap accepted for curriculum stages 1--3",
    )
    parser.add_argument(
        "--precontact-grasp-threshold",
        type=float,
        help=(
            "override the stage-aware pre-lift BODex grasp threshold; defaults are "
            "0.50/0.25/0.10 for curriculum stages 1/2/3"
        ),
    )
    parser.add_argument("--grad-iters", type=int)
    parser.add_argument("--use-particle-opt", action="store_true")
    parser.add_argument(
        "--max-ge-stage",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "last BODex contact-schedule stage that keeps grasp energy active; "
            "0 preserves the pinned upstream default, 2 keeps force closure active "
            "through final exact-contact optimization"
        ),
    )
    parser.add_argument(
        "--postsolve-contact-projection-iterations",
        type=int,
        default=0,
        help=(
            "strictly bisect a no-contact/penetrating BODex trajectory segment and "
            "re-evaluate candidates with the original BODex success criteria; 0 disables"
        ),
    )
    parser.add_argument(
        "--postsolve-contact-projection-max-penetration-m",
        type=float,
        default=0.005,
        help="maximum final BODex contact penetration eligible for strict projection",
    )
    parser.add_argument(
        "--postsolve-contact-projection-outside-tolerance-m",
        type=float,
        default=0.0,
        help=(
            "optional one-sided numerical boundary tolerance: accepts only values in "
            "[-tolerance, 0], never positive penetration"
        ),
    )
    parser.add_argument(
        "--retain-best",
        action="store_true",
        help="retain LBFGS best cost across BODex stage switches (off by default, matching upstream)",
    )
    parser.add_argument(
        "--arm-seed-regularization",
        type=float,
        default=0.0,
        help="null-space weight that keeps the 14 arm joints near the bilateral IK seed",
    )
    parser.add_argument(
        "--arm-line-search-multiplier",
        type=float,
        default=0.001,
        help="multiplier applied to BODex's fixed-base line-search coefficient for 14 arm joints",
    )
    parser.add_argument(
        "--hand-line-search-multiplier",
        type=float,
        default=0.1,
        help="multiplier applied to BODex's fixed-base line-search coefficient for 24 hand joints",
    )
    parser.add_argument("--ik-num-seeds", type=int, default=32)
    parser.add_argument("--ik-grad-iters", type=int)
    parser.add_argument("--ik-use-particle-opt", action="store_true")
    parser.add_argument("--ik-initial-pose", type=Path)
    parser.add_argument("--ik-hand-jitter-rad", type=float, default=0.12)
    parser.add_argument("--ik-hand-jitter-seed", type=int, default=20260829)
    parser.add_argument(
        "--coordinate-seed-replay-diagnostics",
        type=Path,
        help=(
            "optional hash-recorded replay of coordinate_seed_q rows previously generated "
            "by this joint-bimanual BODex runner"
        ),
    )
    parser.add_argument(
        "--coordinate-seed-replay-repeat-first-pair",
        action="store_true",
        help=(
            "reuse the first replayed joint-BODex coordinate-seed batch as an IK warm "
            "start for every new paired-surface target"
        ),
    )
    parser.add_argument("--solver-seed", type=int, default=20260829)
    parser.add_argument(
        "--rejected-diagnostics-output",
        type=Path,
        help="optional JSON sidecar containing final q values for rejected-solution analysis",
    )
    parser.add_argument(
        "--self-collision-sphere-overrides",
        type=Path,
        help="hash-locked YAML of mesh-validated exact collision-sphere false positives",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.pair_start < 0:
        raise ValueError("pair-start must be non-negative")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("max-pairs must be positive")
    if args.lift_height_m <= 0.0:
        raise ValueError("lift-height-m must be positive")
    if (
        not math.isfinite(args.stage1_precontact_max_gap_m)
        or args.stage1_precontact_max_gap_m <= 0.0
    ):
        raise ValueError("stage1-precontact-max-gap-m must be finite and positive")
    if args.precontact_grasp_threshold is not None and (
        not math.isfinite(args.precontact_grasp_threshold)
        or args.precontact_grasp_threshold <= 0.0
    ):
        raise ValueError("precontact-grasp-threshold must be finite and positive")
    stage_precontact_grasp_threshold = (
        float(args.precontact_grasp_threshold)
        if args.precontact_grasp_threshold is not None
        else {1: 0.50, 2: 0.25, 3: float(args.grasp_threshold)}.get(
            int(args.curriculum_seed_stage), float(args.grasp_threshold)
        )
    )
    if args.arm_seed_regularization < 0.0:
        raise ValueError("arm-seed-regularization must be non-negative")
    if not math.isfinite(args.arm_line_search_multiplier) or args.arm_line_search_multiplier <= 0.0:
        raise ValueError("arm-line-search-multiplier must be finite and positive")
    if not math.isfinite(args.hand_line_search_multiplier) or args.hand_line_search_multiplier <= 0.0:
        raise ValueError("hand-line-search-multiplier must be finite and positive")
    if args.postsolve_contact_projection_iterations < 0:
        raise ValueError("postsolve-contact-projection-iterations must be non-negative")
    if (
        not math.isfinite(args.postsolve_contact_projection_max_penetration_m)
        or args.postsolve_contact_projection_max_penetration_m <= 0.0
    ):
        raise ValueError(
            "postsolve-contact-projection-max-penetration-m must be finite and positive"
        )
    if (
        not math.isfinite(args.postsolve_contact_projection_outside_tolerance_m)
        or args.postsolve_contact_projection_outside_tolerance_m < 0.0
    ):
        raise ValueError(
            "postsolve-contact-projection-outside-tolerance-m must be finite and non-negative"
        )
    if (
        args.rejected_diagnostics_output is not None
        and args.rejected_diagnostics_output.exists()
    ):
        raise FileExistsError(args.rejected_diagnostics_output)
    if (
        args.coordinate_seed_replay_repeat_first_pair
        and args.coordinate_seed_replay_diagnostics is None
    ):
        raise ValueError(
            "coordinate-seed-replay-repeat-first-pair requires replay diagnostics"
        )
    random.seed(args.solver_seed)
    np.random.seed(args.solver_seed)
    torch.manual_seed(args.solver_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.solver_seed)
    source = verify_bodex_checkout(args.bodex_root)
    manifest = json.loads(args.manifest.read_text())
    _require(args.object in manifest.get("objects", {}), f"unknown manifest object: {args.object}")
    object_row = manifest["objects"][args.object]
    mesh_path = Path(object_row["mesh"]["path"]).resolve()
    _require(mesh_path.is_file(), f"object mesh does not exist: {mesh_path}")
    _require(sha256_file(mesh_path) == object_row["mesh"]["sha256"], "object mesh hash mismatch")
    pair_payload, local_pairs = load_paired_seeds(args.paired_seeds)
    _require(
        Path(pair_payload["mesh"]).resolve() == mesh_path,
        "paired seeds were generated from a different mesh",
    )
    source_pair_count = len(local_pairs)
    selected_pair_indices = pair_window_indices(
        source_pair_count,
        start=args.pair_start,
        limit=args.max_pairs,
    )
    local_pairs = [local_pairs[index] for index in selected_pair_indices]
    extents = [float(value) for value in object_row["extents_m"]]
    object_position = [0.5, 0.0, float(manifest["table_top_z_m"]) + 0.5 * extents[2]]
    object_quaternion = [1.0, 0.0, 0.0, 0.0]
    world_pairs = [
        transform_pair_to_world(
            pair,
            translation_xyz=object_position,
            quaternion_wxyz=object_quaternion,
        )
        for pair in local_pairs
    ]
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig

    tensor_args = TensorDeviceType(device=torch.device(args.device), dtype=torch.float32)
    joint_names, initial_q, collision_spheres = load_robot_seed_data(
        args.robot_config,
        tensor_args=tensor_args,
    )
    coordinate_seed_replay: dict[int, list[list[float]]] = {}
    coordinate_seed_replay_source = None
    if args.coordinate_seed_replay_diagnostics is not None:
        (
            coordinate_seed_replay,
            coordinate_seed_replay_source,
        ) = load_coordinate_seed_replay(
            args.coordinate_seed_replay_diagnostics,
            object_name=args.object,
            robot_config_path=args.robot_config,
            joint_names=joint_names,
        )
    self_collision_override_source = None
    self_collision_sphere_overrides: list[dict[str, Any]] = []
    self_collision_link_overrides: list[dict[str, Any]] = []
    if args.self_collision_sphere_overrides is not None:
        (
            override_payload,
            self_collision_sphere_overrides,
            self_collision_link_overrides,
        ) = (
            load_self_collision_sphere_overrides(
                args.self_collision_sphere_overrides,
                robot_config_path=args.robot_config,
            )
        )
        self_collision_override_source = {
            "path": str(args.self_collision_sphere_overrides.resolve()),
            "sha256": sha256_file(args.self_collision_sphere_overrides),
            "schema": override_payload["schema"],
            "robot_config_sha256": override_payload["robot_config_sha256"],
        }
    ik_initial_q = None
    ik_initial_pose = None
    if args.ik_initial_pose is not None:
        ik_initial_q, ik_initial_pose = load_ik_initial_pose(
            args.ik_initial_pose,
            joint_names=joint_names,
            fallback_q=initial_q,
        )
    left_contacts = contact_point_names(
        collision_spheres, CONTACT_POINT_LINKS_BY_SIDE["left"]
    )
    right_contacts = contact_point_names(
        collision_spheres, CONTACT_POINT_LINKS_BY_SIDE["right"]
    )
    physical_contact_meshes = [
        *CONTACT_MESH_LINKS_BY_SIDE["left"],
        *CONTACT_MESH_LINKS_BY_SIDE["right"],
    ]
    manip = base_manip_config(
        robot_config_path=args.robot_config,
        initial_q=initial_q,
        seed_num=args.seeds_per_pair,
        translation_jitter_m=args.translation_jitter_m,
        rotation_jitter_deg=args.rotation_jitter_deg,
        max_ge_stage=args.max_ge_stage,
    )
    manip["seeder_cfg"]["ik_init_q"] = ik_initial_q
    fixed_base_gradient_path = materialize_fixed_base_gradient_config(
        args.bodex_root.resolve(),
        args.output.with_suffix(".gradient.yml"),
        retain_best=args.retain_best,
        joint_names=joint_names,
        arm_seed_regularization=args.arm_seed_regularization,
    )
    manip["gradient_file"] = str(fixed_base_gradient_path.resolve())
    manip = configure_joint_bimanual_seed(
        manip,
        world_pairs[0],
        initial_q=initial_q,
        left_contact_names=left_contacts,
        right_contact_names=right_contacts,
    )

    object_urdf_path = materialize_single_mesh_urdf(
        mesh_path, args.output.with_suffix(".object.urdf")
    )
    world = _world_model(
        object_name=args.object,
        mesh_path=mesh_path,
        object_urdf_path=object_urdf_path,
        position=object_position,
        quaternion=object_quaternion,
    )
    from curobo.types.robot import RobotConfig

    solver_robot_payload = yaml.safe_load(args.robot_config.read_text())["robot_cfg"]
    # Keep BODex point constraints and physical GJK meshes separate. The two
    # palms remain transferred pose targets and are not force-contact rows.
    solver_robot_payload["kinematics"]["contact_mesh_names"] = physical_contact_meshes
    solver_robot_config = RobotConfig.from_dict(solver_robot_payload, tensor_args)
    solver_collision_overrides: dict[str, Any] = {
        "nonphysical_contact_point_exclusion": (
            exclude_nonphysical_contact_points_from_self_collision(
                solver_robot_config,
                [
                    *CONTACT_POINT_LINKS_BY_SIDE["left"],
                    *CONTACT_POINT_LINKS_BY_SIDE["right"],
                ],
            )
        ),
        "link_overrides": [],
        "sphere_overrides": [],
    }
    if self_collision_sphere_overrides or self_collision_link_overrides:
        applied_link_overrides = apply_self_collision_link_overrides(
            solver_robot_config,
            self_collision_link_overrides,
        )
        applied_sphere_overrides = apply_self_collision_sphere_overrides(
            solver_robot_config,
            self_collision_sphere_overrides,
        )
        solver_collision_overrides = {
            "link_overrides": applied_link_overrides,
            "sphere_overrides": applied_sphere_overrides,
        }
    grasp_config = GraspSolverConfig.load_from_robot_config(
        robot_cfg=solver_robot_config,
        world_model=[world],
        manip_name_list=[args.object],
        manip_config_data=manip,
        obj_gravity_center=[object_position],
        obj_obb_length=[0.5 * math.sqrt(sum(value * value for value in extents))],
        tensor_args=tensor_args,
        grasp_threshold=float(args.grasp_threshold),
        distance_threshold=float(args.distance_threshold),
        use_cuda_graph=False,
        grad_iters=args.grad_iters,
        use_particle_opt=bool(args.use_particle_opt),
        # BODex's Newton path uses separate floating-base translation,
        # quaternion and hand scales. The full-body XHand has no floating
        # root, so use the upstream LBFGS path intended for fixed-base arms.
        use_gradient_descent=False,
        store_debug=False,
    )
    grouped_line_search = install_grouped_line_search_scales(
        grasp_config,
        joint_names=joint_names,
        arm_multiplier=args.arm_line_search_multiplier,
        hand_multiplier=args.hand_line_search_multiplier,
    )
    mesh_ge_query_refresh = install_mesh_ge_query_refresh(
        grasp_config,
        max_ge_stage=args.max_ge_stage,
    )
    reluqp_device_alignment = install_reluqp_device_alignment(
        grasp_config,
        tensor_args=tensor_args,
    )
    effective_ik_num_seeds = args.ik_num_seeds
    if ik_initial_q is not None and effective_ik_num_seeds != 1:
        # The pinned cuRobo fork interprets an explicit seed_config as exactly
        # one seed per target. Combining it with a larger configured seed count
        # creates an invalid flattened optimizer shape.
        effective_ik_num_seeds = 1
    seed_ik = install_multiseed_seed_ik(
        grasp_config,
        robot_config_path=args.robot_config,
        tensor_args=tensor_args,
        num_seeds=effective_ik_num_seeds,
        use_particle_opt=args.ik_use_particle_opt,
        grad_iters=args.ik_grad_iters,
        self_collision_sphere_overrides=self_collision_sphere_overrides,
        self_collision_link_overrides=self_collision_link_overrides,
    )
    seed_ik["requested_num_seeds"] = int(args.ik_num_seeds)
    seed_ik["explicit_seed_requires_single_seed"] = ik_initial_q is not None
    hand_seed_diversity = install_per_candidate_hand_ik_seeds(
        grasp_config,
        base_q=ik_initial_q,
        joint_names=joint_names,
        candidate_count=args.seeds_per_pair,
        jitter_rad=args.ik_hand_jitter_rad,
        seed=args.ik_hand_jitter_seed,
        tensor_args=tensor_args,
    )
    grasp_solver = GraspSolver(grasp_config)
    samples: list[dict[str, Any]] = []
    rejected_diagnostics: list[dict[str, Any]] = []
    rejected = {
        "bodex_unsuccessful": 0,
        "lift_ik_unsuccessful": 0,
        "upstream_empty_contact_query": 0,
        "nonfinite_solution": 0,
    }
    for pair_progress, (pair_index, pair) in enumerate(
        zip(selected_pair_indices, world_pairs),
        start=1,
    ):
        grasp_solver.q_sample_gen.seeder_cfg["t"] = [
            list(pair.left_position_m),
            list(pair.right_position_m),
        ]
        grasp_solver.q_sample_gen.seeder_cfg["r"] = [
            list(pair.left_rotation_6d),
            list(pair.right_rotation_6d),
        ]
        grasp_solver.q_sample_gen.seeder_cfg["q"] = list(initial_q)
        grasp_solver.reset_seed([args.object])
        if coordinate_seed_replay:
            replay_seeds = coordinate_seed_replay.get(pair_index)
            if replay_seeds is None and args.coordinate_seed_replay_repeat_first_pair:
                replay_seeds = coordinate_seed_replay[min(coordinate_seed_replay)]
            _require(
                replay_seeds is not None and len(replay_seeds) >= args.seeds_per_pair,
                f"coordinate-seed replay lacks pair {pair_index} with "
                f"{args.seeds_per_pair} seeds",
            )
            coordinate_seed = tensor_args.to_device(
                replay_seeds[: args.seeds_per_pair]
            ).unsqueeze(0)
        else:
            coordinate_seed = grasp_solver.generate_seed(
                num_seeds=args.seeds_per_pair,
                batch=1,
                use_nn_seed=False,
            )
        seed_constraint_breakdown = _constraint_breakdown(
            grasp_solver,
            coordinate_seed[0],
        )
        try:
            result = grasp_solver.solve_batch_env(
                retract_config=(
                    coordinate_seed[:, 0, :]
                    if args.arm_seed_regularization > 0.0
                    else None
                ),
                seed_config=coordinate_seed,
                return_seeds=args.seeds_per_pair,
                num_seeds=args.seeds_per_pair,
                use_nn_seed=False,
            )
        except UnboundLocalError as exc:
            # Pinned BODex grasp_cost.py leaves raw_dist undefined when GJK
            # returns no contact-query row for a seed. Treat that geometry as
            # rejected and continue with the remaining independent pairs.
            if "raw_dist" not in str(exc):
                raise
            rejected["upstream_empty_contact_query"] += args.seeds_per_pair
            print(
                json.dumps(
                    {
                        "pair": pair_progress,
                        "pair_count": len(world_pairs),
                        "source_pair_index": pair_index,
                        "accepted_total": len(samples),
                        "rejected": rejected,
                        "upstream_exception": str(exc),
                        "seed_constraint_breakdown": seed_constraint_breakdown,
                    }
                ),
                flush=True,
            )
            continue
        shape_contract = validate_bodex_result_shapes(
            solution_shape=result.solution.shape,
            contact_point_shape=result.contact_point.shape,
            contact_frame_shape=result.contact_frame.shape,
            left_contact_count=len(left_contacts),
            right_contact_count=len(right_contacts),
        )
        trajectories = result.solution[0].clone()
        finite_solutions = finite_trajectory_mask(trajectories)
        nonfinite_indices = torch.nonzero(
            ~finite_solutions, as_tuple=False
        ).squeeze(-1)
        rejected["nonfinite_solution"] += int(nonfinite_indices.numel())
        upstream_success = result.success[0].reshape(-1).clone()
        success = upstream_success.clone() & finite_solutions
        grasp_error = result.grasp_error[0].clone()
        distance_error = result.dist_error[0].clone()
        contact_points = result.contact_point[0].clone()
        contact_frames = result.contact_frame[0].clone()
        contact_forces = result.contact_force[0].clone()
        contact_boundary_projection: dict[str, Any] = {
            "schema": "xhand_bodex_contact_boundary_projection_v2",
            "enabled": False,
            "strict_success_count": 0,
            "tolerance_relaxation": False,
        }
        if args.postsolve_contact_projection_iterations > 0 and bool(
            finite_solutions.all().item()
        ):
            (
                projected_q,
                projected_metrics,
                projection_success,
                contact_boundary_projection,
            ) = project_bodex_contact_boundary(
                grasp_solver,
                trajectories[:, 0, :],
                trajectories[:, -1, :],
                iterations=args.postsolve_contact_projection_iterations,
                maximum_inside_contact_m=(
                    args.postsolve_contact_projection_max_penetration_m
                ),
                outside_tolerance_m=(
                    args.postsolve_contact_projection_outside_tolerance_m
                ),
                grasp_threshold=args.grasp_threshold,
                distance_threshold=args.distance_threshold,
            )
            replace = (~upstream_success) & projection_success
            trajectories[replace, -1, :] = projected_q[replace]
            success |= projection_success
            projected_grasp_error = projected_metrics.grasp_error.detach().reshape(
                grasp_error.shape
            )
            projected_distance_error = projected_metrics.dist_error.detach().reshape(
                success.numel(), -1
            ).max(dim=-1).values
            projected_contact_points = projected_metrics.contact_point.detach().reshape(
                contact_points.shape
            )
            projected_contact_frames = projected_metrics.contact_frame.detach().reshape(
                contact_frames.shape
            )
            projected_contact_forces = projected_metrics.contact_force.detach().reshape(
                contact_forces.shape
            )
            grasp_error[replace] = projected_grasp_error[replace]
            if distance_error.ndim == 1:
                distance_error[replace] = projected_distance_error[replace]
            else:
                distance_error[replace, -1] = projected_distance_error[replace]
            contact_points[replace] = projected_contact_points[replace]
            contact_frames[replace] = projected_contact_frames[replace]
            contact_forces[replace] = projected_contact_forces[replace]
        elif args.postsolve_contact_projection_iterations > 0:
            contact_boundary_projection["skipped_due_to_nonfinite_solutions"] = (
                nonfinite_indices.detach().cpu().tolist()
            )
        strict_bodex_success = success.clone()
        stage_seed_acceptance: dict[str, Any] = {
            "schema": "xhand_bodex_strict_lift_ready_acceptance_v1",
            "accepted": strict_bodex_success.detach().cpu().tolist(),
            "accepted_count": int(strict_bodex_success.sum().item()),
        }
        if args.curriculum_seed_stage <= 3:
            finite_indices = torch.nonzero(
                finite_solutions, as_tuple=False
            ).squeeze(-1)
            success = torch.zeros_like(success)
            if finite_indices.numel() > 0:
                (
                    finite_success,
                    stage_seed_acceptance,
                ) = evaluate_curriculum_precontact_candidates(
                    grasp_solver,
                    trajectories[finite_indices, -1, :],
                    grasp_error=grasp_error[finite_indices],
                    distance_error=distance_error[finite_indices],
                    maximum_gap_m=args.stage1_precontact_max_gap_m,
                    grasp_threshold=stage_precontact_grasp_threshold,
                    distance_threshold=args.distance_threshold,
                )
                success[finite_indices] = finite_success
                stage_seed_acceptance["finite_candidate_indices"] = (
                    finite_indices.detach().cpu().tolist()
                )
            else:
                stage_seed_acceptance = {
                    "schema": "xhand_bodex_curriculum_precontact_acceptance_v1",
                    "accepted": [],
                    "accepted_count": 0,
                    "finite_candidate_indices": [],
                    "skipped": "all BODex solutions were non-finite",
                }
        finite_indices = torch.nonzero(
            finite_solutions, as_tuple=False
        ).squeeze(-1)
        constraint_breakdown = {
            "candidate_count": int(finite_solutions.numel()),
            "finite_candidate_indices": finite_indices.detach().cpu().tolist(),
            "nonfinite_candidate_indices": nonfinite_indices.detach().cpu().tolist(),
            "finite_candidates": (
                _constraint_breakdown(
                    grasp_solver,
                    trajectories[finite_indices, -1, :],
                )
                if finite_indices.numel() > 0
                else None
            ),
        }
        if args.rejected_diagnostics_output is not None:
            rejected_diagnostics.append(
                {
                    "pair_index": pair_index,
                    "joint_names": list(joint_names),
                    "coordinate_seed_q": coordinate_seed[0].detach().cpu().tolist(),
                    "upstream_solver_success": upstream_success.detach().cpu().tolist(),
                    "strict_bodex_success": strict_bodex_success.detach().cpu().tolist(),
                    "solver_success": success.detach().cpu().tolist(),
                    "grasp_error": grasp_error.detach().cpu().tolist(),
                    "distance_error": distance_error.detach().cpu().tolist(),
                    "finite_solution": finite_solutions.detach().cpu().tolist(),
                    "trajectory_q": json_safe_tensor_values(trajectories),
                    "final_q": json_safe_tensor_values(trajectories[:, -1, :]),
                    "contact_boundary_projection": contact_boundary_projection,
                    "stage_seed_acceptance": stage_seed_acceptance,
                    "constraint_breakdown": constraint_breakdown,
                    "seed_constraint_breakdown": seed_constraint_breakdown,
                }
            )
        for seed_index in range(success.numel()):
            if not bool(success[seed_index].item()):
                rejected["bodex_unsuccessful"] += 1
                continue
            trajectory = trajectories[seed_index]
            _require(trajectory.ndim == 2 and trajectory.shape[-1] == len(joint_names), "unexpected BODex trajectory shape")
            pregrasp_q = trajectory[0]
            grasp_q = trajectory[-1]
            if args.curriculum_seed_stage <= 3:
                lift_q = grasp_q.clone()
                lift_mode = "not_required_for_curriculum_stages_1_to_3"
                lift_result = {
                    "success": False,
                    "requested": False,
                    "reason": "curriculum seed profile does not include a lift target",
                }
            else:
                lift_q, lift_result = _lift_with_two_palm_ik(
                    grasp_solver,
                    grasp_q,
                    joint_names,
                    lift_height_m=args.lift_height_m,
                )
                if lift_q is None:
                    rejected["lift_ik_unsuccessful"] += 1
                    continue
                lift_mode = "curobo_collision_aware_two_palm_ik_from_bodex_grasp"
            left_count = len(left_contacts)
            contact_point = contact_points[seed_index]
            contact_frame = contact_frames[seed_index]
            contact_force = contact_forces[seed_index]
            candidate_id = f"{args.object}_pair_{pair_index:05d}_seed_{seed_index:03d}"
            samples.append(
                {
                    "candidate_id": candidate_id,
                    "object_name": args.object,
                    "stage_seed_profile": int(args.curriculum_seed_stage),
                    "object_mesh_path": str(mesh_path),
                    "object_mesh_sha256": sha256_file(mesh_path),
                    "object_pose_world": _object_pose_matrix(object_position, object_quaternion),
                    "joint_names": list(joint_names),
                    "pregrasp_full_body_q": _finite_vector(pregrasp_q, len(joint_names), "pregrasp"),
                    "full_body_q": _finite_vector(grasp_q, len(joint_names), "grasp"),
                    "lift_full_body_q": _finite_vector(lift_q, len(joint_names), "lift"),
                    "paired_surface_seed": {
                        "pair_index": pair_index,
                        "source_surface_indices": list(pair.source_surface_indices),
                        "opposition_cosine": pair.opposition_cosine,
                        "span_m": pair.span_m,
                        "lateral_fraction": pair.lateral_fraction,
                        "side_normal_min_component": pair.side_normal_min_component,
                        "x_offset_m": pair.x_offset_m,
                        "z_offset_m": pair.z_offset_m,
                        "left_position_world_m": list(pair.left_position_m),
                        "right_position_world_m": list(pair.right_position_m),
                    },
                    "contact_point": contact_point.detach().cpu().tolist(),
                    "contact_frame": contact_frame.detach().cpu().tolist(),
                    "contact_force": contact_force.detach().cpu().tolist(),
                    "left_contact_point": contact_point[:left_count].detach().cpu().tolist(),
                    "right_contact_point": contact_point[left_count:].detach().cpu().tolist(),
                    "grasp_error": grasp_error[seed_index].detach().cpu().reshape(-1).tolist(),
                    "distance_error": distance_error[seed_index].detach().cpu().reshape(-1).tolist(),
                    "lift_mode": lift_mode,
                    "lift_result": lift_result,
                    "bodex_result": {
                        **shape_contract,
                        "success": bool(strict_bodex_success[seed_index].item()),
                        "pair_index": pair_index,
                        "seed_index": seed_index,
                        "solver_success": bool(strict_bodex_success[seed_index].item()),
                        "strict_bodex_success": bool(
                            strict_bodex_success[seed_index].item()
                        ),
                        "curriculum_seed_accepted": True,
                        "stage_seed_profile": int(args.curriculum_seed_stage),
                        "upstream_solver_success": bool(
                            upstream_success[seed_index].item()
                        ),
                        "success_source": (
                            "upstream_final"
                            if bool(upstream_success[seed_index].item())
                            else (
                                "strict_contact_boundary_projection"
                                if bool(strict_bodex_success[seed_index].item())
                                else "curriculum_precontact_acceptance"
                            )
                        ),
                        "contact_boundary_projection": contact_boundary_projection,
                        "stage_seed_acceptance": {
                            "schema": stage_seed_acceptance["schema"],
                            "accepted": bool(
                                stage_seed_acceptance["accepted"][seed_index]
                            ),
                            "maximum_gap_m": float(
                                stage_seed_acceptance.get(
                                    "maximum_gap_m", args.stage1_precontact_max_gap_m
                                )
                            ),
                            "world_signed_contact_m": float(
                                stage_seed_acceptance.get(
                                    "world_signed_contact_m", [0.0] * success.numel()
                                )[seed_index]
                            ),
                            "world_gap_m": float(
                                stage_seed_acceptance.get(
                                    "world_gap_m", [0.0] * success.numel()
                                )[seed_index]
                            ),
                            "positive_world_penetration": bool(
                                stage_seed_acceptance.get(
                                    "positive_world_penetration", [False] * success.numel()
                                )[seed_index]
                            ),
                            "joint_bounds_exact_zero": bool(
                                stage_seed_acceptance.get(
                                    "joint_bounds_exact_zero", [True] * success.numel()
                                )[seed_index]
                            ),
                            "self_collision_exact_zero": bool(
                                stage_seed_acceptance.get(
                                    "self_collision_exact_zero", [True] * success.numel()
                                )[seed_index]
                            ),
                            "grasp_error_max": float(
                                stage_seed_acceptance.get(
                                    "grasp_error_max",
                                    grasp_error.detach()
                                    .reshape(success.numel(), -1)
                                    .max(dim=-1)
                                    .values.cpu()
                                    .tolist(),
                                )[seed_index]
                            ),
                            "distance_error_max": float(
                                stage_seed_acceptance.get(
                                    "distance_error_max",
                                    distance_error.detach()
                                    .reshape(success.numel(), -1)
                                    .max(dim=-1)
                                    .values.cpu()
                                    .tolist(),
                                )[seed_index]
                            ),
                            "grasp_threshold": float(
                                stage_seed_acceptance.get(
                                    "grasp_threshold", args.grasp_threshold
                                )
                            ),
                            "distance_threshold": float(args.distance_threshold),
                        },
                        "grasp_threshold": float(args.grasp_threshold),
                        "distance_threshold": float(args.distance_threshold),
                    },
                }
            )
        print(
            json.dumps(
                {
                    "pair": pair_progress,
                    "pair_count": len(world_pairs),
                    "source_pair_index": pair_index,
                    "accepted_total": len(samples),
                    "rejected": rejected,
                    "solver_success_count": int(success.sum().item()),
                    "grasp_error_min": float(
                        grasp_error.detach().reshape(success.numel(), -1).mean(-1).min().item()
                    ),
                    "grasp_error_mean": float(
                        grasp_error.detach().reshape(success.numel(), -1).mean(-1).mean().item()
                    ),
                    "distance_error_min": float(
                        distance_error.detach().reshape(success.numel(), -1).mean(-1).min().item()
                    ),
                    "distance_error_mean": float(
                        distance_error.detach().reshape(success.numel(), -1).mean(-1).mean().item()
                    ),
                    "curriculum_seed_stage": int(args.curriculum_seed_stage),
                    "stage_seed_acceptance": stage_seed_acceptance,
                    "contact_boundary_projection": contact_boundary_projection,
                    "constraint_breakdown": constraint_breakdown,
                    "seed_constraint_breakdown": seed_constraint_breakdown,
                }
            ),
            flush=True,
        )
    if args.rejected_diagnostics_output is not None:
        diagnostic_payload = {
            "schema": "xhand_bodex_rejected_solver_diagnostics_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "object": args.object,
            "robot_config": str(args.robot_config.resolve()),
            "robot_config_sha256": sha256_file(args.robot_config),
            "solver_parameters": {
                "seeds_per_pair": int(args.seeds_per_pair),
                "source_pair_count": source_pair_count,
                "pair_start": int(args.pair_start),
                "pair_stop_exclusive": selected_pair_indices[-1] + 1,
                "pair_count": len(world_pairs),
                "grad_iters": args.grad_iters,
                "curriculum_seed_stage": int(args.curriculum_seed_stage),
                "stage1_precontact_max_gap_m": float(
                    args.stage1_precontact_max_gap_m
                ),
                "effective_precontact_grasp_threshold": float(
                    stage_precontact_grasp_threshold
                ),
                "max_ge_stage": int(args.max_ge_stage),
                "mesh_ge_query_refresh": mesh_ge_query_refresh,
                "postsolve_contact_projection_iterations": int(
                    args.postsolve_contact_projection_iterations
                ),
                "postsolve_contact_projection_max_penetration_m": float(
                    args.postsolve_contact_projection_max_penetration_m
                ),
                "postsolve_contact_projection_outside_tolerance_m": float(
                    args.postsolve_contact_projection_outside_tolerance_m
                ),
                "retain_best": bool(args.retain_best),
                "arm_seed_regularization": float(args.arm_seed_regularization),
                "grouped_line_search": grouped_line_search,
            },
            "rows": rejected_diagnostics,
        }
        args.rejected_diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.rejected_diagnostics_output.write_text(
            json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n"
        )
    if not samples:
        profile_name = (
            "safe precontact" if args.curriculum_seed_stage <= 3 else "lift-ready"
        )
        raise RuntimeError(
            f"BODex produced no {profile_name} bimanual samples for curriculum "
            f"stage {args.curriculum_seed_stage}: {rejected}"
        )
    supported_curriculum_stages = (
        list(range(args.curriculum_seed_stage, 4))
        if args.curriculum_seed_stage <= 3
        else list(range(1, 7))
    )
    payload = {
        "schema": "xhand_bodex_bimanual_solver_results_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "stage_seed_profile": int(args.curriculum_seed_stage),
        "supported_curriculum_stages": supported_curriculum_stages,
        "requires_lift_ik": bool(args.curriculum_seed_stage >= 4),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "paired_seeds": str(args.paired_seeds.resolve()),
        "paired_seeds_sha256": sha256_file(args.paired_seeds),
        "robot_config": str(args.robot_config.resolve()),
        "robot_config_sha256": sha256_file(args.robot_config),
        "object_urdf": str(object_urdf_path.resolve()),
        "object_urdf_sha256": sha256_file(object_urdf_path),
        "fixed_base_gradient_config": str(fixed_base_gradient_path.resolve()),
        "fixed_base_gradient_config_sha256": sha256_file(fixed_base_gradient_path),
        "bodex_source": source,
        "joint_bimanual_optimization": True,
        "combined_grasp_matrix": True,
        "independent_single_hand_pairing": False,
        "object_position_world": object_position,
        "object_quaternion_wxyz": object_quaternion,
        "solver_parameters": {
            "device": args.device,
            "solver_seed": args.solver_seed,
            "seeds_per_pair": args.seeds_per_pair,
            "source_pair_count": source_pair_count,
            "pair_start": int(args.pair_start),
            "pair_stop_exclusive": selected_pair_indices[-1] + 1,
            "pair_count": len(world_pairs),
            "translation_jitter_m": args.translation_jitter_m,
            "rotation_jitter_deg": args.rotation_jitter_deg,
            "lift_height_m": args.lift_height_m,
            "grasp_threshold": args.grasp_threshold,
            "distance_threshold": args.distance_threshold,
            "curriculum_seed_stage": int(args.curriculum_seed_stage),
            "stage1_precontact_max_gap_m": float(
                args.stage1_precontact_max_gap_m
            ),
            "effective_precontact_grasp_threshold": float(
                stage_precontact_grasp_threshold
            ),
            "grad_iters": args.grad_iters,
            "max_ge_stage": int(args.max_ge_stage),
            "mesh_ge_query_refresh": mesh_ge_query_refresh,
            "reluqp_device_alignment": reluqp_device_alignment,
            "postsolve_contact_projection_iterations": int(
                args.postsolve_contact_projection_iterations
            ),
            "postsolve_contact_projection_max_penetration_m": float(
                args.postsolve_contact_projection_max_penetration_m
            ),
            "postsolve_contact_projection_outside_tolerance_m": float(
                args.postsolve_contact_projection_outside_tolerance_m
            ),
            "use_particle_opt": bool(args.use_particle_opt),
            "retain_best": bool(args.retain_best),
            "arm_seed_regularization": float(args.arm_seed_regularization),
            "grouped_line_search": grouped_line_search,
            "gradient_optimizer": "BODex.LBFGSOpt",
            "seed_ik": seed_ik,
            "hand_seed_diversity": hand_seed_diversity,
            "ik_initial_pose": ik_initial_pose,
            "coordinate_seed_replay": coordinate_seed_replay_source,
            "coordinate_seed_replay_repeat_first_pair": bool(
                args.coordinate_seed_replay_repeat_first_pair
            ),
            "left_contact_point_names": left_contacts,
            "right_contact_point_names": right_contacts,
            "physical_contact_mesh_names": physical_contact_meshes,
            "self_collision_sphere_override_source": self_collision_override_source,
            "solver_self_collision_sphere_overrides": solver_collision_overrides,
        },
        "attempt_count": len(world_pairs) * args.seeds_per_pair,
        "accepted_count": len(samples),
        "rejected": rejected,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {key: value for key, value in payload.items() if key != "samples"},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output.resolve()), "accepted_count": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
