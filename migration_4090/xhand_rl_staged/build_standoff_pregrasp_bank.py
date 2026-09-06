#!/usr/bin/env python3
"""Derive collision-audited wrist-standoff pregrasps from a BODex bank.

The jointly optimized BODex grasp remains immutable.  Only the arm joints in
``pregrasp_full_body_q`` are recomputed so that both palms start a requested
distance farther from the object.  The original BODex open-hand posture is
copied exactly, and the complete pregrasp-to-grasp interpolation is diagnosed
against the locked object mesh before a derived bank is written.  PhysX smoke
tests remain authoritative because the static mesh proxy can report intended
closed-finger contact as penetration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import torch
import trimesh
import yourdfpy

MIGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(MIGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MIGRATION_ROOT))

from generate_xhand_compact_candidate_bank_v2 import (  # noqa: E402
    TARGET_LINKS,
    achieved_metrics,
    calibrate_ik_to_actual_rotations,
    make_reusable_pose_solver,
    solve_targets,
)
from generate_xhand_fullbody_grasps_v1 import (  # noqa: E402
    IK_URDF,
    ensure_fixed_ik_urdf,
    geometric_metrics,
    hand_mesh_points,
    link_points,
)
from xhand_strict_contract import FINAL_XHAND_RUNTIME_URDF  # noqa: E402

from migration_4090.xhand_bodex_bimanual.contracts import (  # noqa: E402
    load_bodex_bank,
    sha256_file,
    validate_bodex_bank,
)


DERIVATION_SCHEMA = "xhand_bodex_arm_standoff_pregrasp_v1"


def load_fixed_kinematic_robot() -> Any:
    """Load only the URDF kinematic tree, without multi-million-triangle visuals."""

    ensure_fixed_ik_urdf()
    urdf = yourdfpy.URDF.load(
        str(IK_URDF),
        build_scene_graph=True,
        build_collision_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    return pk.Robot.from_urdf(urdf)


def outward_unit_vectors(
    palm_positions: np.ndarray,
    object_center: np.ndarray,
    *,
    horizontal_only: bool,
) -> np.ndarray:
    """Return one finite object-to-palm unit vector for each palm."""

    palm_positions = np.asarray(palm_positions, dtype=np.float64)
    object_center = np.asarray(object_center, dtype=np.float64)
    if palm_positions.shape != (2, 3) or object_center.shape != (3,):
        raise ValueError("expected two palm positions and one object center")
    directions = palm_positions - object_center[None, :]
    if horizontal_only:
        directions[:, 2] = 0.0
    lengths = np.linalg.norm(directions, axis=-1)
    if not np.isfinite(directions).all() or bool(np.any(lengths <= 1.0e-9)):
        raise ValueError("cannot derive a finite outward palm direction")
    return directions / lengths[:, None]


def preserve_open_hand_posture(
    solved_q: np.ndarray,
    source_pregrasp_q: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    """Keep every BODex pregrasp hand joint bit-for-bit unchanged."""

    solved_q = np.asarray(solved_q, dtype=np.float64).copy()
    source_pregrasp_q = np.asarray(source_pregrasp_q, dtype=np.float64)
    if solved_q.shape != source_pregrasp_q.shape or solved_q.shape != (
        len(joint_names),
    ):
        raise ValueError("joint vectors do not match joint_names")
    for index, name in enumerate(joint_names):
        if "_hand_" in name:
            solved_q[index] = source_pregrasp_q[index]
    return solved_q


def interpolation_rows(
    pregrasp_q: np.ndarray, grasp_q: np.ndarray, *, steps: int
) -> list[np.ndarray]:
    """Match the environment's linear close trajectory, including endpoints."""

    if steps < 2:
        raise ValueError("trajectory audit needs at least two steps")
    pregrasp_q = np.asarray(pregrasp_q, dtype=np.float64)
    grasp_q = np.asarray(grasp_q, dtype=np.float64)
    if pregrasp_q.shape != grasp_q.shape:
        raise ValueError("pregrasp and grasp vectors must have equal shape")
    return [
        (1.0 - alpha) * pregrasp_q + alpha * grasp_q
        for alpha in np.linspace(0.0, 1.0, steps)
    ]


def _world_mesh(sample: dict[str, Any]) -> trimesh.Trimesh:
    mesh_path = Path(sample["object_mesh_path"])
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"object mesh is not triangular: {mesh_path}")
    mesh = mesh.copy()
    mesh.apply_transform(np.asarray(sample["object_pose_world"], dtype=np.float64))
    return mesh


def _mesh_audit(
    *,
    mesh: trimesh.Trimesh,
    trajectory: list[np.ndarray],
    robot: Any,
    left_points: dict[str, np.ndarray],
    right_points: dict[str, np.ndarray],
    link_indices: dict[str, int],
    maximum_penetration_mm: float,
) -> dict[str, Any]:
    rows = []
    maximum_penetration = 0.0
    minimum_hand_clearance = float("inf")
    query_valid = True
    for waypoint_index, q in enumerate(trajectory):
        left = link_points(robot, q, left_points, link_indices)
        right = link_points(robot, q, right_points, link_indices)
        metrics = geometric_metrics(
            mesh,
            None,
            left,
            right,
            contact_m=0.0045,
        )
        rows.append({"waypoint_index": waypoint_index, **metrics})
        maximum_penetration = max(
            maximum_penetration,
            float(metrics["left_penetration_mm"]),
            float(metrics["right_penetration_mm"]),
        )
        minimum_hand_clearance = min(
            minimum_hand_clearance, float(metrics["hand_clearance_mm"])
        )
        query_valid &= bool(metrics["mesh_query_valid"])
    return {
        "mesh_query_valid": query_valid,
        "maximum_sampled_object_penetration_mm": maximum_penetration,
        "minimum_sampled_interhand_clearance_mm": minimum_hand_clearance,
        "maximum_allowed_sampled_penetration_mm": float(maximum_penetration_mm),
        "passed": bool(
            query_valid and maximum_penetration <= float(maximum_penetration_mm)
        ),
        "waypoints": rows,
    }


def _output_name(source: Path, standoff_m: float) -> str:
    millimetres = int(round(1000.0 * standoff_m))
    return f"{source.stem}_pregrasp_standoff_{millimetres:03d}mm_v1.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--standoff-m", type=float, nargs="+", default=(0.005, 0.010, 0.015)
    )
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--wrist-rest-weight", type=float, default=0.03)
    parser.add_argument("--max-position-error-m", type=float, default=0.003)
    parser.add_argument("--minimum-achieved-fraction", type=float, default=0.90)
    parser.add_argument("--trajectory-audit-steps", type=int, default=9)
    parser.add_argument("--mesh-points-per-link", type=int, default=32)
    parser.add_argument("--maximum-sampled-penetration-mm", type=float, default=0.5)
    parser.add_argument(
        "--require-mesh-audit-pass",
        action="store_true",
        help=(
            "reject a derived bank when the conservative static mesh proxy "
            "fails; normally PhysX smoke is the authoritative gate"
        ),
    )
    parser.add_argument("--horizontal-only", action="store_true")
    args = parser.parse_args()

    standoffs = tuple(sorted(set(float(value) for value in args.standoff_m)))
    if not standoffs or any(value <= 0.0 for value in standoffs):
        raise ValueError("standoff distances must be positive")
    if not 0.0 < args.minimum_achieved_fraction <= 1.0:
        raise ValueError("minimum achieved fraction must be in (0, 1]")
    if args.maximum_sampled_penetration_mm < 0.0:
        raise ValueError("maximum sampled penetration must be non-negative")
    if args.mesh_points_per_link <= 0:
        raise ValueError("mesh points per link must be positive")

    source_path = args.bodex_bank.resolve()
    bank = load_bodex_bank(
        source_path,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / _output_name(source_path, standoff) for standoff in standoffs
    ]
    collisions = [path for path in output_paths if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite derived banks: {collisions}")

    robot = load_fixed_kinematic_robot()
    ik_names = list(robot.joints.actuated_names)
    link_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float64)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float64)
    os.environ["XHAND_FORMAL_WRIST_REST_WEIGHT"] = str(args.wrist_rest_weight)

    np.random.seed(20260831)
    sampled_points = hand_mesh_points(
        FINAL_XHAND_RUNTIME_URDF,
        count_per_link=args.mesh_points_per_link,
    )
    left_points = {
        name: points
        for name, points in sampled_points.items()
        if name.startswith("left_hand_")
    }
    right_points = {
        name: points
        for name, points in sampled_points.items()
        if name.startswith("right_hand_")
    }
    point_link_indices = {
        name: robot.links.names.index(name) for name in sampled_points
    }
    if not left_points or not right_points:
        raise RuntimeError("runtime XHand mesh sampling produced no bilateral points")

    source_sha256 = sha256_file(source_path)
    created_at = datetime.now(timezone.utc).isoformat()
    outputs = []
    for standoff_m, output_path in zip(standoffs, output_paths):
        derived = copy.deepcopy(bank)
        diagnostics = []
        for candidate_index, sample in enumerate(derived["samples"]):
            full_names = list(sample["joint_names"])
            grasp_values = dict(zip(full_names, sample["full_body_q"]))
            source_pregrasp_values = dict(
                zip(full_names, sample["pregrasp_full_body_q"])
            )
            missing = [name for name in ik_names if name not in grasp_values]
            if missing:
                raise RuntimeError(
                    f"candidate {candidate_index} lacks IK joints: {missing}"
                )
            grasp_q = np.asarray(
                [grasp_values[name] for name in ik_names], dtype=np.float64
            )
            source_pregrasp_q = np.asarray(
                [source_pregrasp_values[name] for name in ik_names],
                dtype=np.float64,
            )
            object_center = np.asarray(
                sample["object_pose_world"], dtype=np.float64
            )[:3, 3]
            _, _, frame_rotations = calibrate_ik_to_actual_rotations(
                robot, grasp_q, link_indices
            )
            solver = make_reusable_pose_solver(
                robot,
                link_indices,
                ik_to_actual_rotations=frame_rotations,
                palm_direction_weight=args.orientation_weight,
            )
            grasp_fk = robot.forward_kinematics(jnp.asarray(grasp_q))
            grasp_poses = [jaxlie.SE3(grasp_fk[link]) for link in link_indices]
            grasp_positions = np.stack(
                [np.asarray(pose.translation(), dtype=np.float64) for pose in grasp_poses]
            )
            directions = outward_unit_vectors(
                grasp_positions,
                object_center,
                horizontal_only=bool(args.horizontal_only),
            )
            targets = [
                jaxlie.SE3.from_rotation_and_translation(
                    pose.rotation(),
                    jnp.asarray(grasp_positions[side] + standoff_m * directions[side]),
                )
                for side, pose in enumerate(grasp_poses)
            ]
            solved_q = solve_targets(solver, source_pregrasp_q, targets)
            solved_q = preserve_open_hand_posture(
                solved_q, source_pregrasp_q, ik_names
            )
            achieved, position_error, orientation_error = achieved_metrics(
                robot, solved_q, targets, link_indices
            )
            solved_fk = robot.forward_kinematics(jnp.asarray(solved_q))
            solved_positions = np.stack(
                [
                    np.asarray(
                        jaxlie.SE3(solved_fk[link]).translation(), dtype=np.float64
                    )
                    for link in link_indices
                ]
            )
            achieved_projection = np.sum(
                (solved_positions - grasp_positions) * directions, axis=-1
            )
            valid_ik = bool(
                np.isfinite(solved_q).all()
                and np.all(solved_q >= lower - 1.0e-4)
                and np.all(solved_q <= upper + 1.0e-4)
                and max(position_error) <= float(args.max_position_error_m)
                and np.all(
                    achieved_projection
                    >= standoff_m * float(args.minimum_achieved_fraction)
                )
            )
            if not valid_ik:
                raise RuntimeError(
                    f"candidate {candidate_index} standoff IK failed at {standoff_m} m: "
                    f"position_error={position_error}, "
                    f"achieved_projection={achieved_projection.tolist()}"
                )

            trajectory = interpolation_rows(
                solved_q,
                grasp_q,
                steps=args.trajectory_audit_steps,
            )
            mesh_audit = _mesh_audit(
                mesh=_world_mesh(sample),
                trajectory=trajectory,
                robot=robot,
                left_points=left_points,
                right_points=right_points,
                link_indices=point_link_indices,
                maximum_penetration_mm=args.maximum_sampled_penetration_mm,
            )
            if args.require_mesh_audit_pass and not mesh_audit["passed"]:
                raise RuntimeError(
                    f"candidate {candidate_index} failed mesh trajectory audit at "
                    f"{standoff_m} m: {mesh_audit}"
                )

            corrected_values = source_pregrasp_values.copy()
            corrected_values.update(
                {name: float(value) for name, value in zip(ik_names, solved_q)}
            )
            sample["pregrasp_full_body_q"] = [
                float(corrected_values[name]) for name in full_names
            ]
            sample["pregrasp_derivation"] = {
                "schema": DERIVATION_SCHEMA,
                "source_bodex_bank": str(source_path),
                "source_bodex_bank_sha256": source_sha256,
                "requested_standoff_m": standoff_m,
                "outward_direction_mode": (
                    "object_center_to_palm_horizontal"
                    if args.horizontal_only
                    else "object_center_to_palm_3d"
                ),
                "preserved_final_bodex_grasp": True,
                "preserved_open_hand_joint_values": True,
                "solver": "orientation_preserving_two_palm_arm_ik",
                "trajectory_collision_diagnostic": (
                    "sampled_locked_mesh_pregrasp_to_grasp_non_authoritative"
                ),
                "mesh_audit_required_for_write": bool(
                    args.require_mesh_audit_pass
                ),
                "target_palm_positions_world_m": [
                    np.asarray(target.translation()).tolist() for target in targets
                ],
                "achieved_palm_positions_world_m": achieved,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "achieved_outward_projection_m": achieved_projection.tolist(),
                "mesh_audit": mesh_audit,
            }
            diagnostics.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": sample["candidate_id"],
                    "position_error_m": position_error,
                    "achieved_outward_projection_m": achieved_projection.tolist(),
                    "mesh_audit": mesh_audit,
                }
            )

        derived["pregrasp_bank_derivation"] = {
            "schema": DERIVATION_SCHEMA,
            "created_at": created_at,
            "source_bodex_bank": str(source_path),
            "source_bodex_bank_sha256": source_sha256,
            "requested_standoff_m": standoff_m,
            "final_bodex_grasp_fields_unchanged": [
                "full_body_q",
                "lift_full_body_q",
                "bodex_result",
            ],
            "joint_bimanual_bodex_provenance_preserved": True,
            "non_promotional_until_isaac_validated": True,
            "authoritative_collision_gate": "Isaac_PhysX_smoke",
            "diagnostics": diagnostics,
        }
        validate_bodex_bank(
            derived,
            expected_object=args.object,
            verify_source=False,
            intended_stage=2,
        )
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(derived, temporary)
        os.replace(temporary, output_path)
        report_path = output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(derived["pregrasp_bank_derivation"], indent=2) + "\n"
        )
        outputs.append(
            {
                "standoff_m": standoff_m,
                "bank": str(output_path.resolve()),
                "bank_sha256": sha256_file(output_path),
                "report": str(report_path.resolve()),
            }
        )

    print(json.dumps({"source": str(source_path), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
