#!/usr/bin/env python3
"""Build an append-only collision-safe RL nominal bank.

The v3 bank copied a legacy closed-hand posture into a new canonical object
frame.  That posture is not a valid nominal for the locked meshes: a static
mesh audit finds tens of millimetres of hand/object overlap before PPO has a
chance to act.  v4 keeps the old data immutable, retargets all three arm
phases with the actual locked extents, and starts the hands open.  PPO must
close them through its residual actions; no v4 file is an accepted grasp.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from migration_4090.generate_xhand_fullbody_grasps_v1 import (  # noqa: E402
    fixed_solver,
    geometric_metrics,
    hand_mesh_points,
    link_points,
)
from migration_4090.xhand_strict_contract import FINAL_XHAND_RUNTIME_URDF  # noqa: E402
from migration_4090.xhand_rl_embedded.build_corrected_nominals import (  # noqa: E402
    _ik_to_full_q,
    _solve_targets,
)


OBJECTS = ("sphere", "sphere_small", "cracker_large", "cracker", "pyramid", "cube")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_hand_values(q: list[float], reference: list[float], joint_names: list[str]) -> list[float]:
    result = list(q)
    for index, name in enumerate(joint_names):
        if "_hand_" in name:
            result[index] = float(reference[index])
    return result


def phase_mesh_metrics(
    mesh: trimesh.Trimesh,
    q: list[float],
    robot,
    left_points: dict[str, np.ndarray],
    right_points: dict[str, np.ndarray],
    link_indices: dict[str, int],
    joint_names: list[str],
) -> dict:
    by_name = dict(zip(joint_names, q))
    ik_q = np.asarray([by_name[name] for name in robot.joints.actuated_names], dtype=np.float32)
    left = link_points(robot, ik_q, left_points, link_indices)
    right = link_points(robot, ik_q, right_points, link_indices)
    # PhysX uses a 4.5 mm contact offset in the locked manifest.  The legacy
    # generator's 20 mm proximity band was useful for ranking candidates but
    # is not evidence that PhysX will generate a force.  Use the runtime
    # contact band here so a nominal cannot be marked "safe" while its hands
    # are still centimetres away from the mesh.
    return geometric_metrics(mesh, None, left, right, contact_m=0.0045)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lateral-margin", type=float, default=0.07)
    parser.add_argument("--pregrasp-margin", type=float, default=0.12)
    parser.add_argument("--lift-height", type=float, default=0.08)
    parser.add_argument(
        "--closed-hand-grasp",
        action="store_true",
        help=(
            "keep the source nominal's grasp/lift finger posture while "
            "retargeting the arms to the locked mesh; this is still only a "
            "non-accepted RL warm start"
        ),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    solver, _ = fixed_solver()
    robot = solver.robot
    np.random.seed(20260826)
    points = hand_mesh_points(FINAL_XHAND_RUNTIME_URDF, count_per_link=128)
    left_points = {name: value for name, value in points.items() if name.startswith("left_hand_")}
    right_points = {name: value for name, value in points.items() if name.startswith("right_hand_")}
    link_indices = {
        name: robot.links.names.index(name)
        for name in (*left_points, *right_points)
    }

    index = {
        "schema": "xhand_rl_meshaware_nominal_index_v4d",
        "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v4d",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "legacy_data_reused_as_accepted": False,
        "formal_validation_pending": True,
        "nominal_contract": (
            "collision_audited_closed_hand_residual_seed"
            if args.closed_hand_grasp
            else "collision_safe_open_hand_residual_seed"
        ),
        "objects": {},
    }
    for object_name in OBJECTS:
        source_path = (args.source_dir / f"{object_name}.pt").resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        source = copy.deepcopy(source_payload["samples"][0])
        row = manifest["objects"][object_name]
        extents = np.asarray(row["extents_m"], dtype=np.float32)
        object_z = float(manifest["table_top_z_m"]) + 0.5 * float(extents[2])
        half_width = 0.5 * float(extents[1])
        # The large parcel has a broad lateral face.  The generic 230 mm
        # wrist cap still leaves the open fingers inside that mesh, so give it
        # a wider, independently auditable approach band.  This is a nominal
        # seed adjustment only; no old retry is changed.
        lateral_margin = 0.13 if object_name == "cracker_large" else float(args.lateral_margin)
        grasp_y = min(0.32, max(0.14, half_width + lateral_margin))
        if args.closed_hand_grasp:
            # The TCP is not the fingertip contact location.  For the locked
            # wide parcel and enlarged pyramid, the generic half-width+
            # margin target places both wrists outside the usable finger
            # envelope; the closed-hand audit then reports zero contacts even
            # though the IK target is numerically exact.  These values were
            # selected by an append-only mesh sweep: they yield bilateral
            # contact without penetration for the current meshes.  The small
            # sphere likewise needs a closer lateral target because its
            # 199.5-mm diameter is much smaller than the hand span.
            # v4c used a 20-mm proximity count as a contact proxy.  A
            # read-only mesh-distance sweep showed that several of those
            # nominal wrists were still 5--30 mm from the actual collision
            # surface, so PhysX reported no force even with phase-2 residuals
            # disabled.  v4d moves each closed-hand seed into a sub-mm
            # geometric contact basin; the embedded penetration gate remains
            # authoritative and can reject any unsuitable preload.
            grasp_y = {
                "sphere": 0.160,
                "sphere_small": 0.100,
                "cracker_large": 0.190,
                "cracker": 0.140,
                "pyramid": 0.120,
                "cube": 0.100,
            }.get(object_name, grasp_y)
        pregrasp_y = grasp_y + float(args.pregrasp_margin)
        specs = {
            "pregrasp": np.asarray([[0.5, pregrasp_y, object_z], [0.5, -pregrasp_y, object_z]], dtype=np.float32),
            "grasp": np.asarray([[0.5, grasp_y, object_z], [0.5, -grasp_y, object_z]], dtype=np.float32),
            "lift": np.asarray([[0.5, grasp_y, object_z + args.lift_height], [0.5, -grasp_y, object_z + args.lift_height]], dtype=np.float32),
        }
        source_joint_names = list(source["joint_names"])
        source_pregrasp = list(source["pregrasp_full_body_q"])
        source_grasp = list(source.get("full_body_q", source_pregrasp))
        source_lift = list(source.get("lift_full_body_q", source_grasp))

        # Solve arm targets with the open pregrasp hand posture.  The hand
        # values are then explicitly restored so the retargeter cannot import
        # the invalid legacy closure into the new canonical frame.
        phase_q: dict[str, list[float]] = {}
        phase_tcp: dict[str, list[list[float]]] = {}
        phase_source = copy.deepcopy(source)
        phase_source["full_body_q"] = source_pregrasp
        for phase in ("pregrasp", "grasp", "lift"):
            solved, actual = _solve_targets(solver, specs[phase], phase_source)
            # The original v4 bank intentionally kept every phase open.  That
            # makes the formal environment safe, but it leaves PPO with a
            # sparse 38-D closure problem: for some locked meshes even the
            # open grasp posture has no hand/mesh contact.  The optional
            # closed-hand mode supplies only the finger posture from the
            # legacy-derived warm start; arm joints and object geometry are
            # still retargeted here, and every result remains subject to all
            # embedded gates.
            hand_reference = source_pregrasp
            if args.closed_hand_grasp and object_name != "sphere_small" and phase == "grasp":
                hand_reference = source_grasp
            elif args.closed_hand_grasp and object_name != "sphere_small" and phase == "lift":
                hand_reference = source_lift
            solved = replace_hand_values(solved, hand_reference, source_joint_names)
            phase_q[phase] = solved
            phase_tcp[phase] = actual

        sample = copy.deepcopy(source)
        sample.update(
            {
                "object_name": object_name,
                "object_id": object_name,
                "base_object_name": object_name,
                "object_mesh_path": row["mesh"]["path"],
                "object_transform": {
                    "scale_mode": "locked_manifest_extents",
                    "uniform_scale": 1.0,
                    "canonical_rotation": np.eye(3).tolist(),
                    "extents_m": [float(value) for value in extents],
                    "legacy_source": str(source_path),
                },
                "nominal_pose_lock": True,
                "nominal_object_position_world": [0.5, 0.0, object_z],
                "nominal_object_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "full_body_q": phase_q["grasp"],
                "pregrasp_full_body_q": phase_q["pregrasp"],
                "lift_full_body_q": phase_q["lift"],
                "target_tcp_positions_world": specs["grasp"].tolist(),
                "pregrasp_target_tcp_positions_world": specs["pregrasp"].tolist(),
                "lift_target_tcp_positions_world": specs["lift"].tolist(),
                "achieved_tcp_positions_world": phase_tcp["grasp"],
                "pregrasp_achieved_tcp_positions_world": phase_tcp["pregrasp"],
                "lift_achieved_tcp_positions_world": phase_tcp["lift"],
                "generation_backend": (
                    "meshaware_closed_hand_seed_nominal_v4d_preload"
                    if args.closed_hand_grasp
                    else "meshaware_open_hand_nominal_v4"
                ),
                "legacy_source_file": str(source_path),
                "legacy_source_sha256": sha256(source_path),
                "legacy_data_reused_as_accepted": False,
                "isaaclab_physical_validated": False,
                "isaac_gym_physical_validated": False,
                "strict_acceptance": False,
                "formal_validation_pending": True,
                "geometry_pass": False,
                "hand_closure_is_rl_residual": True,
                "closed_hand_grasp_warm_start": bool(args.closed_hand_grasp),
            }
        )
        # Recompute static mesh metrics from the actual locked mesh and current
        # XHand collision meshes; never copy v3's stale metrics.
        mesh = trimesh.load_mesh(row["mesh"]["path"], force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.apply_translation([0.5, 0.0, object_z])
        audit = {
            phase: phase_mesh_metrics(mesh, phase_q[phase], robot, left_points, right_points, link_indices, source_joint_names)
            for phase in ("pregrasp", "grasp", "lift")
        }
        sample["nominal_geometry_audit"] = audit
        sample["metrics"] = audit["grasp"]
        sample["nominal_collision_safe"] = all(
            max(float(values["left_penetration_mm"]), float(values["right_penetration_mm"])) <= 0.5
            and (
                object_name == "sphere_small"
                or (
                    int(values.get("left_contact_points", 0)) > 0
                    and int(values.get("right_contact_points", 0)) > 0
                )
            )
            and (
                object_name == "sphere_small"
                or (
                    float(values.get("left_min_surface_distance_mm", float("inf"))) <= 4.5
                    and float(values.get("right_min_surface_distance_mm", float("inf"))) <= 4.5
                )
            )
            for values in audit.values()
        )

        out_path = output_dir / f"{object_name}.pt"
        torch.save(
            {
                "schema": "xhand_rl_meshaware_nominal_v4d",
                "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v4d",
                "object": object_name,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "source_nominal": str(source_path),
                "source_nominal_sha256": sha256(source_path),
                "legacy_data_reused_as_accepted": False,
                "strict_validated": False,
                "formal_validation_pending": True,
                "nominal_contract": (
                    "collision_audited_closed_hand_residual_seed"
                    if args.closed_hand_grasp
                    else "collision_safe_open_hand_residual_seed"
                ),
                "samples": [sample],
            },
            out_path,
        )
        index["objects"][object_name] = {
            "path": str(out_path),
            "sha256": sha256(out_path),
            "target_tcp_positions_world": specs["grasp"].tolist(),
            "nominal_collision_safe": sample["nominal_collision_safe"],
            "nominal_geometry_audit": audit,
        }
        print(json.dumps({"object": object_name, "collision_safe": sample["nominal_collision_safe"], "audit": audit}, separators=(",", ":")), flush=True)
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
