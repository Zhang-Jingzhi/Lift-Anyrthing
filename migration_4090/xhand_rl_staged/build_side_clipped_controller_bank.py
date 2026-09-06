#!/usr/bin/env python3
"""Build a PhysX-screenable controller bank from genuine bimanual BODex paths.

The raw jointly optimized BODex fields remain immutable.  For each hand, this
tool finds the deepest prefix of the BODex pregrasp-to-grasp trajectory that
still satisfies the sampled mesh penetration limit and has at least one
surface contact.  The two sides may stop at different interpolation fractions;
this prevents the earlier-contacting hand from being driven through the object
while the other hand is still approaching.

The derived targets are written to explicit ``controller_*`` fields.  They are
non-promotional until an Isaac PhysX rollout passes the path-wise lift gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

MIGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(MIGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MIGRATION_ROOT))

from migration_4090.xhand_bodex_bimanual.contracts import (
    load_bodex_bank,
    sha256_file,
    validate_bodex_bank,
)
from migration_4090.xhand_rl_staged.build_standoff_pregrasp_bank import (
    _mesh_audit,
    _world_mesh,
    interpolation_rows,
    load_fixed_kinematic_robot,
)
from generate_xhand_fullbody_grasps_v1 import hand_mesh_points
from xhand_strict_contract import FINAL_XHAND_RUNTIME_URDF


DERIVATION_SCHEMA = "xhand_bodex_side_clipped_controller_bank_v1"


def robot_side_mesh_points(
    urdf_path: Path, *, count_per_link: int = 24
) -> dict[str, dict[str, np.ndarray]]:
    """Sample collision/visual meshes for both arms and hands.

    The original controller-bank audit sampled only ``*_hand_*`` links.  The
    Isaac scene, however, also collides the wrist/forearm links (``left_j*``
    and ``right_j*``).  A controller path can therefore pass the hand-only
    audit while an arm link still pushes the object during approach.  This
    helper intentionally keeps the side split used by ``_mesh_audit`` while
    adding the arm links when explicitly requested.
    """

    if count_per_link <= 0:
        raise ValueError("count_per_link must be positive")
    root = ET.parse(urdf_path).getroot()
    points: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    for link in root.findall("link"):
        name = str(link.get("name", ""))
        side = "left" if name.startswith("left_") else "right" if name.startswith("right_") else None
        if side is None:
            continue
        mesh_node = link.find("collision/geometry/mesh")
        if mesh_node is None:
            mesh_node = link.find("visual/geometry/mesh")
        if mesh_node is None:
            continue
        filename = mesh_node.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        if not mesh_path.is_file():
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
        except Exception:
            continue
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            continue
        points[side][name] = np.asarray(
            mesh.sample(count_per_link), dtype=np.float32
        )
    if not points["left"] or not points["right"]:
        raise RuntimeError("runtime XHand URDF produced no bilateral arm/hand meshes")
    return points


def side_clipped_joint_target(
    pregrasp_q: np.ndarray,
    grasp_q: np.ndarray,
    joint_names: list[str],
    *,
    left_alpha: float,
    right_alpha: float,
) -> np.ndarray:
    """Interpolate the two disjoint robot branches by independent fractions."""

    pregrasp_q = np.asarray(pregrasp_q, dtype=np.float64)
    grasp_q = np.asarray(grasp_q, dtype=np.float64)
    if pregrasp_q.shape != grasp_q.shape or pregrasp_q.shape != (len(joint_names),):
        raise ValueError("joint vectors do not match joint_names")
    if not 0.0 <= left_alpha <= 1.0 or not 0.0 <= right_alpha <= 1.0:
        raise ValueError("side interpolation fractions must be in [0, 1]")
    target = pregrasp_q.copy()
    for index, name in enumerate(joint_names):
        if name.startswith("left_"):
            alpha = left_alpha
        elif name.startswith("right_"):
            alpha = right_alpha
        else:
            raise ValueError(f"cannot assign joint to a robot side: {name}")
        target[index] = pregrasp_q[index] + alpha * (
            grasp_q[index] - pregrasp_q[index]
        )
    return target


def deepest_prefix_safe_contact_waypoints(
    rows: list[dict[str, Any]],
    *,
    side: str,
    maximum_penetration_mm: float,
    minimum_contact_points: int,
) -> list[dict[str, Any]]:
    """Return every prefix-safe contact row, ordered from shallow to deep."""

    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if maximum_penetration_mm < 0.0:
        raise ValueError("maximum penetration must be non-negative")
    if minimum_contact_points <= 0:
        raise ValueError("minimum contact points must be positive")
    penetration_key = f"{side}_penetration_mm"
    contact_key = f"{side}_contact_points"
    prefix_maximum = 0.0
    selected = []
    query_valid = True
    for row in rows:
        query_valid &= bool(row.get("mesh_query_valid", False))
        prefix_maximum = max(prefix_maximum, float(row[penetration_key]))
        if (
            query_valid
            and prefix_maximum <= maximum_penetration_mm
            and int(row[contact_key]) >= minimum_contact_points
        ):
            selected.append({**row, "prefix_maximum_penetration_mm": prefix_maximum})
    return selected


def arm_then_hand_rows(
    pregrasp_q: np.ndarray,
    target_q: np.ndarray,
    joint_names: list[str],
    *,
    arm_steps: int,
    hand_steps: int,
) -> list[np.ndarray]:
    """Match the bridge's open-hand arm approach followed by finger closure."""

    if arm_steps < 2 or hand_steps < 2:
        raise ValueError("arm and hand trajectory audits need at least two steps")
    pregrasp_q = np.asarray(pregrasp_q, dtype=np.float64)
    target_q = np.asarray(target_q, dtype=np.float64)
    if pregrasp_q.shape != target_q.shape or pregrasp_q.shape != (len(joint_names),):
        raise ValueError("joint vectors do not match joint_names")
    arm_mask = np.asarray(["_hand_" not in name for name in joint_names])
    hand_mask = ~arm_mask
    rows = []
    for alpha in np.linspace(0.0, 1.0, arm_steps):
        q = pregrasp_q.copy()
        q[arm_mask] += alpha * (target_q[arm_mask] - pregrasp_q[arm_mask])
        rows.append(q)
    arm_target = rows[-1]
    for alpha in np.linspace(0.0, 1.0, hand_steps)[1:]:
        q = arm_target.copy()
        q[hand_mask] = pregrasp_q[hand_mask] + alpha * (
            target_q[hand_mask] - pregrasp_q[hand_mask]
        )
        rows.append(q)
    return rows


def _side_maximum(audit: dict[str, Any], side: str) -> float:
    return max(
        float(row[f"{side}_penetration_mm"])
        for row in audit["waypoints"]
    )


def _output_report(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["controller_bank_derivation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-steps", type=int, default=65)
    parser.add_argument("--arm-audit-steps", type=int, default=17)
    parser.add_argument("--hand-audit-steps", type=int, default=17)
    parser.add_argument("--mesh-points-per-link", type=int, default=32)
    parser.add_argument(
        "--include-arm-links",
        action="store_true",
        help="audit left_j*/right_j* arm and wrist meshes in addition to hands",
    )
    parser.add_argument("--maximum-sampled-penetration-mm", type=float, default=0.5)
    parser.add_argument("--minimum-contact-points-per-side", type=int, default=1)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite controller bank: {args.output}")
    if args.search_steps < 3:
        raise ValueError("search needs at least three interpolation steps")
    if args.mesh_points_per_link <= 0:
        raise ValueError("mesh points per link must be positive")

    source_path = args.bodex_bank.resolve()
    source = load_bodex_bank(
        source_path,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    derived = copy.deepcopy(source)
    robot = load_fixed_kinematic_robot()
    ik_names = list(robot.joints.actuated_names)
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float64)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float64)

    np.random.seed(20260831)
    if args.include_arm_links:
        sampled_sides = robot_side_mesh_points(
            FINAL_XHAND_RUNTIME_URDF,
            count_per_link=args.mesh_points_per_link,
        )
        left_points = sampled_sides["left"]
        right_points = sampled_sides["right"]
        audit_scope = "arm_and_hand_links"
    else:
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
        audit_scope = "hand_links"
    sampled_for_indices = {**left_points, **right_points}
    point_link_indices = {
        name: robot.links.names.index(name) for name in sampled_for_indices
    }
    if not left_points or not right_points:
        raise RuntimeError("runtime XHand mesh sampling produced no bilateral points")

    maximum_penetration = float(args.maximum_sampled_penetration_mm)
    minimum_contacts = int(args.minimum_contact_points_per_side)
    alphas = np.linspace(0.0, 1.0, args.search_steps)
    source_sha256 = sha256_file(source_path)
    diagnostics = []
    for candidate_index, sample in enumerate(derived["samples"]):
        full_names = list(sample["joint_names"])
        source_sample = source["samples"][candidate_index]
        for immutable_field in ("full_body_q", "lift_full_body_q", "bodex_result"):
            if sample[immutable_field] != source_sample[immutable_field]:
                raise RuntimeError(f"raw BODex field changed before derivation: {immutable_field}")
        name_to_pregrasp = dict(zip(full_names, sample["pregrasp_full_body_q"]))
        name_to_grasp = dict(zip(full_names, sample["full_body_q"]))
        missing = [name for name in ik_names if name not in name_to_grasp]
        if missing:
            raise RuntimeError(f"candidate {candidate_index} lacks IK joints: {missing}")
        pregrasp_q = np.asarray(
            [name_to_pregrasp[name] for name in ik_names], dtype=np.float64
        )
        grasp_q = np.asarray(
            [name_to_grasp[name] for name in ik_names], dtype=np.float64
        )
        search_rows = interpolation_rows(
            pregrasp_q,
            grasp_q,
            steps=args.search_steps,
        )
        search_audit = _mesh_audit(
            mesh=_world_mesh(sample),
            trajectory=search_rows,
            robot=robot,
            left_points=left_points,
            right_points=right_points,
            link_indices=point_link_indices,
            maximum_penetration_mm=maximum_penetration,
        )
        annotated_rows = [
            {**row, "alpha": float(alphas[index])}
            for index, row in enumerate(search_audit["waypoints"])
        ]
        safe_rows = {
            side: deepest_prefix_safe_contact_waypoints(
                annotated_rows,
                side=side,
                maximum_penetration_mm=maximum_penetration,
                minimum_contact_points=minimum_contacts,
            )
            for side in ("left", "right")
        }
        if any(not rows for rows in safe_rows.values()):
            raise RuntimeError(
                f"candidate {candidate_index} has no bilateral side-wise safe contact: "
                f"left={len(safe_rows['left'])}, right={len(safe_rows['right'])}"
            )

        selected_positions = {
            side: len(rows) - 1 for side, rows in safe_rows.items()
        }
        controller_q = None
        controller_audit = None
        while min(selected_positions.values()) >= 0:
            selected = {
                side: safe_rows[side][selected_positions[side]]
                for side in ("left", "right")
            }
            controller_q = side_clipped_joint_target(
                pregrasp_q,
                grasp_q,
                ik_names,
                left_alpha=float(selected["left"]["alpha"]),
                right_alpha=float(selected["right"]["alpha"]),
            )
            controller_rows = arm_then_hand_rows(
                pregrasp_q,
                controller_q,
                ik_names,
                arm_steps=args.arm_audit_steps,
                hand_steps=args.hand_audit_steps,
            )
            controller_audit = _mesh_audit(
                mesh=_world_mesh(sample),
                trajectory=controller_rows,
                robot=robot,
                left_points=left_points,
                right_points=right_points,
                link_indices=point_link_indices,
                maximum_penetration_mm=maximum_penetration,
            )
            if controller_audit["passed"]:
                final = controller_audit["waypoints"][-1]
                if all(
                    int(final[f"{side}_contact_points"]) >= minimum_contacts
                    for side in ("left", "right")
                ):
                    break
            side_to_reduce = max(
                ("left", "right"),
                key=lambda side: _side_maximum(controller_audit, side),
            )
            selected_positions[side_to_reduce] -= 1
        else:
            raise RuntimeError(
                f"candidate {candidate_index} has no safe arm-then-hand controller path"
            )
        assert controller_q is not None and controller_audit is not None
        if not np.isfinite(controller_q).all():
            raise RuntimeError(f"candidate {candidate_index} controller target is non-finite")
        if bool(np.any(controller_q < lower - 1.0e-4)) or bool(
            np.any(controller_q > upper + 1.0e-4)
        ):
            raise RuntimeError(f"candidate {candidate_index} controller target exceeds limits")

        selected = {
            side: safe_rows[side][selected_positions[side]]
            for side in ("left", "right")
        }
        controller_values = name_to_grasp.copy()
        controller_values.update(
            {name: float(value) for name, value in zip(ik_names, controller_q)}
        )
        controller_vector = [float(controller_values[name]) for name in full_names]
        sample["controller_pregrasp_full_body_q"] = list(
            sample["pregrasp_full_body_q"]
        )
        sample["controller_grasp_full_body_q"] = controller_vector
        sample["controller_lift_full_body_q"] = list(controller_vector)
        sample["controller_derivation"] = {
            "schema": DERIVATION_SCHEMA,
            "source_bodex_bank": str(source_path),
            "source_bodex_bank_sha256": source_sha256,
            "raw_bodex_grasp_preserved": True,
            "method": "independent_side_prefix_clip_of_bodex_joint_trajectory",
            "left_alpha": float(selected["left"]["alpha"]),
            "right_alpha": float(selected["right"]["alpha"]),
            "maximum_sampled_penetration_mm": maximum_penetration,
            "minimum_contact_points_per_side": minimum_contacts,
            "controller_path": "open_hand_arm_approach_then_finger_close",
            "audit_scope": audit_scope,
            "controller_path_mesh_audit": controller_audit,
            "non_promotional_until_isaac_validated": True,
        }
        diagnostics.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": sample["candidate_id"],
                "left_alpha": float(selected["left"]["alpha"]),
                "right_alpha": float(selected["right"]["alpha"]),
                "final_contacts": {
                    side: int(
                        controller_audit["waypoints"][-1][
                            f"{side}_contact_points"
                        ]
                    )
                    for side in ("left", "right")
                },
                "controller_path_mesh_audit": controller_audit,
            }
        )

    derived["controller_bank_derivation"] = {
        "schema": DERIVATION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bodex_bank": str(source_path),
        "source_bodex_bank_sha256": source_sha256,
        "raw_bodex_fields_unchanged": [
            "full_body_q",
            "lift_full_body_q",
            "bodex_result",
        ],
        "controller_fields": [
            "controller_pregrasp_full_body_q",
            "controller_grasp_full_body_q",
            "controller_lift_full_body_q",
        ],
        "joint_bimanual_bodex_provenance_preserved": True,
        "non_promotional_until_isaac_validated": True,
        "authoritative_collision_gate": "Isaac_PhysX_smoke",
        "audit_scope": audit_scope,
        "diagnostics": diagnostics,
    }
    validate_bodex_bank(
        derived,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(derived, temporary)
    os.replace(temporary, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(_output_report(derived), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "candidate_count": len(derived["samples"]),
                "alphas": [
                    {
                        "candidate_index": row["candidate_index"],
                        "left": row["left_alpha"],
                        "right": row["right_alpha"],
                    }
                    for row in diagnostics
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
