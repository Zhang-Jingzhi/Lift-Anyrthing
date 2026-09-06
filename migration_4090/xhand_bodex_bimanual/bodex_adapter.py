"""Small, testable adaptations around the unmodified upstream BODex solver."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from .paired_surface import PairedSurfaceSeed


def bimanual_pressure_constraints(
    left_contact_count: int,
    right_contact_count: int,
    *,
    total_lower: float = 1.0,
    per_side_lower: float = 0.35,
) -> list[list[object]]:
    """Require non-zero pressure from both sides in BODex's QP."""

    if min(left_contact_count, right_contact_count) <= 0:
        raise ValueError("both hands need at least one configured contact")
    left = list(range(left_contact_count))
    right = list(range(left_contact_count, left_contact_count + right_contact_count))
    return [
        [left + right, float(total_lower)],
        [left, float(per_side_lower)],
        [right, float(per_side_lower)],
    ]


def configure_joint_bimanual_seed(
    manip_config: dict[str, Any],
    pair: PairedSurfaceSeed,
    *,
    initial_q: Sequence[float],
    left_contact_names: Sequence[str],
    right_contact_names: Sequence[str],
) -> dict[str, Any]:
    """Inject one explicit paired seed so upstream skips its single-hand sampler."""

    configured = deepcopy(manip_config)
    seeder = configured.setdefault("seeder_cfg", {})
    seeder["t"] = [list(pair.left_position_m), list(pair.right_position_m)]
    seeder["r"] = [list(pair.left_rotation_6d), list(pair.right_rotation_6d)]
    seeder["q"] = [float(value) for value in initial_q]
    seeder["load_path"] = None
    contacts = list(left_contact_names) + list(right_contact_names)
    strategy = configured.setdefault("grasp_contact_strategy", {})
    strategy["contact_points_name"] = contacts
    ge_param = configured.setdefault("grasp_cfg", {}).setdefault("ge_param", {})
    ge_param["pressure_constraints"] = bimanual_pressure_constraints(
        len(left_contact_names), len(right_contact_names)
    )
    configured["xhand_bimanual_contract"] = {
        "joint_bimanual_optimization": True,
        "combined_grasp_matrix": True,
        "independent_single_hand_pairing": False,
        "transferred_link_count": 2,
        "contact_counts_by_side": {
            "left": len(left_contact_names),
            "right": len(right_contact_names),
        },
    }
    return configured


def validate_bodex_result_shapes(
    *,
    solution_shape: Sequence[int],
    contact_point_shape: Sequence[int],
    contact_frame_shape: Sequence[int],
    left_contact_count: int,
    right_contact_count: int,
) -> dict[str, Any]:
    total_contacts = left_contact_count + right_contact_count
    if len(solution_shape) < 3 or int(solution_shape[-2]) < 2:
        raise ValueError("BODex solution must contain at least pregrasp and grasp poses")
    if tuple(contact_point_shape[-2:]) != (total_contacts, 3):
        raise ValueError("BODex contact_point does not contain both hands")
    if tuple(contact_frame_shape[-3:]) != (total_contacts, 3, 3):
        raise ValueError("BODex contact_frame does not contain both hands")
    return {
        "success": True,
        "optimizer": "BODex.GraspSolver",
        "joint_bimanual_optimization": True,
        "combined_grasp_matrix": True,
        "independent_single_hand_pairing": False,
        "contact_counts_by_side": {
            "left": int(left_contact_count),
            "right": int(right_contact_count),
        },
        "grasp_matrix_shape": [int(solution_shape[0]), total_contacts, 6, 3],
        "solution_shape": [int(value) for value in solution_shape],
        "contact_point_shape": [int(value) for value in contact_point_shape],
        "contact_frame_shape": [int(value) for value in contact_frame_shape],
    }
