"""Pure embedded gate evaluation shared by the environment and exporter."""

from __future__ import annotations

from typing import Any


PHYSICS_GATE_NAMES = (
    "embedded_isaaclab_episode",
    "finite_state",
    "bilateral_contact_continuity",
    "arm_joint_lift",
    "lift_height",
    "gravity_hold",
    "translation_disturbance",
    "rotation_disturbance",
    "physx_penetration",
    "formal_force_closure",
    "single_hand_ablations",
)


def evaluate_physics_metrics(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
    translation = list(metrics.get("translation_disturbances_m", ()))
    rotation = list(metrics.get("rotation_disturbances_rad", ()))
    continuity = metrics.get("contact_presence_fraction", {})
    inactive = metrics.get("inactive_hand_contact_fraction", {})
    single_success = metrics.get("single_hand_grasp_succeeded", {})
    return {
        "embedded_isaaclab_episode": metrics.get("validator") == "isaac_lab_embedded_rl_v2",
        "finite_state": metrics.get("finite_state_pass") is True,
        "bilateral_contact_continuity": bool(
            continuity.get("left", 0.0) >= thresholds["contact_presence_fraction_min"]
            and continuity.get("right", 0.0) >= thresholds["contact_presence_fraction_min"]
        ),
        "arm_joint_lift": bool(
            metrics.get("lift_mode") == "arm_joint_trajectory"
            and metrics.get("arm_joint_target_delta_l2_rad", 0.0) > 1.0e-4
            and metrics.get("robot_root_teleported_during_grasp_or_lift") is False
            and metrics.get("object_teleported_during_grasp_or_lift") is False
        ),
        "lift_height": metrics.get("lift_displacement_m", float("-inf")) >= thresholds["lift_min_m"],
        "gravity_hold": bool(
            metrics.get("gravity_drift_m", float("inf")) <= thresholds["gravity_drift_max_m"]
            and metrics.get("hold_max_drift_m", float("inf")) <= thresholds["gravity_drift_max_m"]
        ),
        "translation_disturbance": bool(
            len(translation) == 6
            and max(translation, default=float("inf")) <= thresholds["translation_disturbance_max_m"]
        ),
        "rotation_disturbance": bool(
            len(rotation) == 6
            and max(rotation, default=float("inf")) <= thresholds["rotation_disturbance_max_rad"]
        ),
        "physx_penetration": bool(
            metrics.get("physx_penetration_measured") is True
            and metrics.get("maximum_physx_contact_penetration_m", float("inf"))
            <= thresholds["physx_penetration_max_m"]
        ),
        "formal_force_closure": metrics.get("force_closure_pass") is True,
        "single_hand_ablations": bool(
            inactive.get("left_only", 1.0) <= thresholds["inactive_hand_contact_fraction_max"]
            and inactive.get("right_only", 1.0) <= thresholds["inactive_hand_contact_fraction_max"]
            and single_success.get("left") is False
            and single_success.get("right") is False
        ),
    }


def all_physics_gates_pass(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return all(evaluate_physics_metrics(metrics, thresholds).values())
