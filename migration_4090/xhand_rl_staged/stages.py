"""Immutable curriculum definitions and checkpoint-compatible action masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class StageSpec:
    stage_id: int
    name: str
    episode_length_s: float
    phase_fractions: tuple[float, float, float, float]
    active_action_group: str
    lift_target_m: float
    required_bilateral_steps: int
    contact_presence_fraction_min: float
    contact_groups_per_side_min: int
    linear_speed_max_m_s: float
    angular_speed_max_rad_s: float
    stable_hold_steps: int
    penetration_max_m: float
    reset_xy_noise_m: float
    reset_yaw_noise_rad: float
    reset_joint_noise_rad: float
    residual_integration: float
    residual_limit_rad: float
    required_gates: tuple[str, ...]
    # Optional curriculum refinements.  Defaults preserve every historical
    # stage exactly.  Stage-3A can require asymmetric bimanual coverage (for
    # example at least two groups per hand and five across both hands), allow
    # a small no-lift settling tolerance, and use a recoverable terminal
    # penetration gate while keeping Stage 6 unchanged.
    contact_groups_total_min: int = 0
    stability_height_tolerance_m: float = 0.0
    penetration_gate_mode: str = "historical_max"
    stability_gate_mode: str = "current"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        1,
        "bilateral_contact",
        2.0,
        (0.20, 0.50, 0.0, 0.30),
        "hands",
        0.0,
        4,
        0.0,
        1,
        float("inf"),
        float("inf"),
        0,
        0.020,
        0.002,
        0.03,
        0.003,
        0.015,
        0.12,
        ("finite_state", "bilateral_contact"),
    ),
    StageSpec(
        2,
        "contact_continuity",
        3.2,
        (0.15, 0.45, 0.0, 0.40),
        # Preserve the BODex arm trajectory through the complete contact
        # curriculum.  A measured stage-2 expansion to distal-arm residuals
        # reduced deterministic success from about 70% to 62.5%, even after
        # zeroing the newly exposed actor rows; arm refinement therefore stays
        # locked until the stable-grasp stage.
        "hands",
        0.0,
        24,
        0.80,
        2,
        float("inf"),
        float("inf"),
        0,
        0.010,
        0.003,
        0.045,
        0.0045,
        0.015,
        0.15,
        (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
        ),
    ),
    StageSpec(
        3,
        "stable_grasp",
        3.2,
        (0.15, 0.35, 0.0, 0.50),
        "hands_and_distal_arms",
        0.0,
        32,
        0.85,
        3,
        0.05,
        0.50,
        32,
        0.005,
        0.005,
        0.08,
        0.008,
        0.025,
        0.25,
        (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
            "stable_object",
            "catastrophic_penetration",
        ),
    ),
    StageSpec(
        4,
        "micro_lift",
        4.0,
        (0.15, 0.30, 0.25, 0.30),
        "all",
        0.01,
        32,
        0.85,
        3,
        0.05,
        0.50,
        32,
        0.003,
        0.008,
        0.12,
        0.012,
        0.030,
        0.32,
        (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
            "lift_height",
            "stable_lift_hold",
            "catastrophic_penetration",
        ),
    ),
    StageSpec(
        5,
        "partial_lift",
        5.0,
        (0.12, 0.28, 0.30, 0.30),
        "all",
        0.03,
        48,
        0.88,
        3,
        0.04,
        0.40,
        63,
        0.0015,
        0.012,
        0.18,
        0.018,
        0.030,
        0.36,
        (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
            "lift_height",
            "stable_lift_hold",
            "catastrophic_penetration",
        ),
    ),
    StageSpec(
        6,
        "final_lift",
        6.0,
        (0.10, 0.25, 0.30, 0.35),
        "all",
        0.05,
        64,
        0.90,
        3,
        0.03,
        0.30,
        125,
        0.0005,
        0.015,
        0.25,
        0.025,
        0.030,
        0.40,
        (
            "finite_state",
            "bilateral_contact",
            "contact_continuity",
            "distributed_contacts",
            "lift_height",
            "stable_lift_hold",
            "strict_penetration",
            "no_teleport",
        ),
    ),
)


POLICY_STEP_SECONDS = 0.008
PPO_ROLLOUT_STEPS_PER_ENV = 64


def minimum_complete_episode_iterations(stage_id: int) -> int:
    """Iterations required for every new chunk to observe one terminal step."""

    episode_steps = get_stage(stage_id).episode_length_s / POLICY_STEP_SECONDS
    return math.ceil(episode_steps / PPO_ROLLOUT_STEPS_PER_ENV)


def get_stage(stage_id: int) -> StageSpec:
    if not 1 <= int(stage_id) <= len(STAGES):
        raise ValueError(f"stage must be in [1, {len(STAGES)}]")
    return STAGES[int(stage_id) - 1]


def stage_with_overrides(
    stage_id: int,
    *,
    contact_groups_per_side_min: int | None = None,
    contact_groups_total_min: int | None = None,
    penetration_max_m: float | None = None,
    stable_hold_steps: int | None = None,
    stability_height_tolerance_m: float | None = None,
    penetration_gate_mode: str | None = None,
    stability_gate_mode: str | None = None,
) -> StageSpec:
    """Return a validated experimental curriculum spec.

    This is intentionally opt-in.  Omitting every override returns the exact
    immutable production stage, so existing checkpoints and reports retain
    their historical semantics.
    """

    spec = get_stage(stage_id)
    updates: dict[str, object] = {}
    if contact_groups_per_side_min is not None:
        if int(contact_groups_per_side_min) <= 0:
            raise ValueError("contact groups per side must be positive")
        updates["contact_groups_per_side_min"] = int(contact_groups_per_side_min)
    if contact_groups_total_min is not None:
        if int(contact_groups_total_min) < 0:
            raise ValueError("total contact groups must be non-negative")
        updates["contact_groups_total_min"] = int(contact_groups_total_min)
    if penetration_max_m is not None:
        if not math.isfinite(float(penetration_max_m)) or float(penetration_max_m) <= 0.0:
            raise ValueError("penetration maximum must be finite and positive")
        updates["penetration_max_m"] = float(penetration_max_m)
    if stable_hold_steps is not None:
        if int(stable_hold_steps) <= 0:
            raise ValueError("stable hold steps must be positive")
        updates["stable_hold_steps"] = int(stable_hold_steps)
    if stability_height_tolerance_m is not None:
        tolerance = float(stability_height_tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("stability height tolerance must be finite and non-negative")
        updates["stability_height_tolerance_m"] = tolerance
    if penetration_gate_mode is not None:
        if penetration_gate_mode not in ("historical_max", "current"):
            raise ValueError("penetration gate mode must be historical_max or current")
        updates["penetration_gate_mode"] = penetration_gate_mode
    if stability_gate_mode is not None:
        if stability_gate_mode not in ("current", "historical_max"):
            raise ValueError("stability gate mode must be current or historical_max")
        updates["stability_gate_mode"] = stability_gate_mode
    candidate = replace(spec, **updates)
    maximum_groups = 2 * 6
    if candidate.contact_groups_total_min > maximum_groups:
        raise ValueError("total contact groups exceed the twelve available groups")
    minimum_implied_total = 2 * candidate.contact_groups_per_side_min
    if candidate.contact_groups_total_min and candidate.contact_groups_total_min < minimum_implied_total:
        raise ValueError(
            "total contact groups cannot be lower than twice the per-side minimum"
        )
    return candidate


def action_mask_for_stage(
    joint_names: Sequence[str], stage_id: int, *, device: str | torch.device
) -> torch.Tensor:
    return action_mask_for_group(
        joint_names,
        get_stage(stage_id).active_action_group,
        device=device,
    )


def action_mask_for_group(
    joint_names: Sequence[str],
    active_action_group: str,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    """Return a checkpoint-compatible mask for an explicit action group.

    Staged retries sometimes need a conservative transition that keeps the
    formal Stage-3 gates while exposing fewer residual rows than the immutable
    production stage definition.  Making the override explicit preserves the
    38D policy ABI and records exactly which joints PhysX received.
    """

    if active_action_group not in (
        "hands",
        "distal_wrist",
        "hands_and_distal_arms",
        "all",
    ):
        raise ValueError(f"unknown active action group: {active_action_group}")
    mask = torch.zeros(len(joint_names), dtype=torch.float32, device=device)
    for index, name in enumerate(joint_names):
        if active_action_group in ("hands", "hands_and_distal_arms", "all") and "_hand_" in name:
            mask[index] = 1.0
        elif active_action_group == "all":
            mask[index] = 1.0
        elif active_action_group in ("distal_wrist", "hands_and_distal_arms") and name.endswith(("_j5", "_j6", "_j7")):
            mask[index] = 1.0
    return mask


def stage_lift_scale(stage_id: int, nominal_lift_m: float = 0.05) -> float:
    target = get_stage(stage_id).lift_target_m
    if target <= 0.0:
        return 0.0
    return min(target / max(nominal_lift_m, 1.0e-8), 1.0)
