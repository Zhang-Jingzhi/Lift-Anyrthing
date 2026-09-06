"""Pure-Python reward/exploration profiles for the staged curriculum."""

from __future__ import annotations

import math
from collections.abc import Mapping


REWARD_WEIGHT_FIELDS = (
    "bilateral_contact_reward_weight",
    "terminal_success_weight",
    "penetration_reward_weight",
    "penetration_clear_reward_weight",
    "proximity_reward_weight",
    "contact_diversity_reward_weight",
    "contact_continuity_reward_weight",
    "stable_lift_reward_weight",
    "lift_height_reward_weight",
    "force_closure_reward_weight",
    "stage_gate_progress_reward_weight",
    "stage_action_anchor_penalty_weight",
    "stage_residual_anchor_penalty_weight",
    "distributed_contact_reward_weight",
    "terminal_contact_reward_weight",
)


_BASE_PROFILE = {
    "bilateral_contact_reward_weight": 6.0,
    "terminal_success_weight": 30.0,
    "penetration_reward_weight": -8.0,
    "penetration_clear_reward_weight": 8.0,
    "proximity_reward_weight": 8.0,
    "contact_diversity_reward_weight": 2.5,
    "contact_continuity_reward_weight": 6.0,
    "stable_lift_reward_weight": 0.0,
    "lift_height_reward_weight": 0.0,
    "force_closure_reward_weight": 0.0,
    "stage_gate_progress_reward_weight": 0.0,
    "stage_action_anchor_penalty_weight": 0.0,
    "stage_residual_anchor_penalty_weight": 0.0,
    "distributed_contact_reward_weight": 3.0,
    "terminal_contact_reward_weight": 0.0,
}


def stage_reward_profile(
    stage_id: int,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return the effective, validated shaping weights for one stage.

    Stage 1 deliberately prioritizes the exact terminal intersection of
    bilateral contact and penetration clearance.  Earlier runs over-rewarded
    proximity and reached only 47.3% deterministic success after 20 PPO
    iterations, despite an 80.5% penetration pass rate.
    """

    if not 1 <= int(stage_id) <= 6:
        raise ValueError("stage must be in [1, 6]")
    profile = dict(_BASE_PROFILE)
    if stage_id == 1:
        profile.update(
            {
                "bilateral_contact_reward_weight": 10.0,
                "terminal_success_weight": 600.0,
                "penetration_reward_weight": -2.0,
                "penetration_clear_reward_weight": 1.0,
                "proximity_reward_weight": 2.0,
                "contact_diversity_reward_weight": 0.0,
                "contact_continuity_reward_weight": 0.0,
                "stage_gate_progress_reward_weight": 24.0,
                "stage_action_anchor_penalty_weight": 4.0,
                "stage_residual_anchor_penalty_weight": 12.0,
            }
        )
    elif stage_id == 2:
        profile.update(
            {
                "bilateral_contact_reward_weight": 8.0,
                "terminal_success_weight": 800.0,
                "penetration_reward_weight": -2.0,
                "penetration_clear_reward_weight": 1.0,
                "proximity_reward_weight": 1.0,
                "contact_diversity_reward_weight": 10.0,
                "contact_continuity_reward_weight": 14.0,
                "stage_gate_progress_reward_weight": 32.0,
                "stage_action_anchor_penalty_weight": 2.0,
                "stage_residual_anchor_penalty_weight": 16.0,
                "distributed_contact_reward_weight": 12.0,
                # This is a dense late-hold reward, not a relaxed acceptance
                # criterion.  The terminal boolean gate remains unchanged.
                "terminal_contact_reward_weight": 40.0,
            }
        )
    elif stage_id == 3:
        profile["stable_lift_reward_weight"] = 8.0
    elif stage_id in (4, 5):
        profile.update(
            {
                "stable_lift_reward_weight": 8.0,
                "lift_height_reward_weight": 14.0,
            }
        )
    elif stage_id == 6:
        profile.update(
            {
                "stable_lift_reward_weight": 8.0,
                "lift_height_reward_weight": 14.0,
                "force_closure_reward_weight": 2.0,
            }
        )
    if overrides:
        unknown = sorted(set(overrides) - set(REWARD_WEIGHT_FIELDS))
        if unknown:
            raise ValueError(f"unknown staged reward overrides: {unknown}")
        profile.update({name: float(value) for name, value in overrides.items()})
    if not all(math.isfinite(value) for value in profile.values()):
        raise ValueError("staged reward weights must be finite")
    if profile["penetration_reward_weight"] >= 0.0:
        raise ValueError("penetration reward weight must be negative")
    nonnegative = set(REWARD_WEIGHT_FIELDS) - {"penetration_reward_weight"}
    if any(profile[name] < 0.0 for name in nonnegative):
        raise ValueError("all non-penetration staged reward weights must be non-negative")
    return profile


def stage_init_noise_std(stage_id: int) -> float:
    """Use restrained exploration for contact stages, then widen for lifting."""

    if not 1 <= int(stage_id) <= 6:
        raise ValueError("stage must be in [1, 6]")
    if stage_id == 1:
        return 0.15
    return 0.20 if stage_id <= 3 else 0.30


def stage_entropy_coef(stage_id: int) -> float:
    """Keep BODex contact refinement conservative before arm lifting begins."""

    if not 1 <= int(stage_id) <= 6:
        raise ValueError("stage must be in [1, 6]")
    return 0.001 if stage_id <= 3 else 0.005


def stage_freeze_policy_noise(stage_id: int) -> bool:
    """Do not let entropy inflate noise while refining BODex contact seeds."""

    if not 1 <= int(stage_id) <= 6:
        raise ValueError("stage must be in [1, 6]")
    return stage_id <= 3


def stage_learning_rate(stage_id: int) -> float:
    """Use smaller PPO steps while preserving a BODex contact basin."""

    if not 1 <= int(stage_id) <= 6:
        raise ValueError("stage must be in [1, 6]")
    if stage_id <= 3:
        return 1.0e-4
    return 3.0e-4
