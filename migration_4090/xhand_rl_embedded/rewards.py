"""Dense Torch rewards for v2; hard acceptance remains separate."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Formal force closure is useful only when the object is supported by the two
# hands.  A non-zero pre-lift floor was audited in v7 and still let PPO profit
# from table-supported closure, especially for the pyramid and cube.  Keep the
# constant explicit for provenance/tests, but make the strict-aligned v8 floor
# zero so the force-closure term cannot pay before continuous bimanual lift.
FORCE_CLOSURE_LIFT_SUPPORT_FLOOR = 0.0


def strict_disturbance_prefix_fraction(
    translation_displacements_m: torch.Tensor,
    rotation_displacements_rad: torch.Tensor,
    *,
    translation_limit_m: float,
    rotation_limit_rad: float,
) -> torch.Tensor:
    """Return credit for the longest consecutively passed disturbance prefix.

    This is PPO shaping only.  The strict disturbance gate still requires all
    six translations and all six rotations to pass.  Returning zero after all
    twelve pass avoids double-counting the next integer hard-gate frontier
    level; an 11/12 prefix therefore transitions smoothly into that level.
    """

    translation_pass = torch.isfinite(translation_displacements_m) & (
        translation_displacements_m <= translation_limit_m
    )
    rotation_pass = torch.isfinite(rotation_displacements_rad) & (
        rotation_displacements_rad <= rotation_limit_rad
    )
    ordered = torch.cat((translation_pass, rotation_pass), dim=-1)
    consecutive = torch.cumprod(ordered.to(dtype=torch.int64), dim=-1)
    count = consecutive.sum(dim=-1)
    fraction = count.to(dtype=translation_displacements_m.dtype) / float(
        ordered.shape[-1]
    )
    return torch.where(count == ordered.shape[-1], torch.zeros_like(fraction), fraction)


def strict_disturbance_pass_fraction(
    translation_displacements_m: torch.Tensor,
    rotation_displacements_rad: torch.Tensor,
    *,
    translation_limit_m: float,
    rotation_limit_rad: float,
) -> torch.Tensor:
    """Return the fraction of measured disturbance directions that pass.

    This is PPO shaping only.  ``inf`` denotes a direction that has not yet
    completed in the current trajectory, so unmeasured directions never count
    as passes.  The formal gate remains unchanged and still requires all six
    translations and all six rotations to be finite and within threshold.
    Unlike :func:`strict_disturbance_prefix_fraction`, this statistic is not
    ordered; a successful rotation direction therefore provides a learning
    signal even while an earlier translation direction is still a near miss.
    """

    translation_pass = torch.isfinite(translation_displacements_m) & (
        translation_displacements_m <= translation_limit_m
    )
    rotation_pass = torch.isfinite(rotation_displacements_rad) & (
        rotation_displacements_rad <= rotation_limit_rad
    )
    passed = torch.cat((translation_pass, rotation_pass), dim=-1)
    return passed.float().sum(dim=-1) / float(passed.shape[-1])


@dataclass(frozen=True)
class EmbeddedRewardWeights:
    bilateral_contact: float = 6.0
    contact_balance: float = 1.0
    contact_diversity: float = 1.5
    contact_continuity: float = 4.0
    lift_height: float = 12.0
    # Optional run-level shaping.  Zero preserves every historical checkpoint
    # and reward trace; targeted retries enable it to distinguish a maintained
    # bimanual lift from a transient upward impulse.
    stable_lift: float = 0.0
    hold_stability: float = 4.0
    gravity_margin: float = 4.0
    disturbance_recovery: float = 4.0
    # Exploration shaping is deliberately softer than the hard 0.5 mm gate.
    # The gate still rejects every trajectory whose historical maximum exceeds
    # the limit; this coefficient only prevents early contact acquisition from
    # becoming an unrecoverable negative-reward trap.
    penetration: float = -8.0
    penetration_clear: float = 8.0
    force_closure: float = 4.0
    ablation_contract: float = 18.0
    action_rate: float = -0.02
    joint_velocity: float = -0.002
    one_hand_only: float = -6.0
    dropped: float = -12.0
    terminal_success: float = 30.0
    # Exact, phase-safe hard-gate prefixes supplied by the environment.  This
    # is disabled for historical runs and explicitly enabled by v8 retries.
    hard_gate_frontier: float = 0.0
    # Non-sequential disturbance shaping for new retries. This does not alter
    # the strict gate; it only gives credit for directions already measured
    # within tolerance when an earlier direction is still failing.
    disturbance_direction: float = 0.0
    nominal_tracking: float = 2.0
    # Reachability shaping for nominal poses that are outside the strict
    # contact basin.  This is dense exploration credit only; it does not
    # relax any terminal physics gate.
    approach_proximity: float = 8.0
    # Soft geometry shaping used by the fixed-pose palm-alignment ablation.
    # The strict contact/lift contract is unchanged and the default is zero.
    palm_center_alignment: float = 0.0


def compute_embedded_reward(
    *,
    left_contact_force: torch.Tensor,
    right_contact_force: torch.Tensor,
    left_contact_count: torch.Tensor,
    right_contact_count: torch.Tensor,
    left_contact_presence_fraction: torch.Tensor,
    right_contact_presence_fraction: torch.Tensor,
    object_height_delta: torch.Tensor,
    object_linear_velocity: torch.Tensor,
    object_angular_velocity: torch.Tensor,
    hold_max_drift_m: torch.Tensor,
    challenge_translation_m: torch.Tensor,
    challenge_rotation_rad: torch.Tensor,
    current_penetration_m: torch.Tensor,
    penetration_measured: torch.Tensor,
    penetration_limit_m: float,
    force_closure_pass: torch.Tensor,
    force_closure_quality: torch.Tensor,
    ablation_phase: torch.Tensor,
    ablation_stable_phase: torch.Tensor,
    ablation_active_contact_fraction: torch.Tensor,
    ablation_inactive_contact_fraction: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    joint_velocity: torch.Tensor,
    lift_or_later: torch.Tensor,
    hold_or_challenge: torch.Tensor,
    disturbance_phase: torch.Tensor,
    dropped: torch.Tensor,
    terminal_success: torch.Tensor,
    hand_object_distance_m: torch.Tensor | None = None,
    approach_or_close: torch.Tensor | None = None,
    proximity_scale_m: float = 0.18,
    nominal_tracking_score: torch.Tensor | None = None,
    nominal_tracking_phase: torch.Tensor | None = None,
    palm_center_alignment_score: torch.Tensor | None = None,
    lift_reward_target_m: float,
    lift_min_m: float,
    gravity_drift_limit_m: float,
    contact_presence_fraction_min: float,
    inactive_hand_contact_fraction_max: float,
    maximum_penetration_m: torch.Tensor | None = None,
    hard_gate_frontier_score: torch.Tensor | None = None,
    disturbance_direction_score: torch.Tensor | None = None,
    force_closure_ready: torch.Tensor | None = None,
    weights: EmbeddedRewardWeights = EmbeddedRewardWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Keep the reward helper backward-compatible with older diagnostics and
    # unit tests that predate nominal-trajectory tracking.  The environment
    # supplies both tensors explicitly; standalone callers get a neutral
    # tracking contribution instead of a missing-keyword failure.
    if nominal_tracking_score is None:
        nominal_tracking_score = torch.zeros_like(left_contact_force)
    if nominal_tracking_phase is None:
        nominal_tracking_phase = torch.zeros_like(left_contact_force, dtype=torch.bool)
    if palm_center_alignment_score is None:
        palm_center_alignment_score = torch.zeros_like(left_contact_force)
    if hand_object_distance_m is None:
        hand_object_distance_m = torch.zeros(
            (*left_contact_force.shape, 2),
            dtype=left_contact_force.dtype,
            device=left_contact_force.device,
        )
    if approach_or_close is None:
        approach_or_close = torch.zeros_like(left_contact_force, dtype=torch.bool)
    if maximum_penetration_m is None:
        maximum_penetration_m = current_penetration_m
    if hard_gate_frontier_score is None:
        hard_gate_frontier_score = torch.zeros_like(left_contact_force)
    if disturbance_direction_score is None:
        disturbance_direction_score = torch.zeros_like(left_contact_force)
    left = torch.clamp(left_contact_force, min=0.0)
    right = torch.clamp(right_contact_force, min=0.0)
    bilateral = torch.tanh(torch.minimum(left, right) / 5.0)
    balance = torch.exp(-torch.abs(left - right) / (left + right + 1.0)) * bilateral
    diversity = torch.clamp(torch.minimum(left_contact_count, right_contact_count) / 4.0, 0.0, 1.0)
    only_one = torch.logical_xor(left > 0.25, right > 0.25).float()
    height = torch.clamp(object_height_delta / lift_reward_target_m, 0.0, 1.0)
    lift_gate_progress = torch.clamp(object_height_delta / max(lift_min_m, 1.0e-8), 0.0, 1.0)
    continuity = torch.clamp(
        torch.minimum(left_contact_presence_fraction, right_contact_presence_fraction)
        / max(contact_presence_fraction_min, 1.0e-8),
        0.0,
        1.0,
    )
    stable_linear = torch.exp(-10.0 * torch.linalg.vector_norm(object_linear_velocity, dim=-1))
    stable_angular = torch.exp(-2.0 * torch.linalg.vector_norm(object_angular_velocity, dim=-1))
    gravity_margin = torch.exp(
        -hold_max_drift_m / max(gravity_drift_limit_m, 1.0e-8)
    )
    recovery = torch.exp(-40.0 * challenge_translation_m - 5.0 * challenge_rotation_rad)
    current_penetration_ratio = torch.clamp(
        current_penetration_m / max(penetration_limit_m, 1.0e-8), min=0.0
    )
    maximum_penetration_ratio = torch.clamp(
        maximum_penetration_m / max(penetration_limit_m, 1.0e-8), min=0.0
    )
    # The hard contract rejects the maximum penetration over the complete
    # episode.  Penalize that same historical maximum so moving out of a deep
    # overlap later cannot erase the learning signal.  Preserve ordering far
    # outside the limit rather than saturating all large violations together.
    penetration_excess = torch.log1p(
        torch.clamp(maximum_penetration_ratio - 1.0, min=0.0)
    )
    # Clearance is useful only while both hands actually load the object.
    # Otherwise a policy can maximize this term by avoiding all contact.
    penetration_clear = (
        penetration_measured.float()
        * bilateral
        / (1.0 + current_penetration_ratio)
    )
    # Keep a learning signal across the full [0, 1] contact-fraction range.
    # A separate gate bonus still aligns the optimum with the hard threshold.
    inactive_release = torch.clamp(1.0 - ablation_inactive_contact_fraction, 0.0, 1.0)
    inactive_gate = (
        ablation_inactive_contact_fraction <= inactive_hand_contact_fraction_max
    ).float()
    active_single_hand_viability = lift_gate_progress * torch.clamp(
        ablation_active_contact_fraction / max(contact_presence_fraction_min, 1.0e-8),
        0.0,
        1.0,
    )
    # A no-grasp episode trivially releases both hands and makes either hand
    # unable to hold the object. Credit the counterfactual ablation only after
    # the preceding bimanual phases established continuous contact and lift.
    bimanual_prerequisite = continuity * lift_gate_progress
    ablation_contract = bimanual_prerequisite * (1.0 - active_single_hand_viability) * (
        0.75 * inactive_release + 0.25 * inactive_gate
    )
    force_closure_score = torch.clamp(force_closure_quality, 0.0, 1.0)
    # Historical callers only supplied a soft bimanual/lift prerequisite.  New
    # formal retries can additionally pass the same-trajectory strict
    # clearance+gravity prefix.  This prevents a near-miss force-closure
    # quality score from paying while the object is still table-supported or
    # has not passed the historical PhysX penetration audit.  The boolean
    # terminal gate remains unchanged; this only aligns the dense signal with
    # the first missing physical prerequisite.
    if force_closure_ready is None:
        force_closure_support = bimanual_prerequisite
    else:
        force_closure_support = bimanual_prerequisite * force_closure_ready.float()
    force_closure_lift_support = FORCE_CLOSURE_LIFT_SUPPORT_FLOOR + (
        1.0 - FORCE_CLOSURE_LIFT_SUPPORT_FLOOR
    ) * force_closure_support
    action_rate = torch.mean(torch.square(actions - previous_actions), dim=-1)
    joint_speed = torch.mean(torch.square(joint_velocity), dim=-1)
    # Ablation is a counterfactual test: the inactive hand must release and
    # the remaining single hand must fail, so ordinary bimanual terms stop.
    bimanual_phase = ~ablation_phase
    # During exploration, a no-contact policy should not be rewarded for
    # fleeing the object simply because an early one-hand collision was deep.
    # Keep a small barrier without letting it dominate contact acquisition;
    # once both hands load the object, the full penetration penalty applies.
    penetration_contact_gate = 0.25 + 0.75 * bilateral
    proximity = torch.exp(
        -torch.clamp(hand_object_distance_m, min=0.0)
        / max(proximity_scale_m, 1.0e-8)
    ).mean(dim=-1)
    terms = {
        "bilateral_contact": weights.bilateral_contact * bilateral * bimanual_phase.float(),
        "contact_balance": weights.contact_balance * balance * bimanual_phase.float(),
        "contact_diversity": weights.contact_diversity * diversity * bimanual_phase.float(),
        "contact_continuity": weights.contact_continuity
        * continuity
        * lift_or_later.float()
        * bimanual_phase.float(),
        "lift_height": weights.lift_height
        * height
        * continuity
        * lift_or_later.float()
        * bimanual_phase.float()
        * (0.25 + 0.75 * bilateral),
        "stable_lift": weights.stable_lift
        * height
        * continuity
        * stable_linear
        * stable_angular
        * bilateral
        * lift_or_later.float()
        * bimanual_phase.float(),
        "hold_stability": weights.hold_stability
        * stable_linear
        * stable_angular
        * hold_or_challenge.float()
        * height
        * bilateral
        * bimanual_phase.float(),
        "gravity_margin": weights.gravity_margin
        * gravity_margin
        * hold_or_challenge.float()
        * height
        * bilateral
        * bimanual_phase.float(),
        "disturbance_recovery": weights.disturbance_recovery
        * recovery
        * disturbance_phase.float()
        * height
        * bilateral,
        "penetration": weights.penetration * penetration_excess * penetration_contact_gate,
        "penetration_clear": weights.penetration_clear * penetration_clear,
        "force_closure": weights.force_closure
        * (0.75 * force_closure_score + 0.25 * force_closure_pass.float())
        * force_closure_lift_support
        * hold_or_challenge.float()
        * bimanual_phase.float(),
        "ablation_contract": weights.ablation_contract
        * ablation_contract
        * ablation_stable_phase.float(),
        "action_rate": weights.action_rate * action_rate,
        "joint_velocity": weights.joint_velocity * joint_speed,
        "one_hand_only": weights.one_hand_only * only_one * bimanual_phase.float(),
        "dropped": weights.dropped * dropped.float() * bimanual_phase.float(),
        "terminal_success": weights.terminal_success * terminal_success.float(),
        "hard_gate_frontier": weights.hard_gate_frontier
        * torch.clamp(hard_gate_frontier_score, min=0.0),
        "disturbance_direction": weights.disturbance_direction
        * torch.clamp(disturbance_direction_score, min=0.0),
        "nominal_tracking": weights.nominal_tracking
        * nominal_tracking_score
        * nominal_tracking_phase.float(),
        "approach_proximity": weights.approach_proximity
        * proximity
        * approach_or_close.float(),
        "palm_center_alignment": weights.palm_center_alignment
        * torch.clamp(palm_center_alignment_score, 0.0, 1.0)
        * approach_or_close.float(),
    }
    return torch.stack(tuple(terms.values()), dim=0).sum(dim=0), terms
