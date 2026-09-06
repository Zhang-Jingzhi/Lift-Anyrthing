"""Configure a staged environment from the immutable v2 physics manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from migration_4090.xhand_rl_embedded.configuration import configure_env
from migration_4090.xhand_rl_embedded.env import PHASE_HOLD

from .env import XHandStagedEnvCfg
from .reward_profiles import stage_reward_profile
from .stages import stage_with_overrides


def configure_staged_env(
    cfg: XHandStagedEnvCfg,
    *,
    manifest: dict[str, Any],
    object_name: str,
    bodex_bank: Path,
    stage_id: int,
    num_envs: int,
    device: str,
    seed: int,
    record_trajectory: bool,
    allow_test_bank: bool = False,
    candidate_selection: str = "random",
    candidate_repeat_factors: tuple[int, ...] = (),
    fixed_candidate_index: int = -1,
    nominal_pose_lock: bool = False,
    palm_center_alignment_reward_weight: float | None = None,
    reward_overrides: dict[str, float] | None = None,
    terminal_contact_progress_mode: str = "bilateral",
    distributed_reward_requires_current_bilateral_contact: bool = False,
    stage_contact_groups_per_side_min: int | None = None,
    stage_contact_groups_total_min: int | None = None,
    stage_penetration_max_m: float | None = None,
    stage_stable_hold_steps: int | None = None,
    stage_stability_height_tolerance_m: float | None = None,
    stage_penetration_gate_mode: str | None = None,
    stage_stability_gate_mode: str | None = None,
) -> XHandStagedEnvCfg:
    configure_env(
        cfg,
        manifest=manifest,
        object_name=object_name,
        nominal_dataset=bodex_bank,
        nominal_sample_index=0,
        num_envs=num_envs,
        device=device,
        seed=seed,
        record_trajectory=record_trajectory,
    )
    spec = stage_with_overrides(
        stage_id,
        contact_groups_per_side_min=stage_contact_groups_per_side_min,
        contact_groups_total_min=stage_contact_groups_total_min,
        penetration_max_m=stage_penetration_max_m,
        stable_hold_steps=stage_stable_hold_steps,
        stability_height_tolerance_m=stage_stability_height_tolerance_m,
        penetration_gate_mode=stage_penetration_gate_mode,
        stability_gate_mode=stage_stability_gate_mode,
    )
    cfg.stage_id = stage_id
    cfg.allow_test_bank = allow_test_bank
    cfg.candidate_selection = candidate_selection
    cfg.candidate_repeat_factors = candidate_repeat_factors
    cfg.fixed_candidate_index = int(fixed_candidate_index)
    cfg.nominal_pose_lock = bool(nominal_pose_lock)
    if palm_center_alignment_reward_weight is not None:
        cfg.palm_center_alignment_reward_weight = float(
            palm_center_alignment_reward_weight
        )
    if terminal_contact_progress_mode not in ("bilateral", "joint_gate"):
        raise ValueError(
            "terminal_contact_progress_mode must be bilateral or joint_gate"
        )
    cfg.terminal_contact_progress_mode = terminal_contact_progress_mode
    cfg.distributed_reward_requires_current_bilateral_contact = bool(
        distributed_reward_requires_current_bilateral_contact
    )
    cfg.stage_contact_groups_per_side_min = int(
        stage_contact_groups_per_side_min or 0
    )
    cfg.stage_contact_groups_total_min = int(stage_contact_groups_total_min or 0)
    cfg.stage_penetration_max_m = float(stage_penetration_max_m or -1.0)
    cfg.stage_stable_hold_steps = int(stage_stable_hold_steps or 0)
    cfg.stage_stability_height_tolerance_m = (
        float(stage_stability_height_tolerance_m)
        if stage_stability_height_tolerance_m is not None
        else -1.0
    )
    cfg.stage_penetration_gate_mode = stage_penetration_gate_mode or ""
    cfg.stage_stability_gate_mode = stage_stability_gate_mode or ""
    cfg.episode_length_s = spec.episode_length_s
    cfg.approach_fraction, cfg.close_fraction, cfg.lift_fraction, cfg.hold_fraction = (
        spec.phase_fractions
    )
    cfg.disturbance_fraction = 0.0
    cfg.ablation_fraction = 0.0
    cfg.contact_presence_fraction_min = spec.contact_presence_fraction_min
    cfg.physx_penetration_max_m = spec.penetration_max_m
    cfg.reset_xy_noise_m = spec.reset_xy_noise_m
    cfg.reset_yaw_noise_rad = spec.reset_yaw_noise_rad
    cfg.reset_joint_noise_rad = spec.reset_joint_noise_rad
    cfg.residual_integration = spec.residual_integration
    cfg.residual_limit_rad = spec.residual_limit_rad
    # Preserve the genuine BODex approach and closure for continuity training.
    # Stage 2 residuals only refine the already-established grasp during hold;
    # applying them from the first approach step caused candidate-specific
    # drift and destroyed terminal contact on the weaker seeds.
    if stage_id == 2:
        cfg.residual_activation_phase = PHASE_HOLD
    cfg.lift_min_m = max(spec.lift_target_m, 1.0e-4)
    cfg.lift_reward_target_m = max(spec.lift_target_m, 0.01)
    cfg.terminate_on_object_escape = False
    cfg.hard_gate_frontier_reward_weight = 0.0
    cfg.disturbance_direction_reward_weight = 0.0
    for name, value in stage_reward_profile(stage_id, reward_overrides).items():
        setattr(cfg, name, value)
    return cfg
