"""BODex-initialized six-stage Isaac Lab environment with a fixed policy ABI."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from isaaclab.utils import configclass

from migration_4090.xhand_bodex_bimanual.contracts import load_bodex_bank
from migration_4090.xhand_rl_embedded.env import XHandEmbeddedEnv, XHandEmbeddedEnvCfg

from .contact_groups import CONTACT_GROUPS, grouped_filter_indices
from .stages import (
    action_mask_for_group,
    action_mask_for_stage,
    stage_lift_scale,
    stage_with_overrides,
)


@configclass
class XHandStagedEnvCfg(XHandEmbeddedEnvCfg):
    observation_space = 293
    stage_id = 1
    allow_test_bank = False
    candidate_selection = "random"
    candidate_repeat_factors = ()
    # Empty preserves the immutable stage definition.  Conservative retries
    # may explicitly expose a smaller residual action group while keeping the
    # formal gate contract and the fixed 38D policy ABI unchanged.
    active_action_group_override = ""
    # Optional second activation schedule for hand rows.  A negative value
    # keeps the shared residual schedule.  Stable-grasp retries can therefore
    # preserve early wrist refinement while delaying finger residuals until
    # the nominal BODex close trajectory has established contact.
    hand_residual_activation_phase = -1
    hand_residual_activation_close_fraction = 0.0
    # Set to a zero-based BODex candidate index for a fixed-candidate
    # experiment.  Negative values preserve random/round-robin sampling.
    fixed_candidate_index = -1
    penetration_audit_stride = 4
    grouped_contact_force_scale_n = 20.0
    grouped_contact_threshold_n = 0.02
    distributed_contact_reward_weight = 3.0
    stage_stability_reward_weight = 4.0
    stage_lift_hold_reward_weight = 8.0
    stage_gate_progress_reward_weight = 0.0
    stage_action_anchor_penalty_weight = 0.0
    stage_residual_anchor_penalty_weight = 0.0
    terminal_contact_reward_weight = 0.0
    terminal_contact_progress_mode = "bilateral"
    distributed_reward_requires_current_bilateral_contact = False
    stage_stability_requires_current_gate_progress = False
    stage_lift_hold_min_stage = 4
    terminate_on_stage_penetration_failure = False
    # Opt-in Stage-3A curriculum overrides.  Zero/negative/empty values keep
    # the immutable historical stage definition.
    stage_contact_groups_per_side_min = 0
    stage_contact_groups_total_min = 0
    stage_penetration_max_m = -1.0
    stage_stable_hold_steps = 0
    stage_stability_height_tolerance_m = -1.0
    stage_penetration_gate_mode = ""
    stage_stability_gate_mode = ""
    # Lift-bridge training can reject an episode as soon as it violates the
    # path-wise safety contract.  Evaluation keeps this disabled so every
    # fixed-length rollout is still available to the failure-funnel report.
    bridge_terminate_on_unsafe = False


def _quat_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


class XHandStagedEnv(XHandEmbeddedEnv):
    """Keep 38 actions/293 observations fixed while tightening physical gates."""

    cfg: XHandStagedEnvCfg

    def __init__(self, cfg: XHandStagedEnvCfg, *args: Any, **kwargs: Any):
        if cfg.candidate_selection not in ("random", "round_robin"):
            raise ValueError("candidate_selection must be random or round_robin")
        if int(cfg.fixed_candidate_index) < -1:
            raise ValueError("fixed_candidate_index must be -1 or a valid candidate index")
        if int(cfg.hand_residual_activation_phase) < -1:
            raise ValueError("hand_residual_activation_phase must be -1 or a valid phase")
        hand_close_fraction = float(cfg.hand_residual_activation_close_fraction)
        if not 0.0 <= hand_close_fraction <= 1.0:
            raise ValueError(
                "hand_residual_activation_close_fraction must be in [0, 1]"
            )
        self.stage_spec = stage_with_overrides(
            int(cfg.stage_id),
            contact_groups_per_side_min=(
                int(cfg.stage_contact_groups_per_side_min)
                if int(cfg.stage_contact_groups_per_side_min) > 0
                else None
            ),
            contact_groups_total_min=(
                int(cfg.stage_contact_groups_total_min)
                if int(cfg.stage_contact_groups_total_min) > 0
                else None
            ),
            penetration_max_m=(
                float(cfg.stage_penetration_max_m)
                if float(cfg.stage_penetration_max_m) > 0.0
                else None
            ),
            stable_hold_steps=(
                int(cfg.stage_stable_hold_steps)
                if int(cfg.stage_stable_hold_steps) > 0
                else None
            ),
            stability_height_tolerance_m=(
                float(cfg.stage_stability_height_tolerance_m)
                if float(cfg.stage_stability_height_tolerance_m) >= 0.0
                else None
            ),
            penetration_gate_mode=(
                str(cfg.stage_penetration_gate_mode).strip() or None
            ),
            stability_gate_mode=(str(cfg.stage_stability_gate_mode).strip() or None),
        )
        bank = load_bodex_bank(
            Path(cfg.nominal_dataset),
            expected_object=cfg.object_key,
            verify_source=False,
            allow_test_fixture=bool(cfg.allow_test_bank),
            intended_stage=self.stage_spec.stage_id,
        )
        self._bank_payload_before_isaac = bank
        self.bodex_stage_seed_profile = bank.get("stage_seed_profile")
        self.bodex_supported_curriculum_stages = bank.get(
            "supported_curriculum_stages", list(range(1, 7))
        )
        super().__init__(cfg, *args, **kwargs)
        self._load_bank_tensors(bank)
        self._configure_candidate_sampling_schedule()
        if int(self.bodex_pregrasp_bank.shape[0]) != 4:
            raise RuntimeError(
                "the fixed candidate-conditioned policy ABI requires exactly four "
                "diverse BODex candidates"
            )
        if int(cfg.fixed_candidate_index) >= int(self.bodex_pregrasp_bank.shape[0]):
            raise ValueError(
                "fixed_candidate_index is outside the four-candidate BODex bank"
            )
        override = str(self.cfg.active_action_group_override).strip()
        self.effective_active_action_group = (
            override if override else self.stage_spec.active_action_group
        )
        self.action_mask = (
            action_mask_for_group(
                self.joint_names,
                self.effective_active_action_group,
                device=self.device,
            )
            if override
            else action_mask_for_stage(
                self.joint_names, self.stage_spec.stage_id, device=self.device
            )
        ).unsqueeze(0)
        self.hand_action_mask = (
            action_mask_for_group(self.joint_names, "hands", device=self.device)
            * self.action_mask.squeeze(0)
        ).unsqueeze(0)
        body_names = list(self.robot.body_names)
        self.palm_body_ids = {
            side: body_names.index(f"{side}_hand_ee_link") for side in ("left", "right")
        }
        self.group_filter_indices = grouped_filter_indices()
        self.active_bank_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._round_robin_cursor = 0
        self.current_bilateral_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.maximum_bilateral_contact_steps = torch.zeros_like(
            self.current_bilateral_contact_steps
        )
        self.maximum_lift_height = torch.zeros(self.num_envs, device=self.device)
        self.current_stable_lift_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.maximum_stable_lift_steps = torch.zeros_like(self.current_stable_lift_steps)
        self.maximum_contact_group_counts = torch.zeros(
            (self.num_envs, 2), dtype=torch.long, device=self.device
        )
        self._reset_idx(self.robot._ALL_INDICES)

    def _configure_candidate_sampling_schedule(self) -> None:
        bank_count = int(self.bodex_pregrasp_bank.shape[0])
        factors = tuple(int(value) for value in self.cfg.candidate_repeat_factors)
        if not factors:
            factors = (1,) * bank_count
        if len(factors) != bank_count:
            raise ValueError(
                "candidate repeat factor count must match the BODex candidate count: "
                f"{len(factors)} != {bank_count}"
            )
        if any(factor <= 0 for factor in factors):
            raise ValueError("candidate repeat factors must all be positive")
        self.candidate_repeat_factors = factors
        self.candidate_sampling_schedule = torch.tensor(
            [index for index, factor in enumerate(factors) for _ in range(factor)],
            dtype=torch.long,
            device=self.device,
        )

    def _load_bank_tensors(self, payload: dict[str, Any]) -> None:
        pregrasp = []
        grasp = []
        lift = []
        identifiers = []
        for index, sample in enumerate(payload["samples"]):
            source_names = list(sample["joint_names"])
            name_to_index = {name: position for position, name in enumerate(source_names)}
            missing = [name for name in self.joint_names if name not in name_to_index]
            if missing:
                raise RuntimeError(f"BODex sample {index} lacks robot joints: {missing}")

            def aligned(field: str, fallback: str | None = None) -> torch.Tensor:
                values = sample.get(field)
                if values is None and fallback is not None:
                    values = sample.get(fallback)
                if values is None:
                    raise RuntimeError(f"BODex sample {index} lacks {field}")
                return torch.tensor(
                    [values[name_to_index[name]] for name in self.joint_names],
                    dtype=torch.float32,
                    device=self.device,
                )

            pre = aligned(
                "controller_pregrasp_full_body_q", "pregrasp_full_body_q"
            )
            close = aligned("controller_grasp_full_body_q", "full_body_q")
            raw_lift = aligned(
                "controller_lift_full_body_q", "lift_full_body_q"
            )
            lift_scale = stage_lift_scale(self.stage_spec.stage_id)
            pregrasp.append(pre)
            grasp.append(close)
            lift.append(close + lift_scale * (raw_lift - close))
            identifiers.append(str(sample.get("candidate_id", f"candidate_{index:06d}")))
        self.bodex_pregrasp_bank = torch.stack(pregrasp)
        self.bodex_grasp_bank = torch.stack(grasp)
        self.bodex_lift_bank = torch.stack(lift)
        self.bodex_candidate_ids = identifiers

    def _choose_bank_indices(self, count: int) -> torch.Tensor:
        fixed_index = int(self.cfg.fixed_candidate_index)
        if fixed_index >= 0:
            return torch.full(
                (count,), fixed_index, dtype=torch.long, device=self.device
            )
        schedule_count = int(self.candidate_sampling_schedule.numel())
        if self.cfg.candidate_selection == "random":
            positions = torch.randint(schedule_count, (count,), device=self.device)
            return self.candidate_sampling_schedule[positions]
        start = self._round_robin_cursor
        self._round_robin_cursor = (start + count) % schedule_count
        positions = (torch.arange(count, device=self.device) + start) % schedule_count
        return self.candidate_sampling_schedule[positions]

    def _assign_bank_samples(self, env_ids: torch.Tensor) -> None:
        indices = self._choose_bank_indices(len(env_ids))
        self.active_bank_index[env_ids] = indices
        self.nominal_pregrasp[env_ids] = self.bodex_pregrasp_bank[indices]
        self.nominal_grasp[env_ids] = self.bodex_grasp_bank[indices]
        self.nominal_lift[env_ids] = self.bodex_lift_bank[indices]

    def _reset_idx(self, env_ids: Any) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if hasattr(self, "bodex_pregrasp_bank"):
            self._assign_bank_samples(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, "current_bilateral_contact_steps"):
            return
        self.current_bilateral_contact_steps[env_ids] = 0
        self.maximum_bilateral_contact_steps[env_ids] = 0
        self.maximum_lift_height[env_ids] = 0.0
        self.current_stable_lift_steps[env_ids] = 0
        self.maximum_stable_lift_steps[env_ids] = 0
        self.maximum_contact_group_counts[env_ids] = 0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        masked_actions = actions * self.action_mask
        hand_activation_phase = int(self.cfg.hand_residual_activation_phase)
        if hand_activation_phase >= 0 and bool(self.hand_action_mask.any().item()):
            hand_active = self._phase_code() >= hand_activation_phase
            close_activation = float(
                self.cfg.hand_residual_activation_close_fraction
            )
            if close_activation > 0.0:
                close_start, close_end = self.phase_boundaries[:2]
                activation_step = close_start + max(
                    0,
                    int(
                        round(
                            close_activation * max(close_end - close_start, 1)
                        )
                    ),
                )
                hand_active &= self.episode_length_buf >= activation_step
            hand_scale = hand_active.float().unsqueeze(-1)
            masked_actions = masked_actions * (
                1.0 - self.hand_action_mask + self.hand_action_mask * hand_scale
            )
        super()._pre_physics_step(masked_actions)

    def _group_contact_forces(self) -> torch.Tensor:
        matrix = self.object_contacts.data.force_matrix_w
        if matrix is None or matrix.ndim != 4:
            raise RuntimeError("object-filtered contact force matrix is unavailable")
        magnitudes = torch.linalg.vector_norm(matrix[:, 0], dim=-1)
        features = []
        for side in ("left", "right"):
            for group in CONTACT_GROUPS:
                indices = self.group_filter_indices[(side, group)]
                features.append(magnitudes[:, indices].sum(dim=-1))
        return torch.stack(features, dim=-1).reshape(self.num_envs, 2, len(CONTACT_GROUPS))

    def _palm_pose_in_object_frame(self) -> torch.Tensor:
        object_rotation = _quat_to_matrix_wxyz(self.object.data.root_quat_w)
        object_rotation_inverse = object_rotation.transpose(-1, -2)
        values = []
        for side in ("left", "right"):
            body_id = self.palm_body_ids[side]
            relative_position_world = (
                self.robot.data.body_pos_w[:, body_id] - self.object.data.root_pos_w
            )
            relative_position = torch.einsum(
                "bij,bj->bi", object_rotation_inverse, relative_position_world
            )
            palm_rotation = _quat_to_matrix_wxyz(self.robot.data.body_quat_w[:, body_id])
            relative_rotation = object_rotation_inverse @ palm_rotation
            rotation_6d = relative_rotation[..., :, :2].reshape(self.num_envs, 6)
            values.extend((relative_position, rotation_6d))
        return torch.cat(values, dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        base = super()._get_observations()["policy"]
        q = self.robot.data.joint_pos
        limits = self.robot.data.soft_joint_pos_limits
        q_half = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-4)
        target_error = (q - self._base_target(self._phase_code())) / q_half
        group_forces = self._group_contact_forces()
        group_flags = (
            group_forces > float(self.cfg.grouped_contact_threshold_n)
        ).float()
        group_features = torch.stack(
            (
                group_forces / float(self.cfg.grouped_contact_force_scale_n),
                group_flags,
            ),
            dim=-1,
        ).reshape(self.num_envs, -1)
        stage_one_hot = torch.nn.functional.one_hot(
            torch.full(
                (self.num_envs,),
                self.stage_spec.stage_id - 1,
                dtype=torch.long,
                device=self.device,
            ),
            num_classes=6,
        ).float()
        candidate_one_hot = torch.nn.functional.one_hot(
            self.active_bank_index,
            num_classes=4,
        ).float()
        bilateral_steps = (
            self.maximum_bilateral_contact_steps
            if self.stage_spec.stage_id == 1
            else self.current_bilateral_contact_steps
        )
        bilateral_progress = torch.clamp(
            bilateral_steps.float()
            / max(self.stage_spec.required_bilateral_steps, 1),
            0.0,
            1.0,
        ).unsqueeze(-1)
        maximum_group_progress = torch.clamp(
            self.maximum_contact_group_counts.float()
            / max(self.stage_spec.contact_groups_per_side_min, 1),
            0.0,
            1.0,
        )
        stable_hold_progress = torch.clamp(
            self.current_stable_lift_steps.float()
            / max(self.stage_spec.stable_hold_steps, 1),
            0.0,
            1.0,
        ).unsqueeze(-1)
        gate_memory = torch.cat(
            (bilateral_progress, maximum_group_progress, stable_hold_progress),
            dim=-1,
        )
        candidate_gated_memory = (
            candidate_one_hot.unsqueeze(-1) * gate_memory.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        observation = torch.cat(
            (
                base,
                self.integrated_residual / max(float(self.cfg.residual_limit_rad), 1.0e-8),
                target_error,
                self._palm_pose_in_object_frame(),
                group_features,
                stage_one_hot,
                gate_memory,
                candidate_gated_memory,
                candidate_one_hot,
            ),
            dim=-1,
        )
        if observation.shape[-1] != int(self.cfg.observation_space):
            raise RuntimeError(
                f"staged observation ABI changed: {observation.shape[-1]} != {self.cfg.observation_space}"
            )
        return {"policy": observation}

    def _audit_post_physics_step(self) -> torch.Tensor:
        # Sample PhysX contacts before the shared metric update.  Bridge
        # safety and reward code consume the historical penetration bit during
        # that update; auditing afterwards introduced a one-step lag in which
        # a newly unsafe contact could still receive lift credit.
        code = self._executed_phase_code()
        stride = max(int(self.cfg.penetration_audit_stride), 1)
        if bool(torch.any(self.episode_length_buf % stride == 0).item()):
            env_ids = torch.arange(self.num_envs, device=self.device)
            penetration, _ = self._raw_contacts(env_ids)
            measured = torch.isfinite(penetration)
            self.penetration_measured |= measured
            self.current_penetration = torch.where(
                measured, penetration, self.current_penetration
            )
            self.maximum_penetration = torch.maximum(
                self.maximum_penetration,
                torch.where(measured, penetration, self.maximum_penetration),
            )
        super()._audit_post_physics_step()
        return code

    def _update_metrics(self, code: torch.Tensor) -> None:
        super()._update_metrics(code)
        bilateral = (self._sensor_force("left") > 0.02) & (
            self._sensor_force("right") > 0.02
        )
        self.current_bilateral_contact_steps = torch.where(
            bilateral,
            self.current_bilateral_contact_steps + 1,
            torch.zeros_like(self.current_bilateral_contact_steps),
        )
        self.maximum_bilateral_contact_steps = torch.maximum(
            self.maximum_bilateral_contact_steps, self.current_bilateral_contact_steps
        )
        height = self.object.data.root_pos_w[:, 2] - (
            self.initial_object_pos[:, 2] + self.scene.env_origins[:, 2]
        )
        self.maximum_lift_height = torch.maximum(self.maximum_lift_height, height)
        linear_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object.data.root_ang_vel_w, dim=-1)
        stable_lift = (
            bilateral
            & (
                height
                >= self.stage_spec.lift_target_m
                - self.stage_spec.stability_height_tolerance_m
            )
            & (linear_speed <= self.stage_spec.linear_speed_max_m_s)
            & (angular_speed <= self.stage_spec.angular_speed_max_rad_s)
        )
        self.current_stable_lift_steps = torch.where(
            stable_lift,
            self.current_stable_lift_steps + 1,
            torch.zeros_like(self.current_stable_lift_steps),
        )
        self.maximum_stable_lift_steps = torch.maximum(
            self.maximum_stable_lift_steps, self.current_stable_lift_steps
        )
        group_counts = (
            self._group_contact_forces()
            > float(self.cfg.grouped_contact_threshold_n)
        ).sum(dim=-1)
        self.maximum_contact_group_counts = torch.maximum(
            self.maximum_contact_group_counts, group_counts
        )

    def _penetration_gate_value(self) -> torch.Tensor:
        if self.stage_spec.penetration_gate_mode == "current":
            return self.current_penetration
        return self.maximum_penetration

    def _stage_gate_components(self) -> dict[str, torch.Tensor]:
        continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
        group_flags = self._group_contact_forces() > float(
            self.cfg.grouped_contact_threshold_n
        )
        group_counts = group_flags.sum(dim=-1)
        distributed_group_counts = (
            self.maximum_contact_group_counts
            if self.stage_spec.stage_id == 2
            else group_counts
        )
        finite = torch.isfinite(self.robot.data.joint_pos).all(dim=-1) & torch.isfinite(
            self.object.data.root_state_w
        ).all(dim=-1)
        linear_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object.data.root_ang_vel_w, dim=-1)
        penetration_value = self._penetration_gate_value()
        penetration = self.penetration_measured & (
            penetration_value <= self.stage_spec.penetration_max_m
        )
        bilateral_steps = (
            self.maximum_bilateral_contact_steps
            if self.stage_spec.stage_id == 1
            else self.current_bilateral_contact_steps
        )
        if self.stage_spec.stability_gate_mode == "historical_max":
            stable_object = (
                self.maximum_stable_lift_steps >= self.stage_spec.stable_hold_steps
            )
        else:
            stable_object = (
                (linear_speed <= self.stage_spec.linear_speed_max_m_s)
                & (angular_speed <= self.stage_spec.angular_speed_max_rad_s)
                & (self.current_stable_lift_steps >= self.stage_spec.stable_hold_steps)
            )
        return {
            "finite_state": finite,
            "bilateral_contact": bilateral_steps
            >= self.stage_spec.required_bilateral_steps,
            "contact_continuity": (
                (continuity_left >= self.stage_spec.contact_presence_fraction_min)
                & (continuity_right >= self.stage_spec.contact_presence_fraction_min)
            ),
            "distributed_contacts": (
                (
                    distributed_group_counts[:, 0]
                    >= self.stage_spec.contact_groups_per_side_min
                )
                & (
                    distributed_group_counts[:, 1]
                    >= self.stage_spec.contact_groups_per_side_min
                )
                & (
                    (distributed_group_counts.sum(dim=-1)
                     >= self.stage_spec.contact_groups_total_min)
                    if self.stage_spec.contact_groups_total_min > 0
                    else torch.ones(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                )
            ),
            "stable_object": stable_object,
            "lift_height": self.maximum_lift_height >= self.stage_spec.lift_target_m,
            "stable_lift_hold": self.current_stable_lift_steps
            >= self.stage_spec.stable_hold_steps,
            "catastrophic_penetration": penetration,
            "strict_penetration": penetration,
            "no_teleport": torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def _physics_gate_mask(self) -> torch.Tensor:
        components = self._stage_gate_components()
        passed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for name in self.stage_spec.required_gates:
            passed &= components[name]
        return passed

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        current_group_counts = (
            self._group_contact_forces() > float(self.cfg.grouped_contact_threshold_n)
        ).sum(dim=-1)
        group_counts = (
            self.maximum_contact_group_counts
            if self.stage_spec.stage_id == 2
            else current_group_counts
        )
        distributed = torch.clamp(
            torch.minimum(group_counts[:, 0], group_counts[:, 1]).float()
            / max(self.stage_spec.contact_groups_per_side_min, 1),
            0.0,
            1.0,
        )
        linear_speed = torch.linalg.vector_norm(self.object.data.root_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.object.data.root_ang_vel_w, dim=-1)
        stability = torch.exp(-10.0 * linear_speed - 2.0 * angular_speed)
        hold_progress = torch.clamp(
            self.current_stable_lift_steps.float()
            / max(self.stage_spec.stable_hold_steps, 1),
            0.0,
            1.0,
        )
        bilateral_steps = (
            self.maximum_bilateral_contact_steps
            if self.stage_spec.stage_id == 1
            else self.current_bilateral_contact_steps
        )
        bilateral_progress = torch.clamp(
            bilateral_steps.float()
            / max(self.stage_spec.required_bilateral_steps, 1),
            0.0,
            1.0,
        )
        _, _, hold_start, hold_end, _, _ = self.phase_boundaries
        hold_time_progress = torch.clamp(
            (self.episode_length_buf.float() - float(hold_start))
            / max(float(hold_end - hold_start), 1.0),
            0.0,
            1.0,
        )
        bilateral_now = (self._sensor_force("left") > 0.02) & (
            self._sensor_force("right") > 0.02
        )
        penetration_gate = self.penetration_measured & (
            self._penetration_gate_value() <= self.stage_spec.penetration_max_m
        )
        penetration_is_required = any(
            name in self.stage_spec.required_gates
            for name in ("catastrophic_penetration", "strict_penetration")
        )
        frontier_terms = [bilateral_progress]
        frontier = bilateral_progress
        if self.stage_spec.stage_id >= 2:
            continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
            continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
            continuity_progress = torch.clamp(
                torch.minimum(continuity_left, continuity_right)
                / max(self.stage_spec.contact_presence_fraction_min, 1.0e-8),
                0.0,
                1.0,
            )
            frontier = frontier * continuity_progress
            frontier_terms.append(frontier)
            frontier = frontier * distributed
            frontier_terms.append(frontier)
        if penetration_is_required:
            frontier = frontier * penetration_gate.float()
            frontier_terms.append(frontier)
        required_contact_gate_progress = torch.stack(frontier_terms).mean(dim=0)
        contact_gate_conjunction_progress = frontier
        # Reward the exact late-hold conjunction progressively.  The previous
        # term paid only for bilateral contact, so a policy could ignore the
        # continuity/distribution gates until the one-step terminal reward.
        # Multiplying by ``bilateral_now`` also prevents historical maximum
        # group counts from paying after one hand has already fallen away.
        bilateral_terminal_progress = (
            bilateral_now.float()
            * bilateral_progress
            * torch.square(hold_time_progress)
        )
        joint_gate_terminal_progress = (
            bilateral_now.float()
            * required_contact_gate_progress
            * torch.square(hold_time_progress)
        )
        if self.cfg.terminal_contact_progress_mode == "bilateral":
            terminal_contact_progress = bilateral_terminal_progress
        elif self.cfg.terminal_contact_progress_mode == "joint_gate":
            terminal_contact_progress = joint_gate_terminal_progress
        else:
            raise ValueError(
                "terminal_contact_progress_mode must be bilateral or joint_gate"
            )
        maintained_distributed = (
            distributed * bilateral_now.float()
            if bool(self.cfg.distributed_reward_requires_current_bilateral_contact)
            else distributed
        )
        action_anchor = torch.mean(torch.square(self.actions), dim=-1)
        residual_anchor = torch.mean(
            torch.square(
                self.integrated_residual
                / max(float(self.cfg.residual_limit_rad), 1.0e-8)
            ),
            dim=-1,
        )
        if bool(self.cfg.stage_stability_requires_current_gate_progress):
            stability_reward_score = stability * contact_gate_conjunction_progress
            hold_reward_score = hold_progress * contact_gate_conjunction_progress
        else:
            stability_reward_score = stability
            hold_reward_score = hold_progress
        bonus = (
            float(self.cfg.distributed_contact_reward_weight)
            * maintained_distributed
            * float(self.stage_spec.stage_id >= 2)
            + float(self.cfg.stage_stability_reward_weight)
            * stability_reward_score
            * float(self.stage_spec.stage_id >= 3)
            + float(self.cfg.stage_lift_hold_reward_weight)
            * hold_reward_score
            * float(self.stage_spec.stage_id >= int(self.cfg.stage_lift_hold_min_stage))
            + float(self.cfg.stage_gate_progress_reward_weight)
            * required_contact_gate_progress
            + float(self.cfg.terminal_contact_reward_weight)
            * terminal_contact_progress
            - float(self.cfg.stage_action_anchor_penalty_weight) * action_anchor
            - float(self.cfg.stage_residual_anchor_penalty_weight) * residual_anchor
        )
        scaled_bonus = float(self.cfg.training_reward_scale) * bonus
        self.extras.setdefault("log", {})[
            "staged/distributed_contact_bonus"
        ] = maintained_distributed.mean()
        self.extras["log"]["staged/stability_bonus"] = stability.mean()
        self.extras["log"]["staged/lift_hold_progress"] = hold_progress.mean()
        self.extras["log"]["staged/stability_reward_score"] = (
            stability_reward_score.mean()
        )
        self.extras["log"]["staged/lift_hold_reward_score"] = (
            hold_reward_score.mean()
        )
        self.extras["log"][
            "staged/contact_gate_conjunction_progress"
        ] = contact_gate_conjunction_progress.mean()
        self.extras["log"][
            "staged/required_contact_gate_progress"
        ] = required_contact_gate_progress.mean()
        self.extras["log"]["staged/action_anchor_penalty"] = action_anchor.mean()
        self.extras["log"]["staged/residual_anchor_penalty"] = residual_anchor.mean()
        self.extras["log"][
            "staged/terminal_contact_progress"
        ] = terminal_contact_progress.mean()
        self.extras["log"]["staged/gate_success_fraction"] = self._physics_gate_mask().float().mean()
        self.extras["log"]["reward/staged_bonus_unscaled"] = bonus.mean()
        self.extras["log"]["reward/staged_bonus_scaled"] = scaled_bonus.mean()
        total_reward = reward + scaled_bonus
        self.extras["log"]["reward/total_with_staged_scaled"] = total_reward.mean()
        self.extras["log"]["staged/penetration_failure_fraction"] = (
            self.penetration_measured
            & (self.maximum_penetration > self.stage_spec.penetration_max_m)
        ).float().mean()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if bool(self.cfg.terminate_on_stage_penetration_failure):
            penetration_failure = self.penetration_measured & (
                self._penetration_gate_value() > self.stage_spec.penetration_max_m
            )
            terminated |= penetration_failure
            time_out &= ~penetration_failure
        return terminated, time_out

    def _metrics_report(self, env_id: int) -> dict[str, Any]:
        components = self._stage_gate_components()
        gates = {
            name: bool(components[name][env_id].item())
            for name in self.stage_spec.required_gates
        }
        continuity = {
            "left": float(
                (self.left_contact_steps[env_id] / self.contact_steps[env_id].clamp_min(1.0)).item()
            ),
            "right": float(
                (self.right_contact_steps[env_id] / self.contact_steps[env_id].clamp_min(1.0)).item()
            ),
        }
        bank_index = int(self.active_bank_index[env_id].item())
        current_group_counts = (
            self._group_contact_forces()
            > float(self.cfg.grouped_contact_threshold_n)
        ).sum(dim=-1)
        report = {
            "schema": "xhand_rl_staged_episode_metrics_v1",
            "validator": "isaac_lab_staged_bodex_rl_v1",
            "stage": self.stage_spec.to_dict(),
            "stage_gates": gates,
            "stage_pass": all(gates.values()),
            "candidate_index": bank_index,
            "candidate_id": self.bodex_candidate_ids[bank_index],
            "bodex_stage_seed_profile": self.bodex_stage_seed_profile,
            "bodex_supported_curriculum_stages": self.bodex_supported_curriculum_stages,
            "contact_presence_fraction": continuity,
            "bilateral_contact_consecutive_steps": int(
                self.current_bilateral_contact_steps[env_id].item()
            ),
            "maximum_bilateral_contact_consecutive_steps": int(
                self.maximum_bilateral_contact_steps[env_id].item()
            ),
            "maximum_lift_height_m": float(self.maximum_lift_height[env_id].item()),
            "stable_lift_consecutive_steps": int(
                self.current_stable_lift_steps[env_id].item()
            ),
            "maximum_stable_lift_consecutive_steps": int(
                self.maximum_stable_lift_steps[env_id].item()
            ),
            "maximum_contact_group_counts_by_side": {
                "left": int(self.maximum_contact_group_counts[env_id, 0].item()),
                "right": int(self.maximum_contact_group_counts[env_id, 1].item()),
            },
            "terminal_contact_group_counts_by_side": {
                "left": int(current_group_counts[env_id, 0].item()),
                "right": int(current_group_counts[env_id, 1].item()),
            },
            "current_physx_contact_penetration_m": float(
                self.current_penetration[env_id].item()
            ),
            "maximum_physx_contact_penetration_m": float(
                self.maximum_penetration[env_id].item()
            ),
            "physx_penetration_measured": bool(self.penetration_measured[env_id].item()),
            "object_linear_speed_m_s": float(
                torch.linalg.vector_norm(self.object.data.root_lin_vel_w[env_id]).item()
            ),
            "object_angular_speed_rad_s": float(
                torch.linalg.vector_norm(self.object.data.root_ang_vel_w[env_id]).item()
            ),
            "robot_root_teleported_during_grasp_or_lift": False,
            "object_teleported_during_grasp_or_lift": False,
            "evaluated_policy_steps": int(self.episode_length_buf[env_id].item()),
            "final_audited_policy_step": int(self.last_audited_episode_step[env_id].item()),
            "formal_stage6_complete": self.stage_spec.stage_id == 6 and all(gates.values()),
        }
        if float(self.cfg.palm_center_alignment_reward_weight) != 0.0:
            alignment, left_cosine, right_cosine = self._palm_center_alignment()
            report["palm_center_alignment_score"] = float(alignment[env_id].item())
            report["left_palm_center_cosine"] = float(left_cosine[env_id].item())
            report["right_palm_center_cosine"] = float(right_cosine[env_id].item())
        return report
