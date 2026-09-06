"""Stage-2.5 environment for controlled, stable paired-palm micro-lifts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from migration_4090.xhand_bodex_bimanual.contracts import sha256_file
from migration_4090.xhand_rl_embedded.env import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_HOLD,
    PHASE_LIFT,
)

from .bridge_profiles import LiftBridgeProfile
from .env import XHandStagedEnv


class XHandLiftBridgeEnv(XHandStagedEnv):
    """Keep the Stage-2 policy ABI while training a bounded lift trajectory."""

    def __init__(
        self,
        cfg: Any,
        *,
        lift_targets: Path,
        source_bodex_bank: Path,
        profile: LiftBridgeProfile,
        **kwargs: Any,
    ):
        self.bridge_profile = profile
        self.bridge_lift_targets_path = Path(lift_targets)
        self.bridge_source_bodex_bank = Path(source_bodex_bank)
        super().__init__(cfg, **kwargs)
        payload = torch.load(
            self.bridge_lift_targets_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema") != "xhand_bodex_nonpromotional_micro_lift_targets_v1":
            raise RuntimeError("wrong micro-lift target schema")
        if payload.get("source_bodex_bank_sha256") != sha256_file(
            self.bridge_source_bodex_bank
        ):
            raise RuntimeError("micro-lift targets do not match the BODex bank")
        if list(payload.get("candidate_ids", [])) != list(self.bodex_candidate_ids):
            raise RuntimeError("micro-lift target candidate order changed")
        key = f"{profile.target_height_m:.6f}"
        rows = payload.get("targets_by_height", {}).get(key)
        if not isinstance(rows, list) or len(rows) != len(self.bodex_candidate_ids):
            raise RuntimeError(f"micro-lift targets lack height {key}")
        source_index = {
            name: index for index, name in enumerate(payload["joint_names"])
        }
        missing = [name for name in self.joint_names if name not in source_index]
        if missing:
            raise RuntimeError(f"micro-lift targets lack joints: {missing}")
        self.bodex_lift_bank = torch.tensor(
            [
                [row[source_index[name]] for name in self.joint_names]
                for row in rows
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if profile.action_group not in ("hands", "distal_wrist"):
            raise ValueError(
                "lift bridge action group must be hands or distal_wrist"
            )
        if profile.action_group == "distal_wrist":
            # Keep the 38D policy ABI, but expose only the two distal wrist
            # triplets to this isolated formation retry.  Nominal BODex
            # finger closure remains entirely controller-driven.
            self.action_mask.zero_()
            for index, name in enumerate(self.joint_names):
                if name.endswith(("_j5", "_j6", "_j7")):
                    self.action_mask[0, index] = 1.0
        self.bridge_current_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.bridge_maximum_stable_steps = torch.zeros_like(
            self.bridge_current_stable_steps
        )
        self.bridge_physically_bounded = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.bridge_current_safe = torch.ones_like(self.bridge_physically_bounded)
        self.bridge_maximum_linear_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self.bridge_maximum_angular_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self.bridge_phase_height_bounded = torch.ones(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self.bridge_phase_speed_safe = torch.ones_like(
            self.bridge_phase_height_bounded
        )
        self.bridge_phase_maximum_height = torch.full(
            (self.num_envs, 4), -torch.inf, device=self.device
        )
        self.bridge_phase_minimum_height = torch.full(
            (self.num_envs, 4), torch.inf, device=self.device
        )
        self.bridge_phase_maximum_linear_speed = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self.bridge_phase_maximum_angular_speed = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._reset_idx(self.robot._ALL_INDICES)

    def _residual_contact_gate(self) -> torch.Tensor:
        """Return the per-environment gate for contact-conditioned residuals.

        The gate is deliberately computed from current bilateral contact and
        the Stage-2 conjunction progress, rather than from historical maximum
        contact alone.  This prevents a policy from retaining a learned hand
        offset after one hand has already slipped away.
        """

        profile = self.bridge_profile
        if not bool(getattr(profile, "contact_gated_residual", False)):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        bilateral_now = (self._sensor_force("left") > 0.02) & (
            self._sensor_force("right") > 0.02
        )
        threshold = float(getattr(profile, "contact_gate_activation_threshold", 1.0))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("contact_gate_activation_threshold must be in [0, 1]")
        return bilateral_now & (self._contact_gate_progress() >= threshold)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Keep approach/closure exactly on the BODex/IK nominal trajectory.
        # Once a gated profile is active, a contact loss immediately removes
        # both new action input and any accumulated residual offset.
        gate = self._residual_contact_gate()
        if bool(getattr(self.bridge_profile, "contact_gated_residual", False)):
            actions = actions * gate.unsqueeze(-1).to(actions.dtype)
        super()._pre_physics_step(actions)
        if bool(getattr(self.bridge_profile, "contact_gated_residual", False)):
            self.integrated_residual[~gate] = 0.0
            self.extras.setdefault("log", {})["bridge/residual_gate_fraction"] = (
                gate.float().mean()
            )
            self.extras["log"]["bridge/residual_gate_progress"] = (
                self._contact_gate_progress().mean()
            )

    def _reset_idx(self, env_ids: Any) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "bridge_current_stable_steps"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.bridge_current_stable_steps[env_ids] = 0
        self.bridge_maximum_stable_steps[env_ids] = 0
        self.bridge_physically_bounded[env_ids] = True
        self.bridge_current_safe[env_ids] = True
        self.bridge_maximum_linear_speed[env_ids] = 0.0
        self.bridge_maximum_angular_speed[env_ids] = 0.0
        self.bridge_phase_height_bounded[env_ids] = True
        self.bridge_phase_speed_safe[env_ids] = True
        self.bridge_phase_maximum_height[env_ids] = -torch.inf
        self.bridge_phase_minimum_height[env_ids] = torch.inf
        self.bridge_phase_maximum_linear_speed[env_ids] = 0.0
        self.bridge_phase_maximum_angular_speed[env_ids] = 0.0

    def _bridge_state(self) -> dict[str, torch.Tensor]:
        profile = self.bridge_profile
        height = self.object.data.root_pos_w[:, 2] - (
            self.initial_object_pos[:, 2] + self.scene.env_origins[:, 2]
        )
        linear_speed = torch.linalg.vector_norm(
            self.object.data.root_lin_vel_w, dim=-1
        )
        angular_speed = torch.linalg.vector_norm(
            self.object.data.root_ang_vel_w, dim=-1
        )
        finite = (
            torch.isfinite(height)
            & torch.isfinite(linear_speed)
            & torch.isfinite(angular_speed)
        )
        bounded = (
            finite
            & (height >= profile.minimum_height_m)
            & (height <= profile.maximum_height_m)
        )
        safe = (
            bounded
            & (linear_speed <= profile.safety_linear_speed_max_m_s)
            & (angular_speed <= profile.safety_angular_speed_max_rad_s)
        )
        # Some historical bridge profiles intentionally did not make the
        # expensive PhysX penetration audit part of the path-wise termination
        # contract.  Stable-lift profiles opt in through
        # ``penetration_limit_m`` so PPO cannot trade deeper contact for lift
        # reward and then recover before the terminal report.
        penetration_safe = torch.ones_like(safe)
        if profile.penetration_limit_m is not None:
            limit = float(profile.penetration_limit_m)
            if limit <= 0.0:
                raise ValueError("penetration_limit_m must be positive")
            penetration_safe = (~self.penetration_measured) | (
                self.maximum_penetration <= limit
            )
            safe &= penetration_safe
        stable = (
            safe
            & (height >= 0.8 * profile.target_height_m)
            & (linear_speed <= profile.linear_speed_max_m_s)
            & (angular_speed <= profile.angular_speed_max_rad_s)
        )
        return {
            "height": height,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "finite": finite,
            "bounded": bounded,
            "safe": safe,
            "penetration_safe": penetration_safe,
            "stable": stable,
        }

    def _update_metrics(self, code: torch.Tensor) -> None:
        super()._update_metrics(code)
        if not hasattr(self, "bridge_current_stable_steps"):
            return
        state = self._bridge_state()
        self.bridge_current_safe = state["safe"]
        self.bridge_physically_bounded &= state["safe"]
        self.bridge_maximum_linear_speed = torch.maximum(
            self.bridge_maximum_linear_speed,
            torch.where(
                state["finite"],
                state["linear_speed"],
                self.bridge_maximum_linear_speed,
            ),
        )
        self.bridge_maximum_angular_speed = torch.maximum(
            self.bridge_maximum_angular_speed,
            torch.where(
                state["finite"],
                state["angular_speed"],
                self.bridge_maximum_angular_speed,
            ),
        )
        self.bridge_current_stable_steps = torch.where(
            state["stable"],
            self.bridge_current_stable_steps + 1,
            torch.zeros_like(self.bridge_current_stable_steps),
        )
        self.bridge_maximum_stable_steps = torch.maximum(
            self.bridge_maximum_stable_steps,
            self.bridge_current_stable_steps,
        )
        height_bounded = (
            state["finite"]
            & (state["height"] >= self.bridge_profile.minimum_height_m)
            & (state["height"] <= self.bridge_profile.maximum_height_m)
        )
        speed_safe = (
            state["finite"]
            & (
                state["linear_speed"]
                <= self.bridge_profile.safety_linear_speed_max_m_s
            )
            & (
                state["angular_speed"]
                <= self.bridge_profile.safety_angular_speed_max_rad_s
            )
        )
        for phase_index, phase_value in enumerate(
            (PHASE_APPROACH, PHASE_CLOSE, PHASE_LIFT, PHASE_HOLD)
        ):
            mask = code == phase_value
            self.bridge_phase_height_bounded[:, phase_index] &= (~mask) | height_bounded
            self.bridge_phase_speed_safe[:, phase_index] &= (~mask) | speed_safe
            self.bridge_phase_maximum_height[:, phase_index] = torch.where(
                mask & state["finite"],
                torch.maximum(
                    self.bridge_phase_maximum_height[:, phase_index], state["height"]
                ),
                self.bridge_phase_maximum_height[:, phase_index],
            )
            self.bridge_phase_minimum_height[:, phase_index] = torch.where(
                mask & state["finite"],
                torch.minimum(
                    self.bridge_phase_minimum_height[:, phase_index], state["height"]
                ),
                self.bridge_phase_minimum_height[:, phase_index],
            )
            self.bridge_phase_maximum_linear_speed[:, phase_index] = torch.where(
                mask & state["finite"],
                torch.maximum(
                    self.bridge_phase_maximum_linear_speed[:, phase_index],
                    state["linear_speed"],
                ),
                self.bridge_phase_maximum_linear_speed[:, phase_index],
            )
            self.bridge_phase_maximum_angular_speed[:, phase_index] = torch.where(
                mask & state["finite"],
                torch.maximum(
                    self.bridge_phase_maximum_angular_speed[:, phase_index],
                    state["angular_speed"],
                ),
                self.bridge_phase_maximum_angular_speed[:, phase_index],
            )

    def _contact_gate_progress(self) -> torch.Tensor:
        spec = self.stage_spec
        bilateral_progress = torch.clamp(
            self.current_bilateral_contact_steps.float()
            / max(spec.required_bilateral_steps, 1),
            0.0,
            1.0,
        )
        continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_progress = torch.clamp(
            torch.minimum(continuity_left, continuity_right)
            / max(spec.contact_presence_fraction_min, 1.0e-8),
            0.0,
            1.0,
        )
        distributed = torch.clamp(
            torch.minimum(
                self.maximum_contact_group_counts[:, 0],
                self.maximum_contact_group_counts[:, 1],
            ).float()
            / max(spec.contact_groups_per_side_min, 1),
            0.0,
            1.0,
        )
        return bilateral_progress * continuity_progress * distributed

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        if not hasattr(self, "bridge_current_stable_steps"):
            return reward
        profile = self.bridge_profile
        state = self._bridge_state()
        # The acceptance contract is path-wise: one unsafe height/speed event
        # permanently invalidates the episode.  Previously the dense reward
        # only checked the current frame, so an overshooting policy could
        # recover later and continue collecting stability rewards even though
        # controlled_micro_lift was already impossible.  Keep all positive
        # shaping aligned with the same historical validity bit used by the
        # evaluator.
        trajectory_valid = self.bridge_physically_bounded.float()
        reward = reward * trajectory_valid
        code = self._executed_phase_code()
        lift_or_hold = (code == PHASE_LIFT) | (code == PHASE_HOLD)
        hold = code == PHASE_HOLD
        phase = lift_or_hold.float()
        valid_phase = trajectory_valid * phase
        contact_progress = self._contact_gate_progress()
        bilateral_now = (self._sensor_force("left") > 0.02) & (
            self._sensor_force("right") > 0.02
        )
        target = max(profile.target_height_m, 1.0e-6)
        height_progress = torch.clamp(state["height"] / target, 0.0, 1.0)
        tracking_width = max(0.75 * target, 0.003)
        height_tracking = torch.exp(
            -torch.square((state["height"] - target) / tracking_width)
        )
        stable_score = torch.exp(
            -torch.relu(
                state["linear_speed"] - profile.linear_speed_max_m_s
            )
            / 0.10
            - torch.relu(
                state["angular_speed"] - profile.angular_speed_max_rad_s
            )
            / 1.0
        )
        height_valid = trajectory_valid * state["safe"].float() * bilateral_now.float()
        height_progress = height_progress * height_valid * phase
        height_tracking = height_tracking * height_valid * phase
        # Keep the speed/settling signal dense while the object is still
        # approaching the target height.  The previous binary 0.8*target
        # cutoff made every pre-target frame indistinguishable to PPO, so it
        # learned to chase height and only discovered excessive velocity after
        # the lift was already unrecoverable.
        height_stability_progress = torch.clamp(
            state["height"] / max(0.8 * target, 1.0e-6), 0.0, 1.0
        )
        stable_score = (
            stable_score
            * height_valid
            * height_stability_progress
            * phase
        )
        stable_progress = torch.clamp(
            self.bridge_current_stable_steps.float()
            / max(profile.stable_hold_steps, 1),
            0.0,
            1.0,
        )
        overshoot = torch.clamp(
            torch.relu(state["height"] - profile.maximum_height_m) / target,
            0.0,
            5.0,
        )
        drop = torch.clamp(
            torch.relu(0.8 * target - state["height"]) / target,
            0.0,
            5.0,
        )
        linear_penalty = torch.clamp(
            torch.relu(state["linear_speed"] - profile.linear_speed_max_m_s),
            0.0,
            5.0,
        )
        angular_penalty = torch.clamp(
            torch.relu(state["angular_speed"] - profile.angular_speed_max_rad_s),
            0.0,
            20.0,
        )
        controlled = (
            (self.maximum_lift_height >= target)
            & (state["height"] >= 0.8 * target)
            & self.bridge_physically_bounded
        )
        stable_ready = controlled & (
            self.bridge_current_stable_steps >= profile.stable_hold_steps
        )
        final_step = self.episode_length_buf >= self.max_episode_length - 1
        bonus = (
            profile.height_progress_reward_weight * height_progress
            + profile.height_tracking_reward_weight * height_tracking
            + profile.stability_reward_weight * stable_score
            + profile.stable_hold_reward_weight * stable_progress * valid_phase
            + profile.contact_gate_reward_weight * contact_progress * valid_phase
            + profile.terminal_controlled_reward_weight
            * controlled.float()
            * final_step.float()
            + profile.terminal_stable_reward_weight
            * stable_ready.float()
            * final_step.float()
            - profile.overshoot_penalty_weight * torch.square(overshoot) * phase
            - profile.drop_penalty_weight * torch.square(drop) * phase
            - profile.linear_speed_penalty_weight * linear_penalty * phase
            - profile.angular_speed_penalty_weight * angular_penalty * phase
            - profile.unsafe_state_penalty_weight
            * (~self.bridge_physically_bounded).float()
        )
        self.extras.setdefault("log", {})["bridge/height_progress"] = (
            height_progress.mean()
        )
        self.extras["log"]["bridge/height_tracking"] = height_tracking.mean()
        self.extras["log"]["bridge/stability_score"] = stable_score.mean()
        self.extras["log"]["bridge/stable_hold_progress"] = stable_progress.mean()
        self.extras["log"]["bridge/contact_gate_progress"] = contact_progress.mean()
        self.extras["log"]["bridge/controlled_fraction"] = controlled.float().mean()
        self.extras["log"]["bridge/stable_ready_fraction"] = stable_ready.float().mean()
        self.extras["log"]["bridge/overshoot_penalty"] = overshoot.mean()
        self.extras["log"]["bridge/angular_speed_rad_s"] = state[
            "angular_speed"
        ].mean()
        self.extras["log"]["bridge/trajectory_valid_fraction"] = (
            trajectory_valid.mean()
        )
        self.extras["log"]["bridge/penetration_safety_fraction"] = state[
            "penetration_safe"
        ].float().mean()
        return reward + float(self.cfg.training_reward_scale) * bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if bool(self.cfg.bridge_terminate_on_unsafe):
            # Failure-funnel rejection sampling: once the path-wise bridge
            # contract is false, this rollout can never become an accepted
            # sample.  End it immediately instead of spending the remainder
            # of the PPO horizon assigning credit on an unrecoverable state.
            terminated |= ~self.bridge_physically_bounded
            time_out &= ~terminated
        return terminated, time_out

    def bridge_metrics_report(self, env_id: int) -> dict[str, Any]:
        report = self._metrics_report(env_id)
        state = self._bridge_state()
        phase_names = ("approach", "close", "lift", "hold")
        phase_metrics = {}
        first_unsafe_phase = None
        for phase_index, phase_name in enumerate(phase_names):
            height_bounded = bool(
                self.bridge_phase_height_bounded[env_id, phase_index].item()
            )
            speed_safe = bool(
                self.bridge_phase_speed_safe[env_id, phase_index].item()
            )
            if first_unsafe_phase is None and not (height_bounded and speed_safe):
                first_unsafe_phase = phase_name
            phase_metrics[phase_name] = {
                "height_bounded": height_bounded,
                "speed_safe": speed_safe,
                "safe": height_bounded and speed_safe,
                "maximum_height_m": float(
                    self.bridge_phase_maximum_height[env_id, phase_index].item()
                ),
                "minimum_height_m": float(
                    self.bridge_phase_minimum_height[env_id, phase_index].item()
                ),
                "maximum_linear_speed_m_s": float(
                    self.bridge_phase_maximum_linear_speed[env_id, phase_index].item()
                ),
                "maximum_angular_speed_rad_s": float(
                    self.bridge_phase_maximum_angular_speed[env_id, phase_index].item()
                ),
            }
        report.update(
            {
                "bridge_profile": self.bridge_profile.to_dict(),
                "final_lift_height_m": float(state["height"][env_id].item()),
                "micro_lift_physically_bounded": bool(
                    self.bridge_physically_bounded[env_id].item()
                ),
                "maximum_micro_lift_stable_steps": int(
                    self.bridge_maximum_stable_steps[env_id].item()
                ),
                "maximum_micro_lift_linear_speed_m_s": float(
                    self.bridge_maximum_linear_speed[env_id].item()
                ),
                "maximum_micro_lift_angular_speed_rad_s": float(
                    self.bridge_maximum_angular_speed[env_id].item()
                ),
                "bridge_phase_metrics": phase_metrics,
                "first_unsafe_bridge_phase": first_unsafe_phase,
                "penetration_safety_limit_m": self.bridge_profile.penetration_limit_m,
                "penetration_safe_at_terminal": bool(
                    state["penetration_safe"][env_id].item()
                ),
            }
        )
        return report
