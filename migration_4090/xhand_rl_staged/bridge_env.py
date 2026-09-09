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
        retracted_pregrasp: Path | None = None,
        retract_distance_m: float = 0.10,
        finger_close_scale: float = 1.0,
        freeze_object_until_bilateral: bool = False,
        freeze_release_force_n: float = 0.02,
        freeze_release_hold_steps: int = 4,
        latch_fingers_on_release: bool = False,
        **kwargs: Any,
    ):
        self.bridge_profile = profile
        self.bridge_lift_targets_path = Path(lift_targets)
        self.bridge_source_bodex_bank = Path(source_bodex_bank)
        self.bridge_retracted_pregrasp_path = (
            Path(retracted_pregrasp) if retracted_pregrasp else None
        )
        self.bridge_retract_distance_m = float(retract_distance_m)
        # How far past the BODex grasp pose the fingers are driven.
        #
        # Measured 2026-09-08: the grasp poses do not touch the object.  Their
        # minimum clearance to the exact mesh is 0.34, 0.96, 2.48 and 13.47 mm on
        # the four zero_overlap4 candidates, and against the convex hulls the
        # simulator actually collides with it is -0.44 to -13.17 mm, i.e. clear
        # either way.  Executed faithfully the hands close on empty space: with
        # self-collisions disabled, so that the fingers hold their commanded
        # pose, contact is 0.000, penetration is 0.000 and the ball rises 3.3 mm.
        #
        # Every contact this pipeline has ever made came from phantom
        # self-collisions between convex hulls pushing the fingers 48 degrees off
        # target into positions that happen to reach the ball, which is why lift
        # success tracks grasp clearance so exactly: 0.34 mm gives 176/256 and
        # 13.47 mm gives 0/256.
        #
        # This extends the close along the same pregrasp-to-grasp direction so
        # the fingers reach the surface deliberately.  1.0 reproduces the old
        # behaviour exactly.
        self.bridge_finger_close_scale = float(finger_close_scale)
        self.bridge_freeze_object = bool(freeze_object_until_bilateral)
        self.bridge_freeze_release_force_n = float(freeze_release_force_n)
        self.bridge_freeze_release_hold_steps = int(freeze_release_hold_steps)
        self.bridge_latch_fingers_on_release = bool(latch_fingers_on_release)
        self.bridge_retracted_pregrasp = None
        super().__init__(cfg, **kwargs)
        self._load_retracted_pregrasp()
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
        if profile.action_group not in ("hands", "distal_wrist", "all"):
            raise ValueError(
                "lift bridge action group must be hands, distal_wrist or all"
            )
        # "all" is the standoff formulation and deliberately breaks the bridge's
        # founding assumption that the BODex arm trajectory is replayed
        # untouched.  That assumption is what the 2026-09-07 measurements ran
        # out of road on: with the hands starting between -9.6 and +2.4 mm from
        # the ball and the arms on rails, no reward, iteration count or gain
        # setting moved a policy past a zero action vector.  The mask the parent
        # built from the action group already covers every joint, so nothing is
        # overridden here.
        if profile.action_group == "distal_wrist":
            # Keep the 38D policy ABI, but expose only the two distal wrist
            # triplets to this isolated formation retry.  Nominal BODex
            # finger closure remains entirely controller-driven.
            self.action_mask.zero_()
            for index, name in enumerate(self.joint_names):
                if name.endswith(("_j5", "_j6", "_j7")):
                    self.action_mask[0, index] = 1.0
        self.bridge_hand_joint_mask = torch.tensor(
            [["_hand_" in name for name in self.joint_names]],
            dtype=torch.bool,
            device=self.device,
        )
        # Grasp-acquisition freeze.  frozen[i] is true while environment i still
        # has the object pinned at its reset pose; released_steps counts the
        # consecutive steps both hands have been in contact, and the latch opens
        # once for good.
        # Armed by _reset_idx, never here.  The pose to hold is only known once
        # an episode has been reset; writing the zero-initialised state would
        # push a degenerate quaternion into PhysX, which is what tripped the
        # device-side assert on the first freeze runs.
        self.bridge_object_frozen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.bridge_freeze_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.bridge_frozen_object_state = torch.zeros(
            (self.num_envs, 13), dtype=torch.float32, device=self.device
        )
        self.bridge_freeze_release_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        # Finger targets captured at release, so the hand stops closing once it
        # has the object instead of driving further into it.
        self.bridge_latched_finger_target = torch.zeros(
            (self.num_envs, int(self.cfg.action_space)),
            dtype=torch.float32,
            device=self.device,
        )
        self.bridge_fingers_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Consecutive steps where the stability condition and bilateral contact
        # hold at the same time.  See _update_metrics.
        self.bridge_current_held_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.bridge_maximum_held_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
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

    def _load_retracted_pregrasp(self) -> None:
        """Load the per-candidate retracted start pose, if one was supplied."""
        if self.bridge_retracted_pregrasp_path is None:
            return
        payload = torch.load(
            self.bridge_retracted_pregrasp_path, map_location="cpu", weights_only=False
        )
        if payload.get("schema") != "xhand_bodex_retracted_pregrasp_v1":
            raise RuntimeError("wrong retracted pregrasp schema")
        if payload.get("source_bodex_bank_sha256") != sha256_file(
            self.bridge_source_bodex_bank
        ):
            raise RuntimeError("retracted pregrasp was built from a different bank")
        key = f"{self.bridge_retract_distance_m:.6f}"
        table = payload["retracted_pregrasp_full_body_q"]
        if key not in table:
            raise RuntimeError(
                f"retract distance {key} not in {sorted(table)}"
            )
        source_names = list(payload["joint_names"])
        rows = table[key].tolist()
        index_of = {name: i for i, name in enumerate(source_names)}
        missing = [n for n in self.joint_names if n not in index_of]
        if missing:
            raise RuntimeError(f"retracted pregrasp lacks joints: {missing}")
        self.bridge_retracted_pregrasp = torch.tensor(
            [[row[index_of[name]] for name in self.joint_names] for row in rows],
            dtype=torch.float32,
            device=self.device,
        )

    def _retracted_start(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        if self.bridge_retracted_pregrasp is None:
            return None
        return self.bridge_retracted_pregrasp[self.active_bank_index[env_ids]]

    def _base_target(self, code: torch.Tensor) -> torch.Tensor:
        """Interpolate the retracted start into the pregrasp during approach.

        The parent clamps its close factor to zero before PHASE_CLOSE, which
        makes the approach phase a constant hold.  Here the same phase carries
        the robot from the retracted pose to the BODex pregrasp; every later
        phase is left to the parent untouched.
        """
        target = super()._base_target(code)
        if self.bridge_finger_close_scale != 1.0:
            hand = self.bridge_hand_joint_mask
            closing = target - self.nominal_pregrasp
            extended = self.nominal_pregrasp + (
                closing * self.bridge_finger_close_scale
            )
            limits = self.robot.data.soft_joint_pos_limits
            extended = torch.clamp(extended, limits[..., 0], limits[..., 1])
            beyond_approach = (code != PHASE_APPROACH).unsqueeze(-1)
            target = torch.where(hand & beyond_approach, extended, target)
        if self.bridge_latch_fingers_on_release and bool(
            self.bridge_fingers_latched.any()
        ):
            hand = self.bridge_hand_joint_mask & self.bridge_fingers_latched.unsqueeze(
                -1
            )
            target = torch.where(hand, self.bridge_latched_finger_target, target)
        if self.bridge_retracted_pregrasp is None:
            return target
        approach = code == PHASE_APPROACH
        if not bool(approach.any()):
            return target
        start, _, _, _, _, _ = self.phase_boundaries
        alpha = (
            (self.episode_length_buf.float() / max(float(start), 1.0))
            .clamp(0.0, 1.0)
            .unsqueeze(-1)
        )
        retracted = self.bridge_retracted_pregrasp[self.active_bank_index]
        blended = retracted + alpha * (self.nominal_pregrasp - retracted)
        return torch.where(approach.unsqueeze(-1), blended, target)

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

    def _hold_frozen_object(self) -> None:
        """Pin the object at its reset pose until both hands are on it.

        The latch opens when both sides report contact force for
        freeze_release_hold_steps consecutive steps, and never closes again.  A
        single hand is not enough: releasing on one contact is what turns the
        neighbouring task's lifts into ejections, their own report showing
        final_lift 1.000 at 90 degrees of tilt with no bilateral hold.

        While frozen the pose is written every step with zero velocity.  That is
        safe only before contact; once a hand presses on it, re-asserting the
        pose drives the body through the collider, which is why release is
        checked first and the write is skipped for any environment released this
        step.
        """
        if not self.bridge_freeze_object:
            return
        frozen = self.bridge_object_frozen
        if not bool(frozen.any()):
            return
        threshold = self.bridge_freeze_release_force_n
        both = (self._sensor_force("left") > threshold) & (
            self._sensor_force("right") > threshold
        )
        self.bridge_freeze_contact_steps = torch.where(
            both & frozen,
            self.bridge_freeze_contact_steps + 1,
            torch.zeros_like(self.bridge_freeze_contact_steps),
        )
        # The freeze must not outlive the closure phase.  Measured with it
        # unbounded, 70% of episodes were still holding the object when the lift
        # phase began, so the arms drove upward against a pinned ball for about
        # 120 steps and wedged into it: 21-29 mm of penetration, present even
        # with no extra finger closure at all.  Releasing here hands a possibly
        # ungrasped object to the lift, which the lift criteria will fail
        # honestly, rather than manufacturing contact by pressing.
        _, close_end, _, _, _, _ = self.phase_boundaries
        release = frozen & (
            (self.bridge_freeze_contact_steps >= self.bridge_freeze_release_hold_steps)
            | (self.episode_length_buf >= close_end)
        )
        if bool(release.any()):
            self.bridge_freeze_release_step[release] = self.episode_length_buf[release]
            self.bridge_object_frozen[release] = False
            if self.bridge_latch_fingers_on_release:
                # The hand has the object; closing further only drives into it.
                # Measured without this: penetration reaches 22-29 mm by the end
                # of the closure phase and the overlap resolves against the ball
                # when the freeze lifts, which is what contact_continuity of 0.12
                # looks like from the object's side.
                self.bridge_latched_finger_target[release] = (
                    self.robot.data.joint_pos[release]
                )
                self.bridge_fingers_latched[release] = True
            frozen = self.bridge_object_frozen
        ids = torch.nonzero(frozen, as_tuple=False).squeeze(-1)
        if ids.numel():
            state = self.bridge_frozen_object_state[ids].clone()
            state[:, 7:] = 0.0
            self.object.write_root_state_to_sim(state, env_ids=ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Keep approach/closure exactly on the BODex/IK nominal trajectory.
        # Once a gated profile is active, a contact loss immediately removes
        # both new action input and any accumulated residual offset.
        gate = self._residual_contact_gate()
        if bool(getattr(self.bridge_profile, "contact_gated_residual", False)):
            actions = actions * gate.unsqueeze(-1).to(actions.dtype)
        super()._pre_physics_step(actions)
        # Before the physics of this step, so a frozen object never integrates
        # the disturbance the approach would otherwise give it.
        self._hold_frozen_object()
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
        start = self._retracted_start(env_ids)
        if start is not None:
            # The parent already wrote nominal_pregrasp and its reset noise;
            # carry that same perturbation over so the retracted start keeps the
            # episode-to-episode variation rather than removing it.
            offset = self.robot.data.joint_pos[env_ids] - self.nominal_pregrasp[env_ids]
            q = start + offset
            limits = self.robot.data.soft_joint_pos_limits[env_ids]
            q = torch.clamp(q, limits[..., 0], limits[..., 1])
            self.robot.write_joint_state_to_sim(
                q, torch.zeros_like(q), env_ids=env_ids
            )
            self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.bridge_current_stable_steps[env_ids] = 0
        self.bridge_current_held_steps[env_ids] = 0
        self.bridge_maximum_held_steps[env_ids] = 0
        if self.bridge_freeze_object:
            self.bridge_object_frozen[env_ids] = True
            self.bridge_freeze_contact_steps[env_ids] = 0
            self.bridge_freeze_release_step[env_ids] = -1
            self.bridge_fingers_latched[env_ids] = False
            self.bridge_frozen_object_state[env_ids] = self.object.data.root_state_w[
                env_ids
            ].clone()
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
        held = state["stable"] & (
            (self._sensor_force("left") > 0.02)
            & (self._sensor_force("right") > 0.02)
        )
        self.bridge_current_held_steps = torch.where(
            held,
            self.bridge_current_held_steps + 1,
            torch.zeros_like(self.bridge_current_held_steps),
        )
        self.bridge_maximum_held_steps = torch.maximum(
            self.bridge_maximum_held_steps,
            self.bridge_current_held_steps,
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
                # Longest run where stability and bilateral contact held at the
                # same time.  The field above counts stability alone, so an
                # episode could satisfy the hold from one stretch of the rollout
                # and the contact criteria from another.
                "maximum_micro_lift_held_steps": int(
                    self.bridge_maximum_held_steps[env_id].item()
                ),
                # -1 means the object was never frozen, or was still frozen when
                # the episode ended: no bilateral contact ever formed.
                "freeze_release_step": int(
                    self.bridge_freeze_release_step[env_id].item()
                ),
                "object_frozen_at_end": bool(
                    self.bridge_object_frozen[env_id].item()
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
