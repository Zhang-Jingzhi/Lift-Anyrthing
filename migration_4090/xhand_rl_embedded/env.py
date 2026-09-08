"""Isaac Lab environment with physical acceptance embedded in each RL episode."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from migration_4090.xhand_rl.contact_layout import (
    HAND_BODY_SUFFIXES,
    contact_filter_layout,
    side_filter_indices,
)
from migration_4090.xhand_rl.env import load_nominal_joint_trajectory
from migration_4090.xhand_rl.force_closure import evaluate_force_closure

from .gates import evaluate_physics_metrics
from .rewards import (
    EmbeddedRewardWeights,
    compute_embedded_reward,
    strict_disturbance_pass_fraction,
    strict_disturbance_prefix_fraction,
)


PHASE_APPROACH = 0
PHASE_CLOSE = 1
PHASE_LIFT = 2
PHASE_HOLD = 3
PHASE_DISTURBANCE_FIRST = 4
PHASE_DISTURBANCE_LAST = 15
PHASE_LEFT_ONLY = 16
PHASE_RIGHT_ONLY = 17
PHASE_COUNT = 18


@configclass
class XHandEmbeddedEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 8.0
    action_space = 38
    observation_space = 145
    state_space = 0
    is_finite_horizon = True

    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8),
        physics_material=RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=16,
            max_position_iteration_count=16,
            min_velocity_iteration_count=4,
            max_velocity_iteration_count=4,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**21,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    robot_usd = ""
    object_usd = ""
    object_key = ""
    object_extents_m = (0.2, 0.2, 0.2)
    object_mass_kg = 0.55
    contact_friction = 2.0
    contact_offset_m = 0.0045
    rest_offset_m = 0.0005
    # Position-control gains for the arms and hands.
    #
    # These were 1500/120 and 50/2, roughly twice the 800/60 and 30/2 the
    # neighbouring skrl task uses on the same robot.  A stiffer position-
    # controlled arm drives through an obstacle instead of yielding to it, which
    # is what the 2026-09-07 measurements show: when the ball is knocked clear
    # before the fingers close, hand-object penetration is 3 mm; when a retracted
    # start leaves the ball where the pregrasp expects it, the same pose
    # penetrates 10-12 mm.  Exposed so the gains can be compared rather than
    # assumed.
    # PhysX separates interpenetrating bodies at up to this speed.  Measured
    # 2026-09-08 on the standoff configuration: median hand-object penetration
    # is 8.2 mm and the object's micro-lift speeds are 0.567 m/s linear and
    # 5.84 rad/s angular against bounds of 0.05 and 0.5 -- eleven times over,
    # at the median rather than in a tail.  The per-step capture shows the
    # signature of injected energy rather than a launch: velocity spikes
    # throughout with only a slow net climb.  Depenetration at 2 m/s against
    # 8 mm of overlap is the obvious candidate, so it is exposed to be tested.
    # Measured 2026-09-08: left_hand_pinky_joint1 leaves its commanded position
    # by 0.84-0.89 rad (about 50 degrees) within 40 steps and stays there, in the
    # stock configuration and in every variant tried -- with the hands 89 mm
    # clear of the ball, so nothing external touches it, and with the arm joints
    # tracking to 0.0007 rad over the same window.  Raising hand stiffness from
    # 4 to 50 barely moves it (0.832 -> 0.878 rad) because the URDF caps that
    # joint's effort at 1.1 N.m while the PD would need about 44.  Self-collision
    # is the remaining candidate for what pushes it, so it is exposed to be
    # tested rather than assumed.
    robot_self_collisions = True
    object_max_depenetration_velocity = 2.0
    arm_stiffness = 1500.0
    arm_damping = 120.0
    hand_stiffness = 50.0
    hand_damping = 2.0
    table_top_z_m = 0.72
    nominal_dataset = ""
    nominal_sample_index = 0
    # Run-level override for experiments that deliberately remove reset-pose
    # randomization. A value of True is ORed with per-sample metadata so a
    # BODex bank need not be rewritten for a fixed-pose ablation.
    nominal_pose_lock = False
    # Legacy BODex/strict trajectories describe the object pose reached after
    # closure, while the replay starts with the object centered on the table.
    # Keep the nominal pose in the sample for provenance, but do not teleport
    # the object to that post-closure pose at reset unless a caller explicitly
    # opts into the historical behavior.
    use_nominal_object_pose_for_reset = False
    reset_xy_noise_m = 0.015
    reset_yaw_noise_rad = 0.25
    reset_joint_noise_rad = 0.025
    # The original embedded run used a very small instantaneous residual.  It
    # was enough to perturb a nominal pose, but not enough to move the fingers
    # several millimetres out of a convex-decomposition contact while keeping
    # the object loaded.  Formal retries use an integrated residual with a
    # bounded workspace so the policy can make and then maintain that change.
    arm_action_scale_rad = 0.12
    hand_action_scale_rad = 0.24
    # ``instantaneous_residual_weight`` keeps the action semantics used by
    # the original warm-start policies available as an explicit mode.  The
    # current v5r runs use the integrated-only mode (weight 0); future retry
    # workers can select direct-only compatibility with weight 1 and zero
    # integration without changing any historical checkpoint.
    instantaneous_residual_weight = 0.0
    residual_integration = 0.03
    residual_limit_rad = 0.40
    # New retry runs can keep the nominal approach/close trajectory intact and
    # expose residual control only once the nominal grasp has been established.
    # The historical default (PHASE_APPROACH) preserves old checkpoint
    # semantics; supervisors opt into PHASE_LIFT for independent retries.
    residual_activation_phase = PHASE_APPROACH
    # A value below one lets the policy make an early hold correction and
    # then freezes the accumulated residual for passive settling.  Historical
    # runs integrate throughout hold.
    residual_integration_end_fraction_of_hold = 1.0
    approach_fraction = 0.10
    close_fraction = 0.25
    # Optional BODex-style decoupled close.  A positive value reserves the
    # first part of PHASE_CLOSE for arm/wrist approach while preserving the
    # open pregrasp fingers; the remaining close phase fixes the arms at the
    # final grasp pose and closes only the fingers.  Zero preserves every
    # historical trajectory and checkpoint evaluation exactly.
    arm_approach_fraction_of_close = 0.0
    # End of the finger-closing subphase.  Values below one reserve the tail
    # of PHASE_CLOSE as an explicit grasp-settle interval before lift.
    finger_close_end_fraction_of_close = 1.0
    # Formation retries may activate a tightly bounded residual only during
    # the tail of PHASE_CLOSE.  The accumulated residual is then held fixed
    # through lift/hold, so the nominal BODex finger closure remains intact.
    residual_activation_close_fraction = 0.0
    lift_fraction = 0.20
    hold_fraction = 0.15
    disturbance_fraction = 0.20
    ablation_fraction = 0.10
    lift_min_m = 0.05
    lift_reward_target_m = 0.08
    gravity_drift_max_m = 0.01
    translation_disturbance_max_m = 0.015
    rotation_disturbance_max_rad = 0.12
    contact_presence_fraction_min = 0.90
    inactive_hand_contact_fraction_max = 0.10
    physx_penetration_max_m = 0.0005
    force_closure_residual_max = 0.08
    force_closure_epsilon_min = 1.0e-4
    force_closure_reward_weight = 4.0
    stable_lift_reward_weight = 0.0
    # Phase-safe exact gate-prefix shaping.  Historical attempts keep zero;
    # strict-aligned v8 retries opt in explicitly.
    hard_gate_frontier_reward_weight = 0.0
    disturbance_direction_reward_weight = 0.0
    # Optional soft shaping term for the fixed-pose palm-orientation ablation.
    # Zero preserves all historical reward traces and hard gates.
    palm_center_alignment_reward_weight = 0.0
    # Optional PPO shaping only.  Strict acceptance is unchanged.  When true,
    # evaluate the real PhysX contact wrench geometry for every hold endpoint
    # so near-miss grasps receive continuous force-closure quality instead of
    # waiting until all lift/continuity prerequisites already pass.
    force_closure_shape_all_hold_contacts = False
    # New formal retries can gate dense force-closure shaping on the strict
    # same-trajectory clearance+gravity prefix.  Keep the historical default
    # false so older checkpoints and standalone reward tests retain their
    # original semantics.
    force_closure_reward_requires_clearance = False
    # Future formal retries can require the same-trajectory stable force-
    # closure prefix before receiving disturbance-direction shaping. This
    # removes a near-miss shortcut without changing strict acceptance.
    disturbance_reward_requires_force_closure = False
    disturbance_force_n = 0.275
    disturbance_torque_nm = 0.04
    training_reward_scale = 0.01
    # PPO shaping only; the terminal physical acceptance contract is
    # unchanged. Retry runs can raise this so a rare full-gate episode is not
    # numerically lost at the end of the 1000-policy-step horizon.
    terminal_success_weight = 30.0
    # Optional run-level override used for barrier-strength ablations.  The
    # locked manifest still defines the hard 0.5 mm acceptance gate; this only
    # changes the dense exploration shaping and is recorded in provenance.
    penetration_reward_weight = -8.0
    # Formal retries can lower approach proximity once a checkpoint already
    # contains contact/lift evidence, while increasing the clearance signal so
    # PPO cannot settle for a visually plausible but deeply penetrating pose.
    penetration_clear_reward_weight = 8.0
    proximity_reward_weight = 8.0
    bilateral_contact_reward_weight = 6.0
    contact_continuity_reward_weight = 4.0
    contact_diversity_reward_weight = 1.5
    lift_height_reward_weight = 12.0
    # Surface-distance shaping should distinguish a true outer-surface
    # approach from the object's center.  A tighter scale gives a useful
    # gradient near the contact band without changing any strict gate.
    proximity_scale_m = 0.08
    record_trajectory = False
    terminate_on_object_escape = True


def _quat_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dot = torch.abs(torch.sum(first * second, dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _quat_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized WXYZ quaternions to rotation matrices."""

    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
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


class XHandEmbeddedEnv(DirectRLEnv):
    """Residual PPO whose terminal success is the embedded physical contract."""

    cfg: XHandEmbeddedEnvCfg

    def __init__(self, cfg: XHandEmbeddedEnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.robot_usd or not cfg.object_usd or not cfg.nominal_dataset:
            raise ValueError("robot_usd, object_usd and nominal_dataset are required")
        if not 0.0 < cfg.training_reward_scale <= 1.0:
            raise ValueError("training_reward_scale must be in (0, 1]")
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_names = list(self.robot.joint_names)
        if len(self.joint_names) != self.cfg.action_space:
            raise RuntimeError(f"expected 38 actuated joints, got {len(self.joint_names)}")
        pre, grasp, lift, self.nominal_sample = load_nominal_joint_trajectory(
            Path(self.cfg.nominal_dataset),
            self.cfg.nominal_sample_index,
            self.joint_names,
            self.device,
        )
        self.nominal_pregrasp = pre.unsqueeze(0).repeat(self.num_envs, 1)
        self.nominal_grasp = grasp.unsqueeze(0).repeat(self.num_envs, 1)
        self.nominal_lift = lift.unsqueeze(0).repeat(self.num_envs, 1)
        self.nominal_pose_lock = bool(
            getattr(self.cfg, "nominal_pose_lock", False)
            or self.nominal_sample.get("nominal_pose_lock", False)
        )
        if self.nominal_pose_lock and self.cfg.use_nominal_object_pose_for_reset:
            position = self.nominal_sample.get("nominal_object_position_world")
            quaternion = self.nominal_sample.get("nominal_object_quaternion_wxyz")
            if (
                not isinstance(position, (list, tuple))
                or len(position) != 3
                or not isinstance(quaternion, (list, tuple))
                or len(quaternion) != 4
            ):
                raise RuntimeError("nominal_pose_lock requires explicit object position and quaternion")
            self.nominal_object_position = torch.tensor(
                [float(value) for value in position], dtype=torch.float32, device=self.device
            )
            nominal_quaternion = torch.tensor(
                [float(value) for value in quaternion], dtype=torch.float32, device=self.device
            )
            self.nominal_object_quaternion = nominal_quaternion / nominal_quaternion.norm().clamp_min(1.0e-8)
        else:
            self.nominal_object_position = None
            self.nominal_object_quaternion = None
        self.arm_ids = torch.tensor(
            [index for index, name in enumerate(self.joint_names) if "_hand_" not in name],
            dtype=torch.long,
            device=self.device,
        )
        self.left_ids = torch.tensor(
            [index for index, name in enumerate(self.joint_names) if name.startswith("left_")],
            dtype=torch.long,
            device=self.device,
        )
        self.right_ids = torch.tensor(
            [index for index, name in enumerate(self.joint_names) if name.startswith("right_")],
            dtype=torch.long,
            device=self.device,
        )
        body_names = list(self.robot.body_names)
        self.hand_body_ids = {}
        for side in ("left", "right"):
            ids = [
                index
                for index, name in enumerate(body_names)
                if name.startswith(f"{side}_hand")
            ]
            if not ids:
                raise RuntimeError(f"no {side} hand bodies found in robot articulation")
            self.hand_body_ids[side] = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.integrated_residual = torch.zeros_like(self.actions)
        self.targets = self.nominal_pregrasp.clone()
        self.action_scale = torch.full_like(self.actions, self.cfg.hand_action_scale_rad)
        self.action_scale[:, self.arm_ids] = self.cfg.arm_action_scale_rad
        self.object_start_z = self.cfg.table_top_z_m + 0.5 * self.cfg.object_extents_m[2]
        self.object_start_pos = torch.tensor(
            [0.5, 0.0, self.object_start_z], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.initial_object_pos = self.object_start_pos.clone()
        self.last_start_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.pregrasp_q = self.nominal_pregrasp.clone()
        self.closure_q = self.nominal_grasp.clone()
        self.lift_q = self.nominal_lift.clone()
        self.hold_robot_q = self.nominal_lift.clone()
        self.hold_object_state = torch.zeros((self.num_envs, 13), device=self.device)
        self.hold_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_quaternion = torch.zeros((self.num_envs, 4), device=self.device)
        self.lift_object_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.hold_state_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_phase = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.current_wrench = torch.zeros((self.num_envs, 6), device=self.device)
        self.contact_steps = torch.zeros(self.num_envs, device=self.device)
        self.left_contact_steps = torch.zeros(self.num_envs, device=self.device)
        self.right_contact_steps = torch.zeros(self.num_envs, device=self.device)
        self.max_hold_drift = torch.zeros(self.num_envs, device=self.device)
        self.gravity_drift = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.translation_displacements = torch.full((self.num_envs, 6), float("inf"), device=self.device)
        self.rotation_displacements = torch.full((self.num_envs, 6), float("inf"), device=self.device)
        # ``current_penetration`` drives the contact-clearance reward.
        # ``maximum_penetration`` is the episode-historical value used by both
        # the v8 barrier reward and the hard terminal gate.
        self.current_penetration = torch.zeros(self.num_envs, device=self.device)
        self.maximum_penetration = torch.zeros(self.num_envs, device=self.device)
        self.penetration_measured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ablation_steps = torch.zeros((self.num_envs, 2), device=self.device)
        self.ablation_inactive_contact_steps = torch.zeros((self.num_envs, 2), device=self.device)
        self.ablation_active_contact_steps = torch.zeros((self.num_envs, 2), device=self.device)
        self.single_hand_success = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.force_closure_pass = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_closure_evaluated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.force_closure_quality = torch.zeros(self.num_envs, device=self.device)
        self.force_closure_reports: list[dict[str, Any] | None] = [None] * self.num_envs
        self.completed_force_closure_reports: list[dict[str, Any] | None] = [None] * self.num_envs
        self.completed_reports: list[dict[str, Any] | None] = [None] * self.num_envs
        self.completed_pregrasp_q = torch.zeros_like(self.pregrasp_q)
        self.completed_closure_q = torch.zeros_like(self.closure_q)
        self.completed_lift_q = torch.zeros_like(self.lift_q)
        self.completed_start_pose = torch.zeros_like(self.last_start_pose)
        self.completed_episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.completed_success_serial = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_audited_episode_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_terminal_audited_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_terminal_phase = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_terminal_translation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_terminal_rotation_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_terminal_penetration_measured = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_terminal_force_closure_evaluated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.last_terminal_gate_pass = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Accumulate exact same-trajectory frontier statistics over one PPO
        # rollout.  The training hook consumes and resets these counters before
        # every optimizer update, allowing a rare near-miss policy to be saved
        # before PPO drifts away from it.
        self.training_prefix_score_sum = torch.zeros((), device=self.device)
        self.training_prefix_sample_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.training_prefix_max_level = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        # A scalar prefix level is insufficient to reproduce a rare contact
        # basin. Keep one exact same-trajectory state for append-only replay;
        # it is written by the PPO capture hook before the optimizer update.
        self.training_prefix_max_level_env = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.training_prefix_snapshot: dict[str, Any] | None = None
        self.training_prefix_snapshot_level = 0
        self._success_serial_counter = 0
        self._configure_schedule()
        self._allocate_trajectory_buffers()

    def _configure_schedule(self) -> None:
        if not 0.0 <= float(self.cfg.arm_approach_fraction_of_close) < 1.0:
            raise ValueError(
                "arm_approach_fraction_of_close must be in [0, 1)"
            )
        close_end = float(self.cfg.finger_close_end_fraction_of_close)
        if not float(self.cfg.arm_approach_fraction_of_close) < close_end <= 1.0:
            raise ValueError(
                "finger_close_end_fraction_of_close must be greater than "
                "arm_approach_fraction_of_close and at most one"
            )
        residual_end = float(
            self.cfg.residual_integration_end_fraction_of_hold
        )
        if not 0.0 < residual_end <= 1.0:
            raise ValueError(
                "residual_integration_end_fraction_of_hold must be in (0, 1]"
            )
        close_activation = float(self.cfg.residual_activation_close_fraction)
        if not 0.0 <= close_activation <= 1.0:
            raise ValueError(
                "residual_activation_close_fraction must be in [0, 1]"
            )
        fractions = (
            self.cfg.approach_fraction,
            self.cfg.close_fraction,
            self.cfg.lift_fraction,
            self.cfg.hold_fraction,
            self.cfg.disturbance_fraction,
            self.cfg.ablation_fraction,
        )
        if abs(sum(fractions) - 1.0) > 1.0e-6:
            raise ValueError("embedded episode fractions must sum to one")
        cumulative = 0.0
        boundaries = []
        for fraction in fractions:
            cumulative += fraction
            boundaries.append(round(self.max_episode_length * cumulative))
        boundaries[-1] = self.max_episode_length
        self.phase_boundaries = tuple(boundaries)

    def _allocate_trajectory_buffers(self) -> None:
        self.trajectory_joint_q = None
        self.trajectory_object_state = None
        self.trajectory_actions = None
        self.trajectory_phase = None
        self.completed_trajectory_joint_q = None
        self.completed_trajectory_object_state = None
        self.completed_trajectory_actions = None
        self.completed_trajectory_phase = None
        if not self.cfg.record_trajectory:
            return
        shape = (self.num_envs, self.max_episode_length)
        self.trajectory_joint_q = torch.zeros((*shape, self.cfg.action_space), device=self.device)
        self.trajectory_object_state = torch.zeros((*shape, 13), device=self.device)
        self.trajectory_actions = torch.zeros((*shape, self.cfg.action_space), device=self.device)
        self.trajectory_phase = torch.zeros(shape, dtype=torch.int8, device=self.device)
        self.completed_trajectory_joint_q = torch.zeros_like(self.trajectory_joint_q)
        self.completed_trajectory_object_state = torch.zeros_like(self.trajectory_object_state)
        self.completed_trajectory_actions = torch.zeros_like(self.trajectory_actions)
        self.completed_trajectory_phase = torch.zeros_like(self.trajectory_phase)

    def _setup_scene(self) -> None:
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.robot_usd,
                activate_contact_sensors=True,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=self.cfg.contact_offset_m,
                    rest_offset=self.cfg.rest_offset_m,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=bool(self.cfg.robot_self_collisions),
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["(left|right)_j[1-7]"],
                    stiffness=float(self.cfg.arm_stiffness),
                    damping=float(self.cfg.arm_damping),
                ),
                "hands": ImplicitActuatorCfg(
                    joint_names_expr=["(left|right)_hand_.*"],
                    stiffness=float(self.cfg.hand_stiffness),
                    damping=float(self.cfg.hand_damping),
                ),
            },
        )
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.object_usd,
                activate_contact_sensors=True,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=self.cfg.contact_offset_m,
                    rest_offset=self.cfg.rest_offset_m,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=float(
                        self.cfg.object_max_depenetration_velocity
                    ),
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.object_mass_kg),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    0.5,
                    0.0,
                    self.cfg.table_top_z_m + 0.5 * self.cfg.object_extents_m[2],
                )
            ),
        )
        layout = contact_filter_layout("/World/envs/env_.*/Robot")
        sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Object",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=[row[2] for row in layout],
            track_contact_points=True,
            track_friction_forces=False,
            # 256 x 1 body x 64 envs is 16384, and PhysX silently truncates
            # past that: "Incomplete contact data is reported in
            # GpuRigidContactView::getContactData".  Holding the object still
            # during grasp acquisition keeps the hands pressed against it
            # instead of pushing it away, so contact points accumulate and
            # overrun the budget -- and the readings that get dropped are
            # exactly the ones the stage-2 contact criteria are computed from.
            max_contact_data_count_per_prim=1024,
        )
        self.robot = Articulation(robot_cfg)
        self.object = RigidObject(object_cfg)
        self.object_contacts = ContactSensor(sensor_cfg)
        friction_cfg = RigidBodyMaterialCfg(
            static_friction=self.cfg.contact_friction,
            dynamic_friction=self.cfg.contact_friction,
            restitution=0.0,
        )
        friction_path = "/World/XHandEmbeddedPhysicsMaterial"
        friction_cfg.func(friction_path, friction_cfg)
        stage = get_current_stage()
        bind_physics_material("/World/envs/env_0/Robot", friction_path, stage=stage)
        bind_physics_material("/World/envs/env_0/Object", friction_path, stage=stage)
        table_thickness_m = 0.04
        table_cfg = sim_utils.CuboidCfg(
            size=(100.0, 100.0, table_thickness_m),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.0
            ),
        )
        table_cfg.func(
            "/World/TableTop",
            table_cfg,
            translation=(0.0, 0.0, self.cfg.table_top_z_m - 0.5 * table_thickness_m),
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["object_contacts"] = self.object_contacts
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light.func("/World/Light", light)

    def _phase_code(self, steps: torch.Tensor | None = None) -> torch.Tensor:
        steps = self.episode_length_buf if steps is None else steps
        b0, b1, b2, b3, b4, b5 = self.phase_boundaries
        code = torch.full_like(steps, PHASE_RIGHT_ONLY)
        code = torch.where(steps < b4, PHASE_DISTURBANCE_FIRST + ((steps - b3).clamp_min(0) * 12 // max(b4 - b3, 1)).clamp_max(11), code)
        code = torch.where(steps < b3, PHASE_HOLD, code)
        code = torch.where(steps < b2, PHASE_LIFT, code)
        code = torch.where(steps < b1, PHASE_CLOSE, code)
        code = torch.where(steps < b0, PHASE_APPROACH, code)
        ablation_code = PHASE_LEFT_ONLY + ((steps - b4).clamp_min(0) * 2 // max(b5 - b4, 1)).clamp_max(1)
        code = torch.where(steps >= b4, ablation_code, code)
        return code.long()

    def _coarse_phase(self, code: torch.Tensor) -> torch.Tensor:
        coarse = torch.empty_like(code)
        coarse[code == PHASE_APPROACH] = 0
        coarse[code == PHASE_CLOSE] = 1
        coarse[code == PHASE_LIFT] = 2
        coarse[code == PHASE_HOLD] = 3
        coarse[(code >= PHASE_DISTURBANCE_FIRST) & (code <= PHASE_DISTURBANCE_LAST)] = 4
        coarse[code >= PHASE_LEFT_ONLY] = 5
        return coarse

    def _base_target(self, code: torch.Tensor) -> torch.Tensor:
        steps = self.episode_length_buf.float()
        b0, b1, b2, _, b4, b5 = self.phase_boundaries
        close_alpha = ((steps - b0) / max(b1 - b0, 1)).clamp(0.0, 1.0).unsqueeze(-1)
        lift_alpha = ((steps - b1) / max(b2 - b1, 1)).clamp(0.0, 1.0).unsqueeze(-1)
        arm_fraction = float(self.cfg.arm_approach_fraction_of_close)
        if arm_fraction > 0.0:
            finger_close_end = float(
                self.cfg.finger_close_end_fraction_of_close
            )
            arm_alpha = (close_alpha / arm_fraction).clamp(0.0, 1.0)
            hand_alpha = (
                (close_alpha - arm_fraction)
                / max(finger_close_end - arm_fraction, 1.0e-8)
            ).clamp(0.0, 1.0)
            joint_alpha = hand_alpha.expand(-1, self.cfg.action_space).clone()
            joint_alpha[:, self.arm_ids] = arm_alpha
            close = self.nominal_pregrasp + joint_alpha * (
                self.nominal_grasp - self.nominal_pregrasp
            )
        else:
            close = self.nominal_pregrasp + close_alpha * (
                self.nominal_grasp - self.nominal_pregrasp
            )
        target = close + lift_alpha * (self.nominal_lift - self.nominal_grasp)
        # Once the hold snapshot exists, the learned hold posture is the
        # reference for all disturbance phases.  Falling back to nominal_lift
        # here silently erased the policy's corrective wrist/finger offsets on
        # every wrench transition and made a stable hold difficult to learn.
        disturbance_or_later = code >= PHASE_DISTURBANCE_FIRST
        if disturbance_or_later.any():
            target[disturbance_or_later] = self.hold_robot_q[disturbance_or_later]
        left_only = code == PHASE_LEFT_ONLY
        right_only = code == PHASE_RIGHT_ONLY
        if left_only.any():
            phase_length = max((b5 - b4) // 2, 1)
            alpha = ((steps[left_only] - b4) / max(phase_length // 2, 1)).clamp(0.0, 1.0).unsqueeze(-1)
            ids = torch.nonzero(left_only, as_tuple=False).squeeze(-1)
            target[ids[:, None], self.right_ids] = (
                self.hold_robot_q[ids[:, None], self.right_ids]
                + alpha * (self.nominal_pregrasp[ids[:, None], self.right_ids] - self.hold_robot_q[ids[:, None], self.right_ids])
            )
        if right_only.any():
            phase_start = b4 + (b5 - b4) // 2
            phase_length = max(b5 - phase_start, 1)
            alpha = ((steps[right_only] - phase_start) / max(phase_length // 2, 1)).clamp(0.0, 1.0).unsqueeze(-1)
            ids = torch.nonzero(right_only, as_tuple=False).squeeze(-1)
            target[ids[:, None], self.left_ids] = (
                self.hold_robot_q[ids[:, None], self.left_ids]
                + alpha * (self.nominal_pregrasp[ids[:, None], self.left_ids] - self.hold_robot_q[ids[:, None], self.left_ids])
            )
        return target

    def _restore_hold(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        valid = self.hold_state_valid[env_ids]
        env_ids = env_ids[valid]
        if env_ids.numel() == 0:
            return
        self.object.write_root_state_to_sim(self.hold_object_state[env_ids], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.hold_robot_q[env_ids], torch.zeros_like(self.hold_robot_q[env_ids]), env_ids=env_ids
        )
        self.robot.set_joint_position_target(self.hold_robot_q[env_ids], env_ids=env_ids)

    def _configure_wrench(self, code: torch.Tensor, transitions: torch.Tensor) -> None:
        transition_ids = torch.nonzero(transitions, as_tuple=False).squeeze(-1)
        if transition_ids.numel() == 0:
            return
        zero = torch.zeros((len(transition_ids), 1, 3), device=self.device)
        self.object.permanent_wrench_composer.set_forces_and_torques(
            zero, zero, env_ids=transition_ids, is_global=True
        )
        self.current_wrench[transition_ids] = 0.0
        for challenge in range(12):
            env_ids = torch.nonzero(
                transitions & (code == PHASE_DISTURBANCE_FIRST + challenge),
                as_tuple=False,
            ).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            axis = (challenge // 2) % 3
            sign = -1.0 if challenge % 2 == 0 else 1.0
            force = torch.zeros((len(env_ids), 1, 3), device=self.device)
            torque = torch.zeros_like(force)
            if challenge < 6:
                force[:, 0, axis] = sign * self.cfg.disturbance_force_n
                self.current_wrench[env_ids, axis] = force[:, 0, axis]
            else:
                torque[:, 0, axis] = sign * self.cfg.disturbance_torque_nm
                self.current_wrench[env_ids, 3 + axis] = torque[:, 0, axis]
            self.object.permanent_wrench_composer.set_forces_and_torques(
                force, torque, env_ids=env_ids, is_global=True
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = torch.clamp(actions, -1.0, 1.0)
        code = self._phase_code()
        challenge = code >= PHASE_DISTURBANCE_FIRST
        phase_transitions = code != self.previous_phase
        residual_active = code >= self.cfg.residual_activation_phase
        close_activation = float(self.cfg.residual_activation_close_fraction)
        if close_activation > 0.0:
            close_start, close_end = self.phase_boundaries[:2]
            activation_step = close_start + max(
                0, int(round(close_activation * max(close_end - close_start, 1)))
            )
            residual_active &= self.episode_length_buf >= activation_step
        residual_end = float(
            self.cfg.residual_integration_end_fraction_of_hold
        )
        if residual_end < 1.0:
            _, _, hold_start, hold_end, _, _ = self.phase_boundaries
            integration_end = hold_start + max(
                1, int(round(residual_end * max(hold_end - hold_start, 1)))
            )
            residual_active &= self.episode_length_buf < integration_end
        residual_active = residual_active.float().unsqueeze(-1)
        self.integrated_residual += (
            self.cfg.residual_integration * self.action_scale * self.actions * residual_active
        )
        self.integrated_residual = torch.clamp(
            self.integrated_residual,
            -self.cfg.residual_limit_rad,
            self.cfg.residual_limit_rad,
        )
        challenge_transitions = challenge & phase_transitions
        self._restore_hold(torch.nonzero(challenge_transitions, as_tuple=False).squeeze(-1))
        target = (
            self._base_target(code)
            + self.cfg.instantaneous_residual_weight
            * self.action_scale
            * self.actions
            * residual_active
            + self.integrated_residual
        )
        for phase, side_ids in ((PHASE_LEFT_ONLY, self.right_ids), (PHASE_RIGHT_ONLY, self.left_ids)):
            ids = torch.nonzero(code == phase, as_tuple=False).squeeze(-1)
            if ids.numel():
                base = self._base_target(code)
                target[ids[:, None], side_ids] = base[ids[:, None], side_ids]
        limits = self.robot.data.soft_joint_pos_limits
        self.targets = torch.clamp(target, limits[..., 0], limits[..., 1])
        self._configure_wrench(code, phase_transitions)
        self.previous_phase.copy_(code)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.targets)

    def _sensor_force(self, side: str) -> torch.Tensor:
        matrix = self.object_contacts.data.force_matrix_w
        if matrix is None:
            raise RuntimeError("object-filtered force matrix is unavailable")
        expected = 2 * len(side_filter_indices("left"))
        if matrix.ndim != 4 or matrix.shape[2] != expected:
            raise RuntimeError(f"unexpected contact matrix shape: {tuple(matrix.shape)}")
        values = matrix[:, :, side_filter_indices(side), :]
        return torch.linalg.vector_norm(values, dim=-1).reshape(self.num_envs, -1).sum(dim=-1)

    def _contact_count(self, side: str) -> torch.Tensor:
        matrix = self.object_contacts.data.force_matrix_w
        values = matrix[:, :, side_filter_indices(side), :]
        return (torch.linalg.vector_norm(values, dim=-1) > 0.02).reshape(self.num_envs, -1).sum(dim=-1).float()

    @staticmethod
    def _quat_apply_inverse_wxyz(
        quaternion: torch.Tensor, vectors: torch.Tensor
    ) -> torch.Tensor:
        """Rotate world-frame vectors into a WXYZ quaternion's local frame."""

        xyz = quaternion[..., 1:].unsqueeze(1)
        scalar = quaternion[..., :1].unsqueeze(1)
        first_cross = torch.linalg.cross(xyz.expand_as(vectors), vectors, dim=-1)
        return vectors - 2.0 * scalar * first_cross + 2.0 * torch.linalg.cross(
            xyz.expand_as(vectors), first_cross, dim=-1
        )

    def _hand_object_distance(self) -> torch.Tensor:
        """Minimum hand-link distance to the locked object's outer surface.

        The previous reachability shaping used distance to the object *center*.
        That target is structurally wrong for the locked large sphere, parcels,
        and 200% pyramid: a physically correct contact is already roughly one
        half-extent away from the center, while maximizing the old reward pulls
        fingers through the mesh.  Use a cheap analytic surface proxy for PPO
        shaping only.  The immutable PhysX penetration and real contact gates
        still decide whether a trajectory is accepted.
        """

        body_pos = self.robot.data.body_pos_w
        object_center = self.object.data.root_pos_w.unsqueeze(1)
        object_quaternion = self.object.data.root_quat_w
        half_extents = torch.tensor(
            self.cfg.object_extents_m,
            dtype=body_pos.dtype,
            device=self.device,
        ).mul(0.5)
        distances = []
        for side in ("left", "right"):
            positions = body_pos[:, self.hand_body_ids[side], :]
            local = self._quat_apply_inverse_wxyz(
                object_quaternion, positions - object_center
            )
            if self.cfg.object_key in ("sphere", "sphere_small"):
                # The locked sphere assets are mildly anisotropically scaled.
                # Radial distance in normalized ellipsoid coordinates gives a
                # stable surface proxy without a mesh query in every PPO step.
                normalized_radius = torch.linalg.vector_norm(
                    local / half_extents.clamp_min(1.0e-6), dim=-1
                )
                surface_distance = torch.abs(normalized_radius - 1.0) * float(
                    min(half_extents.tolist())
                )
            else:
                # Unsigned oriented-box SDF.  It is conservative for the
                # pyramid's sloped sides, but still targets the outer envelope
                # rather than the unreachable center; strict mesh/PhysX gates
                # reject any proxy-induced penetration.
                delta = torch.abs(local) - half_extents
                outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
                inside = torch.clamp(delta.max(dim=-1).values, max=0.0)
                surface_distance = torch.abs(outside + inside)
            distances.append(surface_distance.min(dim=-1).values)
        return torch.stack(distances, dim=-1)

    def _palm_center_alignment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bilateral palm +X alignment with the object center.

        XHand's anatomical palm normal is the local +X axis of each
        ``*_hand_ee_link``.  The score is deliberately symmetric: it combines
        the mean and the weaker hand so one well-oriented hand cannot hide a
        badly-oriented partner.  This is dense shaping only; contact and lift
        gates remain unchanged.
        """

        body_names = list(self.robot.body_names)
        body_ids = [body_names.index(f"{side}_hand_ee_link") for side in ("left", "right")]
        palm_positions = self.robot.data.body_pos_w[:, body_ids, :]
        palm_rotations = _quat_to_matrix_wxyz(self.robot.data.body_quat_w[:, body_ids, :])
        palm_axes_world = palm_rotations[..., :, 0]
        direction = self.object.data.root_pos_w[:, None, :] - palm_positions
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        cosine = torch.sum(palm_axes_world * direction, dim=-1).clamp(-1.0, 1.0)
        alignment01 = 0.5 * (cosine + 1.0)
        mean_alignment = alignment01.mean(dim=-1)
        minimum_alignment = alignment01.min(dim=-1).values
        score = 0.5 * (mean_alignment + minimum_alignment)
        return score, cosine[:, 0], cosine[:, 1]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel
        limits = self.robot.data.soft_joint_pos_limits
        q_mid = 0.5 * (limits[..., 0] + limits[..., 1])
        q_half = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-4)
        code = self._phase_code()
        coarse = torch.nn.functional.one_hot(self._coarse_phase(code), num_classes=6).float()
        progress = (self.episode_length_buf.float() / max(self.max_episode_length - 1, 1)).unsqueeze(-1)
        challenge = ((code - PHASE_DISTURBANCE_FIRST).clamp(0, 13).float() / 13.0).unsqueeze(-1)
        continuity = torch.stack(
            (
                self.left_contact_steps / self.contact_steps.clamp_min(1.0),
                self.right_contact_steps / self.contact_steps.clamp_min(1.0),
            ),
            dim=-1,
        )
        wrench_scale = torch.tensor(
            [self.cfg.disturbance_force_n] * 3 + [self.cfg.disturbance_torque_nm] * 3,
            device=self.device,
        ).clamp_min(1.0e-8)
        observation = torch.cat(
            (
                (q - q_mid) / q_half,
                0.05 * qd,
                self.object.data.root_pos_w - self.initial_object_pos - self.scene.env_origins,
                self.object.data.root_quat_w,
                0.1 * self.object.data.root_vel_w,
                torch.stack((self._sensor_force("left"), self._sensor_force("right")), dim=-1) / 20.0,
                coarse,
                progress,
                challenge,
                self.current_wrench / wrench_scale,
                continuity,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _is_phase_end(self, code: torch.Tensor) -> torch.Tensor:
        # DirectRLEnv increments episode_length_buf before task hooks run. The
        # buffer therefore identifies the next action, while ``code`` describes
        # the action whose physics was just simulated.
        next_steps = torch.clamp(self.episode_length_buf, max=self.max_episode_length - 1)
        next_code = self._phase_code(next_steps)
        return (next_code != code) | (self.episode_length_buf >= self.max_episode_length - 1)

    def _executed_phase_code(self) -> torch.Tensor:
        executed_steps = torch.clamp(self.episode_length_buf - 1, min=0)
        return self._phase_code(executed_steps)

    def _raw_contacts(
        self, env_ids: torch.Tensor, *, capture_records: set[int] | None = None
    ) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
        penetration = torch.zeros(len(env_ids), device=self.device)
        records: dict[int, dict[str, Any]] = {}
        if env_ids.numel() == 0:
            return penetration, records
        capture_records = capture_records or set()
        try:
            forces, points, normals, separations, counts, starts = (
                self.object_contacts.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
            )
            counts = counts.view(self.num_envs, -1)
            starts = starts.view(self.num_envs, -1)
            pair_forces = self.object_contacts.data.force_matrix_w[:, 0]
            layout = contact_filter_layout("/World/Robot")
            for local_index, env_id_tensor in enumerate(env_ids):
                env_id = int(env_id_tensor.item())
                hands = {"left": [], "right": []}
                maximum = 0.0
                for filter_index, (side, body_name, _) in enumerate(layout):
                    count = int(counts[env_id, filter_index].item())
                    start = int(starts[env_id, filter_index].item())
                    if count <= 0:
                        continue
                    group_separation = separations[start : start + count]
                    maximum = max(maximum, float(torch.clamp(-group_separation, min=0.0).max().item()))
                    if env_id not in capture_records:
                        continue
                    raw_force = forces[start : start + count] * normals[start : start + count]
                    pair_force = pair_forces[env_id, filter_index]
                    if torch.dot(raw_force.sum(dim=0), pair_force) < 0.0:
                        raw_force = -raw_force
                    for patch in range(count):
                        point = points[start + patch]
                        force = raw_force[patch]
                        if torch.isfinite(point).all() and torch.isfinite(force).all():
                            hands[side].append(
                                {
                                    "body_name": body_name,
                                    "position_world_m": point.detach().cpu().tolist(),
                                    "normal_force_on_object_world_N": force.detach().cpu().tolist(),
                                    "separation_m": float(group_separation[patch].item()),
                                }
                            )
                penetration[local_index] = maximum
                if env_id in capture_records:
                    records[env_id] = {"stage": "hold", "hands": hands}
        except Exception:
            return torch.full_like(penetration, float("inf")), records
        return penetration, records

    def _evaluate_hold_force_closure(
        self, env_ids: torch.Tensor, *, finalize: bool = True
    ) -> None:
        if env_ids.numel() == 0:
            return
        continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
        lift = self.lift_object_position[:, 2] - (self.initial_object_pos[:, 2] + self.scene.env_origins[:, 2])
        arm_delta = torch.linalg.vector_norm(
            self.lift_q[:, self.arm_ids] - self.closure_q[:, self.arm_ids], dim=-1
        )
        eligible = env_ids[
            (continuity_left[env_ids] >= self.cfg.contact_presence_fraction_min)
            & (continuity_right[env_ids] >= self.cfg.contact_presence_fraction_min)
            & (lift[env_ids] >= self.cfg.lift_min_m)
            & (arm_delta[env_ids] > 1.0e-4)
            & (self.max_hold_drift[env_ids] <= self.cfg.gravity_drift_max_m)
        ]
        evaluation_ids = (
            env_ids if self.cfg.force_closure_shape_all_hold_contacts else eligible
        )
        penetration, records = self._raw_contacts(
            env_ids,
            capture_records={int(i.item()) for i in evaluation_ids},
        )
        measured = torch.isfinite(penetration)
        self.penetration_measured[env_ids] |= measured
        self.current_penetration[env_ids] = torch.where(
            measured, penetration, self.current_penetration[env_ids]
        )
        self.maximum_penetration[env_ids] = torch.maximum(
            self.maximum_penetration[env_ids], torch.where(measured, penetration, self.maximum_penetration[env_ids])
        )
        characteristic_length = 0.5 * sum(float(value) ** 2 for value in self.cfg.object_extents_m) ** 0.5
        for env_id_tensor in evaluation_ids:
            env_id = int(env_id_tensor.item())
            record = records.get(env_id)
            if record is None:
                continue
            center = self.object.data.root_pos_w[env_id].detach().cpu().tolist()
            report = evaluate_force_closure(
                [record],
                object_center_world_m=center,
                friction=self.cfg.contact_friction,
                characteristic_length_m=characteristic_length,
                residual_max=self.cfg.force_closure_residual_max,
                epsilon_min=self.cfg.force_closure_epsilon_min,
            )
            # During hold refinement, expose the dense near-miss quality on
            # every sampled hold state, but keep the boolean formal gate tied
            # to the terminal hold audit.  This prevents a transient contact
            # from becoming an accepted sample while giving PPO a usable
            # force-closure gradient before the single end-of-hold test.
            if finalize:
                self.force_closure_reports[env_id] = report
                self.force_closure_evaluated[env_id] = True
                self.force_closure_pass[env_id] = bool(report["force_closure_pass"])
            residual = float(report["max_signed_unit_wrench_residual"])
            residual_score = (
                1.0 / (1.0 + residual / max(self.cfg.force_closure_residual_max, 1.0e-8))
                if math.isfinite(residual)
                else 0.0
            )
            contact_score = min(float(report["contact_count"]) / 4.0, 1.0)
            bilateral_score = float(
                all(int(report["contact_counts_by_side"].get(side, 0)) > 0 for side in ("left", "right"))
            )
            rank_score = min(float(report["wrench_rank"]) / 6.0, 1.0)
            origin_score = float(report["origin_strictly_inside_wrench_hull"])
            epsilon_score = min(
                float(report["ferrari_canny_epsilon_normalized"])
                / max(self.cfg.force_closure_epsilon_min, 1.0e-12),
                1.0,
            )
            strict_quality = (
                contact_score
                * bilateral_score
                * rank_score
                * (0.5 * residual_score + 0.25 * origin_score + 0.25 * epsilon_score)
            )
            # The boolean gate below remains the exact Ferrari--Canny/
            # signed-wrench contract.  For PPO shaping, however, a product is
            # too sparse: rank=5 or an origin just outside the hull makes the
            # entire quality zero and provides no direction toward closure.
            # Blend the strict product with a bounded additive near-miss score;
            # this changes no acceptance decision and is only used by future
            # training retries.
            dense_quality = (
                0.20 * contact_score
                + 0.20 * bilateral_score
                + 0.20 * rank_score
                + 0.20 * residual_score
                + 0.10 * origin_score
                + 0.10 * epsilon_score
            )
            self.force_closure_quality[env_id] = (
                0.35 * strict_quality + 0.65 * dense_quality
            )

    def _update_metrics(self, code: torch.Tensor) -> None:
        end = self._is_phase_end(code)
        left_force = self._sensor_force("left")
        right_force = self._sensor_force("right")
        lift_hold = (code == PHASE_LIFT) | (code == PHASE_HOLD)
        self.contact_steps += lift_hold.float()
        self.left_contact_steps += (lift_hold & (left_force > 0.02)).float()
        self.right_contact_steps += (lift_hold & (right_force > 0.02)).float()
        q = self.robot.data.joint_pos
        ids = torch.nonzero(end & (code == PHASE_APPROACH), as_tuple=False).squeeze(-1)
        if ids.numel():
            self.pregrasp_q[ids] = q[ids]
        ids = torch.nonzero(end & (code == PHASE_CLOSE), as_tuple=False).squeeze(-1)
        if ids.numel():
            self.closure_q[ids] = q[ids]
        ids = torch.nonzero(end & (code == PHASE_LIFT), as_tuple=False).squeeze(-1)
        if ids.numel():
            self.lift_q[ids] = q[ids]
            self.lift_object_position[ids] = self.object.data.root_pos_w[ids]
        hold = code == PHASE_HOLD
        if hold.any():
            drift = torch.linalg.vector_norm(
                self.object.data.root_pos_w - self.lift_object_position, dim=-1
            )
            self.max_hold_drift = torch.where(hold, torch.maximum(self.max_hold_drift, drift), self.max_hold_drift)
            # The old implementation evaluated force closure only once at
            # the hold boundary.  That made the 96-point shaping term almost
            # binary and left PPO with no signal while it was actually
            # adjusting the fingers/wrists.  Sample every four hold steps in
            # the new all-hold-contact mode.  The expensive visual-mesh audit
            # remains disabled; this is the embedded PhysX/contact-wrench
            # calculation already used by the RL environment.
            if self.cfg.force_closure_shape_all_hold_contacts:
                sample_ids = torch.nonzero(
                    hold & ((self.episode_length_buf % 4) == 0),
                    as_tuple=False,
                ).squeeze(-1)
                self._evaluate_hold_force_closure(sample_ids, finalize=False)
        ids = torch.nonzero(end & hold, as_tuple=False).squeeze(-1)
        if ids.numel():
            self.hold_object_state[ids] = self.object.data.root_state_w[ids]
            self.hold_robot_q[ids] = q[ids]
            self.hold_position[ids] = self.object.data.root_pos_w[ids]
            self.hold_quaternion[ids] = self.object.data.root_quat_w[ids]
            self.gravity_drift[ids] = torch.linalg.vector_norm(
                self.hold_position[ids] - self.lift_object_position[ids], dim=-1
            )
            self.hold_state_valid[ids] = True
            self._evaluate_hold_force_closure(ids, finalize=True)
        challenge_end = end & (code >= PHASE_DISTURBANCE_FIRST) & (code <= PHASE_DISTURBANCE_LAST)
        ids = torch.nonzero(challenge_end, as_tuple=False).squeeze(-1)
        if ids.numel():
            for challenge in range(12):
                selected = ids[code[ids] == PHASE_DISTURBANCE_FIRST + challenge]
                if selected.numel() == 0:
                    continue
                if challenge < 6:
                    self.translation_displacements[selected, challenge] = torch.linalg.vector_norm(
                        self.object.data.root_pos_w[selected] - self.hold_position[selected], dim=-1
                    )
                else:
                    self.rotation_displacements[selected, challenge - 6] = _quat_angle(
                        self.object.data.root_quat_w[selected], self.hold_quaternion[selected]
                    )
        for phase, column, active, inactive in (
            (PHASE_LEFT_ONLY, 0, left_force, right_force),
            (PHASE_RIGHT_ONLY, 1, right_force, left_force),
        ):
            _, _, _, _, b4, b5 = self.phase_boundaries
            midpoint = b4 + (b5 - b4) // 2
            phase_start = b4 if phase == PHASE_LEFT_ONLY else midpoint
            phase_stop = midpoint if phase == PHASE_LEFT_ONLY else b5
            stable_start = phase_start + (phase_stop - phase_start) // 2
            mask = (code == phase) & (self.episode_length_buf >= stable_start)
            self.ablation_steps[:, column] += mask.float()
            self.ablation_active_contact_steps[:, column] += (mask & (active > 0.02)).float()
            self.ablation_inactive_contact_steps[:, column] += (mask & (inactive > 0.02)).float()
            selected = torch.nonzero(end & mask, as_tuple=False).squeeze(-1)
            if selected.numel():
                height = self.object.data.root_pos_w[selected, 2] - (
                    self.initial_object_pos[selected, 2] + self.scene.env_origins[selected, 2]
                )
                active_fraction = self.ablation_active_contact_steps[selected, column] / self.ablation_steps[selected, column].clamp_min(1.0)
                inactive_fraction = self.ablation_inactive_contact_steps[selected, column] / self.ablation_steps[selected, column].clamp_min(1.0)
                self.single_hand_success[selected, column] = (
                    (height >= self.cfg.lift_min_m)
                    & (active_fraction >= self.cfg.contact_presence_fraction_min)
                    & (inactive_fraction <= self.cfg.inactive_hand_contact_fraction_max)
                )
        boundary_ids = torch.nonzero(end, as_tuple=False).squeeze(-1)
        if boundary_ids.numel():
            penetration, _ = self._raw_contacts(boundary_ids)
            measured = torch.isfinite(penetration)
            self.penetration_measured[boundary_ids] |= measured
            self.current_penetration[boundary_ids] = torch.where(
                measured, penetration, self.current_penetration[boundary_ids]
            )
            self.maximum_penetration[boundary_ids] = torch.maximum(
                self.maximum_penetration[boundary_ids],
                torch.where(measured, penetration, self.maximum_penetration[boundary_ids]),
            )

    def _physics_gate_components(self) -> dict[str, torch.Tensor]:
        """Return tensorized counterparts of every embedded physics gate."""

        continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
        inactive_fraction = self.ablation_inactive_contact_steps / self.ablation_steps.clamp_min(1.0)
        lift = self.lift_object_position[:, 2] - (
            self.initial_object_pos[:, 2] + self.scene.env_origins[:, 2]
        )
        arm_delta = torch.linalg.vector_norm(
            self.lift_q[:, self.arm_ids] - self.closure_q[:, self.arm_ids], dim=-1
        )
        finite = (
            torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
            & torch.isfinite(self.object.data.root_state_w).all(dim=-1)
        )
        return {
            "embedded_isaaclab_episode": self.last_audited_episode_step
            == self.episode_length_buf,
            "finite_state": finite,
            "bilateral_contact_continuity": (
                (continuity_left >= self.cfg.contact_presence_fraction_min)
                & (continuity_right >= self.cfg.contact_presence_fraction_min)
            ),
            "arm_joint_lift": arm_delta > 1.0e-4,
            "lift_height": lift >= self.cfg.lift_min_m,
            "gravity_hold": (
                (self.gravity_drift <= self.cfg.gravity_drift_max_m)
                & (self.max_hold_drift <= self.cfg.gravity_drift_max_m)
            ),
            "translation_disturbance": (
                torch.isfinite(self.translation_displacements).all(dim=-1)
                & (
                    self.translation_displacements.max(dim=-1).values
                    <= self.cfg.translation_disturbance_max_m
                )
            ),
            "rotation_disturbance": (
                torch.isfinite(self.rotation_displacements).all(dim=-1)
                & (
                    self.rotation_displacements.max(dim=-1).values
                    <= self.cfg.rotation_disturbance_max_rad
                )
            ),
            "physx_penetration": (
                self.penetration_measured
                & (self.maximum_penetration <= self.cfg.physx_penetration_max_m)
            ),
            "formal_force_closure": self.force_closure_pass,
            "single_hand_ablations": (
                (inactive_fraction[:, 0] <= self.cfg.inactive_hand_contact_fraction_max)
                & (inactive_fraction[:, 1] <= self.cfg.inactive_hand_contact_fraction_max)
                & ~self.single_hand_success[:, 0]
                & ~self.single_hand_success[:, 1]
            ),
        }

    def _physics_gate_mask(self) -> torch.Tensor:
        components = self._physics_gate_components()
        return torch.stack(tuple(components.values()), dim=0).all(dim=0)

    def _hard_gate_frontier_prefixes(
        self, code: torch.Tensor, components: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the phase-safe, same-trajectory strict gate prefixes.

        A component is rewarded only after the phase that defines it has
        completed.  Prefix products make later credit conditional on every
        earlier strict gate, so independently good contact, lift, clearance,
        or force closure cannot be combined across different trajectories.

        Keeping the individual masks lets TensorBoard report the exact first
        failed conjunction.  This is materially different from the existing
        per-gate fractions, which may each be non-zero in different envs.
        """

        post_lift = code >= PHASE_HOLD
        post_hold = code >= PHASE_DISTURBANCE_FIRST
        post_disturbance = code >= PHASE_LEFT_ONLY
        final_step = self.episode_length_buf >= self.max_episode_length - 1

        contact = components["bilateral_contact_continuity"]
        lifted = contact & components["arm_joint_lift"] & components["lift_height"]
        clear = lifted & components["physx_penetration"]
        stable_closure = (
            clear
            & components["gravity_hold"]
            & components["formal_force_closure"]
        )
        disturbed = (
            stable_closure
            & components["translation_disturbance"]
            & components["rotation_disturbance"]
        )
        ablated = disturbed & components["single_hand_ablations"]

        return {
            "contact": post_lift & contact,
            "lifted": post_hold & lifted,
            "clear": post_hold & clear,
            "stable_force_closure": post_hold & stable_closure,
            "disturbed": post_disturbance & disturbed,
            "ablated": final_step & ablated,
        }

    def _hard_gate_frontier_score(
        self, code: torch.Tensor, components: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return monotonic strict-prefix credit for formal PPO."""

        prefixes = self._hard_gate_frontier_prefixes(code, components)
        return torch.stack(
            tuple(mask.float() for mask in prefixes.values()), dim=0
        ).sum(dim=0)

    def _capture_training_prefix_snapshot(self, env_id: int, level: int) -> None:
        """Keep one exact Isaac state for the strongest rollout prefix."""

        if level <= int(self.training_prefix_snapshot_level):
            return

        def cpu(value: torch.Tensor) -> torch.Tensor:
            return value.detach().cpu().clone()

        snapshot: dict[str, Any] = {
            "schema": "xhand_rl_exact_prefix_state_v1",
            "object": self.cfg.object_key,
            "env_id": int(env_id),
            "env_origin": cpu(self.scene.env_origins[env_id]),
            "prefix_level": int(level),
            "phase_code": int(self._executed_phase_code()[env_id].item()),
            "episode_length": int(self.episode_length_buf[env_id].item()),
            "phase_boundaries": [int(value) for value in self.phase_boundaries],
            "robot_joint_pos": cpu(self.robot.data.joint_pos[env_id]),
            "robot_joint_vel": cpu(self.robot.data.joint_vel[env_id]),
            "object_root_state": cpu(self.object.data.root_state_w[env_id]),
            "initial_object_pos": cpu(self.initial_object_pos[env_id]),
            "last_start_pose": cpu(self.last_start_pose[env_id]),
            "actions": cpu(self.actions[env_id]),
            "previous_actions": cpu(self.previous_actions[env_id]),
            "integrated_residual": cpu(self.integrated_residual[env_id]),
            "pregrasp_q": cpu(self.pregrasp_q[env_id]),
            "closure_q": cpu(self.closure_q[env_id]),
            "lift_q": cpu(self.lift_q[env_id]),
            "hold_robot_q": cpu(self.hold_robot_q[env_id]),
            "hold_object_state": cpu(self.hold_object_state[env_id]),
            "hold_position": cpu(self.hold_position[env_id]),
            "hold_quaternion": cpu(self.hold_quaternion[env_id]),
            "lift_object_position": cpu(self.lift_object_position[env_id]),
            "hold_state_valid": bool(self.hold_state_valid[env_id].item()),
            "current_wrench": cpu(self.current_wrench[env_id]),
            "contact_steps": float(self.contact_steps[env_id].item()),
            "left_contact_steps": float(self.left_contact_steps[env_id].item()),
            "right_contact_steps": float(self.right_contact_steps[env_id].item()),
            "max_hold_drift": float(self.max_hold_drift[env_id].item()),
            "gravity_drift": float(self.gravity_drift[env_id].item()),
            "translation_displacements": cpu(self.translation_displacements[env_id]),
            "rotation_displacements": cpu(self.rotation_displacements[env_id]),
            "current_penetration": float(self.current_penetration[env_id].item()),
            "maximum_penetration": float(self.maximum_penetration[env_id].item()),
            "penetration_measured": bool(self.penetration_measured[env_id].item()),
            "ablation_steps": cpu(self.ablation_steps[env_id]),
            "ablation_inactive_contact_steps": cpu(
                self.ablation_inactive_contact_steps[env_id]
            ),
            "ablation_active_contact_steps": cpu(
                self.ablation_active_contact_steps[env_id]
            ),
            "single_hand_success": cpu(self.single_hand_success[env_id]),
            "force_closure_pass": bool(self.force_closure_pass[env_id].item()),
            "force_closure_evaluated": bool(
                self.force_closure_evaluated[env_id].item()
            ),
            "force_closure_quality": float(self.force_closure_quality[env_id].item()),
            "force_closure_report": copy.deepcopy(self.force_closure_reports[env_id]),
            "trajectory_joint_q": (
                cpu(self.trajectory_joint_q[env_id])
                if self.trajectory_joint_q is not None
                else None
            ),
            "trajectory_object_state": (
                cpu(self.trajectory_object_state[env_id])
                if self.trajectory_object_state is not None
                else None
            ),
            "trajectory_actions": (
                cpu(self.trajectory_actions[env_id])
                if self.trajectory_actions is not None
                else None
            ),
            "trajectory_phase": (
                cpu(self.trajectory_phase[env_id])
                if self.trajectory_phase is not None
                else None
            ),
        }
        self.training_prefix_snapshot = snapshot
        self.training_prefix_snapshot_level = int(level)

    def consume_training_prefix_statistics(self) -> dict[str, Any]:
        """Return and reset the strict-prefix statistics for one PPO rollout."""

        count = int(self.training_prefix_sample_count.item())
        score_sum = float(self.training_prefix_score_sum.item())
        result: dict[str, Any] = {
            "sample_count": count,
            "mean_score": score_sum / max(count, 1),
            "max_level": int(self.training_prefix_max_level.item()),
            "snapshot": self.training_prefix_snapshot,
        }
        self.training_prefix_score_sum.zero_()
        self.training_prefix_sample_count.zero_()
        self.training_prefix_max_level.zero_()
        self.training_prefix_max_level_env.zero_()
        self.training_prefix_snapshot = None
        self.training_prefix_snapshot_level = 0
        return result

    def restore_training_prefix_state(self, state_path: str | Path) -> None:
        """Restore an exact saved prefix state into every vectorized env.

        A policy checkpoint plus an RNG seed does not reproduce the contact
        state that generated a rare prefix.  This method restores that state
        and resumes at the missing gate: contact/lift/clear prefixes continue
        from the formal hold boundary, while stable force closure continues at
        disturbance.  No gate is marked as
        passed by the restore; the normal terminal contract remains the only
        acceptance authority.
        """

        payload = torch.load(
            Path(state_path).resolve(), map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict) or payload.get("schema") != "xhand_rl_exact_prefix_state_v1":
            raise RuntimeError("invalid exact prefix state schema")
        if payload.get("object") not in (None, "", self.cfg.object_key):
            raise RuntimeError("prefix state object does not match environment")
        level = int(payload.get("prefix_level", 0))
        if level <= 0 or level >= 6:
            raise RuntimeError("prefix state level must be in [1, 5]")
        b0, b1, b2, b3, b4, b5 = self.phase_boundaries
        # The strict prefix levels are audited at different phase boundaries:
        # level 1 (bilateral contact continuity) is defined in HOLD, while
        # levels 2--5 (lift/clear/force-closure/disturbance) are defined from
        # the first DISTURBANCE step onward.  Restoring a level-3/4 snapshot
        # to ``b2 + 1`` silently rewinds the episode into LIFT, so the nominal
        # target jumps backwards and the captured contact/lift basin collapses
        # on the first PPO rollout.  Resume at the earliest phase that still
        # makes the captured prefix predicates meaningful.
        if level <= 1:
            resume_step = b2 + 1
        else:
            resume_step = b3 + 1
        env_ids = torch.arange(self.num_envs, device=self.device)
        source_env_id = int(payload.get("env_id", 0))
        source_env_id = max(0, min(source_env_id, self.num_envs - 1))
        saved_origin = payload.get("env_origin")
        source_origin = (
            torch.as_tensor(saved_origin, device=self.device, dtype=self.scene.env_origins.dtype)
            if saved_origin is not None
            else self.scene.env_origins[source_env_id]
        )
        if tuple(source_origin.shape) != (3,):
            raise RuntimeError("prefix env_origin must have shape [3]")

        def tensor_value(name: str) -> torch.Tensor:
            value = payload.get(name)
            if value is None:
                raise RuntimeError(f"prefix state missing {name}")
            return torch.as_tensor(value, device=self.device)

        def replicate(name: str, target: torch.Tensor) -> None:
            value = tensor_value(name).to(dtype=target.dtype)
            expected = tuple(target.shape[1:])
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"prefix state {name} shape {tuple(value.shape)} != {expected}"
                )
            target.copy_(value.unsqueeze(0).expand_as(target))

        object_state = tensor_value("object_root_state").to(dtype=self.object.data.root_state_w.dtype)
        if tuple(object_state.shape) != (13,):
            raise RuntimeError("prefix object_root_state must have shape [13]")
        object_states = object_state.unsqueeze(0).repeat(self.num_envs, 1)
        object_states[:, :3] = (
            object_state[:3].unsqueeze(0) - source_origin.unsqueeze(0)
            + self.scene.env_origins
        )
        self.object.write_root_state_to_sim(object_states, env_ids=env_ids)

        joint_pos = tensor_value("robot_joint_pos").to(dtype=self.robot.data.joint_pos.dtype)
        joint_vel = tensor_value("robot_joint_vel").to(dtype=self.robot.data.joint_vel.dtype)
        if tuple(joint_pos.shape) != (self.cfg.action_space,) or tuple(joint_vel.shape) != (self.cfg.action_space,):
            raise RuntimeError("prefix robot state has an unexpected action-space shape")
        self.robot.write_joint_state_to_sim(
            joint_pos.unsqueeze(0).repeat(self.num_envs, 1),
            joint_vel.unsqueeze(0).repeat(self.num_envs, 1),
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            joint_pos.unsqueeze(0).repeat(self.num_envs, 1), env_ids=env_ids
        )

        for name, target in (
            ("initial_object_pos", self.initial_object_pos),
            ("last_start_pose", self.last_start_pose),
            ("actions", self.actions),
            ("previous_actions", self.previous_actions),
            ("integrated_residual", self.integrated_residual),
            ("pregrasp_q", self.pregrasp_q),
            ("closure_q", self.closure_q),
            ("lift_q", self.lift_q),
            ("hold_robot_q", self.hold_robot_q),
            ("hold_object_state", self.hold_object_state),
            ("hold_position", self.hold_position),
            ("hold_quaternion", self.hold_quaternion),
            ("lift_object_position", self.lift_object_position),
            ("current_wrench", self.current_wrench),
            ("translation_displacements", self.translation_displacements),
            ("rotation_displacements", self.rotation_displacements),
            ("ablation_steps", self.ablation_steps),
            ("ablation_inactive_contact_steps", self.ablation_inactive_contact_steps),
            ("ablation_active_contact_steps", self.ablation_active_contact_steps),
            ("single_hand_success", self.single_hand_success),
        ):
            replicate(name, target)

        # These values are scalar per environment in the live tensors.
        for name, target in (
            ("contact_steps", self.contact_steps),
            ("left_contact_steps", self.left_contact_steps),
            ("right_contact_steps", self.right_contact_steps),
            ("max_hold_drift", self.max_hold_drift),
            ("gravity_drift", self.gravity_drift),
            ("current_penetration", self.current_penetration),
            ("maximum_penetration", self.maximum_penetration),
            ("force_closure_quality", self.force_closure_quality),
        ):
            value = tensor_value(name).to(dtype=target.dtype)
            target.copy_(value.expand_as(target))
        for name, target in (
            ("hold_state_valid", self.hold_state_valid),
            ("penetration_measured", self.penetration_measured),
            ("force_closure_pass", self.force_closure_pass),
            ("force_closure_evaluated", self.force_closure_evaluated),
        ):
            value = torch.as_tensor(payload.get(name), device=self.device, dtype=target.dtype)
            target.copy_(value.expand_as(target))

        # World-frame buffers must be rebased from the source environment to
        # each destination environment.  All other saved positions are local.
        origin_delta = self.scene.env_origins - source_origin.unsqueeze(0)
        self.last_start_pose[:, :3] += origin_delta
        self.hold_object_state[:, :3] += origin_delta
        self.hold_position += origin_delta
        self.lift_object_position += origin_delta
        if self.trajectory_object_state is not None and payload.get("trajectory_object_state") is not None:
            trajectory_object = torch.as_tensor(
                payload["trajectory_object_state"],
                device=self.device,
                dtype=self.trajectory_object_state.dtype,
            )
            if tuple(trajectory_object.shape) != tuple(self.trajectory_object_state.shape[1:]):
                raise RuntimeError("prefix trajectory object state has an unexpected shape")
            self.trajectory_object_state.copy_(trajectory_object.unsqueeze(0).expand_as(self.trajectory_object_state))
            self.trajectory_object_state[:, :, :3] += origin_delta[:, None, :]
        for name, target in (
            ("trajectory_joint_q", self.trajectory_joint_q),
            ("trajectory_actions", self.trajectory_actions),
            ("trajectory_phase", self.trajectory_phase),
        ):
            if target is not None and payload.get(name) is not None:
                value = torch.as_tensor(payload[name], device=self.device, dtype=target.dtype)
                if tuple(value.shape) != tuple(target.shape[1:]):
                    raise RuntimeError(f"prefix {name} has an unexpected shape")
                target.copy_(value.unsqueeze(0).expand_as(target))

        self.episode_length_buf[:] = int(resume_step)
        self.previous_phase[:] = -1
        self.last_audited_episode_step[:] = int(resume_step - 1)
        self.force_closure_reports = [
            copy.deepcopy(payload.get("force_closure_report"))
            for _ in range(self.num_envs)
        ]
        # A level-3 state is intentionally resumed before a fresh hold audit;
        # never let the source boolean satisfy formal force closure by itself.
        if level <= 3:
            self.force_closure_pass[:] = False
            self.force_closure_evaluated[:] = False

    def _record_trajectory(self, code: torch.Tensor) -> None:
        if self.trajectory_joint_q is None:
            return
        indices = (self.episode_length_buf - 1).clamp(0, self.max_episode_length - 1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.trajectory_joint_q[env_ids, indices] = self.robot.data.joint_pos
        self.trajectory_object_state[env_ids, indices] = self.object.data.root_state_w
        self.trajectory_actions[env_ids, indices] = self.actions
        self.trajectory_phase[env_ids, indices] = code.to(torch.int8)

    def _audit_post_physics_step(self) -> torch.Tensor:
        """Audit exactly the state produced by the most recent physics step."""

        code = self._executed_phase_code()
        self._update_metrics(code)
        self._record_trajectory(code)
        self.last_audited_episode_step.copy_(self.episode_length_buf)
        return code

    def _get_rewards(self) -> torch.Tensor:
        code = self._executed_phase_code()
        height = self.object.data.root_pos_w[:, 2] - (
            self.initial_object_pos[:, 2] + self.scene.env_origins[:, 2]
        )
        continuity_left = self.left_contact_steps / self.contact_steps.clamp_min(1.0)
        continuity_right = self.right_contact_steps / self.contact_steps.clamp_min(1.0)
        inactive_fraction = self.ablation_inactive_contact_steps / self.ablation_steps.clamp_min(1.0)
        active_fraction = self.ablation_active_contact_steps / self.ablation_steps.clamp_min(1.0)
        ablation_phase = code >= PHASE_LEFT_ONLY
        ablation_column = torch.where(
            code == PHASE_LEFT_ONLY,
            torch.zeros_like(code),
            torch.ones_like(code),
        )
        selected_inactive_fraction = inactive_fraction.gather(1, ablation_column.unsqueeze(-1)).squeeze(-1)
        selected_active_fraction = active_fraction.gather(1, ablation_column.unsqueeze(-1)).squeeze(-1)
        _, _, _, _, b4, b5 = self.phase_boundaries
        midpoint = b4 + (b5 - b4) // 2
        left_stable_start = b4 + (midpoint - b4) // 2
        right_stable_start = midpoint + (b5 - midpoint) // 2
        ablation_stable_phase = (
            ((code == PHASE_LEFT_ONLY) & (self.episode_length_buf >= left_stable_start))
            | ((code == PHASE_RIGHT_ONLY) & (self.episode_length_buf >= right_stable_start))
        )
        challenge_translation = torch.linalg.vector_norm(
            self.object.data.root_pos_w - self.hold_position, dim=-1
        )
        challenge_rotation = _quat_angle(self.object.data.root_quat_w, self.hold_quaternion)
        disturbance = (code >= PHASE_DISTURBANCE_FIRST) & (code <= PHASE_DISTURBANCE_LAST)
        base_target = self._base_target(code)
        nominal_target_error = torch.mean(
            torch.square(self.robot.data.joint_pos - base_target), dim=-1
        )
        nominal_tracking_score = torch.exp(-nominal_target_error / 0.01)
        nominal_tracking_phase = (code == PHASE_APPROACH) | (code == PHASE_CLOSE)
        hand_object_distance = self._hand_object_distance()
        if float(self.cfg.palm_center_alignment_reward_weight) != 0.0:
            palm_alignment_score, left_palm_cosine, right_palm_cosine = (
                self._palm_center_alignment()
            )
        else:
            palm_alignment_score = torch.zeros_like(nominal_tracking_score)
            left_palm_cosine = torch.zeros_like(nominal_tracking_score)
            right_palm_cosine = torch.zeros_like(nominal_tracking_score)
        final_step = self.episode_length_buf >= self.max_episode_length - 1
        gate_components = self._physics_gate_components()
        hard_gate_prefixes = self._hard_gate_frontier_prefixes(
            code, gate_components
        )
        hard_gate_frontier_score = self._hard_gate_frontier_score(
            code, gate_components
        )
        disturbance_direction_progress = strict_disturbance_prefix_fraction(
            self.translation_displacements,
            self.rotation_displacements,
            translation_limit_m=self.cfg.translation_disturbance_max_m,
            rotation_limit_rad=self.cfg.rotation_disturbance_max_rad,
        ) * hard_gate_prefixes["stable_force_closure"].float()
        # PPO-only shaping for already-measured directions. Legacy retries use
        # a stable bimanual near-miss signal; future formal retries can require
        # the same-trajectory stable force-closure prefix instead.
        disturbance_direction_fraction = strict_disturbance_pass_fraction(
            self.translation_displacements,
            self.rotation_displacements,
            translation_limit_m=self.cfg.translation_disturbance_max_m,
            rotation_limit_rad=self.cfg.rotation_disturbance_max_rad,
        ) * (
            hard_gate_prefixes["stable_force_closure"]
            if self.cfg.disturbance_reward_requires_force_closure
            else (
                gate_components["bilateral_contact_continuity"]
                & gate_components["gravity_hold"]
                & gate_components["physx_penetration"]
            )
        ).float() * disturbance.float()
        hard_gate_shaping_score = (
            hard_gate_frontier_score + disturbance_direction_progress
        )
        # Preserve the strongest same-trajectory prefix before any timed-out
        # vector environments are reset.  The aggregate curves remain exactly
        # as before; this adds only replay evidence for a future retry.
        prefix_level_per_env = hard_gate_frontier_score.to(dtype=torch.long)
        improved = prefix_level_per_env > self.training_prefix_max_level_env
        improved_ids = torch.nonzero(improved, as_tuple=False).squeeze(-1)
        if improved_ids.numel():
            best_local = improved_ids[
                torch.argmax(prefix_level_per_env[improved_ids])
            ]
            best_level = int(prefix_level_per_env[best_local].item())
            self._capture_training_prefix_snapshot(int(best_local.item()), best_level)
            self.training_prefix_max_level_env[improved_ids] = prefix_level_per_env[
                improved_ids
            ]
        self.training_prefix_score_sum += hard_gate_frontier_score.sum()
        self.training_prefix_sample_count += hard_gate_frontier_score.numel()
        self.training_prefix_max_level.copy_(
            torch.maximum(
                self.training_prefix_max_level,
                hard_gate_frontier_score.max().to(dtype=torch.long),
            )
        )
        # Keep terminal acceptance polymorphic: the short curriculum
        # environment overrides _physics_gate_mask with its reduced contract,
        # while formal PPO uses all components above.
        terminal_success = final_step & self._physics_gate_mask()
        dropped = self.object.data.root_pos_w[:, 2] < self.cfg.table_top_z_m - 0.03
        reward_weights = EmbeddedRewardWeights(
            bilateral_contact=float(self.cfg.bilateral_contact_reward_weight),
            contact_continuity=float(self.cfg.contact_continuity_reward_weight),
            contact_diversity=float(self.cfg.contact_diversity_reward_weight),
            lift_height=float(self.cfg.lift_height_reward_weight),
            penetration=float(self.cfg.penetration_reward_weight),
            penetration_clear=float(self.cfg.penetration_clear_reward_weight),
            terminal_success=float(self.cfg.terminal_success_weight),
            approach_proximity=float(self.cfg.proximity_reward_weight),
            stable_lift=float(self.cfg.stable_lift_reward_weight),
            force_closure=float(self.cfg.force_closure_reward_weight),
            hard_gate_frontier=float(
                self.cfg.hard_gate_frontier_reward_weight
            ),
            disturbance_direction=float(
                self.cfg.disturbance_direction_reward_weight
            ),
            palm_center_alignment=float(
                self.cfg.palm_center_alignment_reward_weight
            ),
        )
        reward, terms = compute_embedded_reward(
            left_contact_force=self._sensor_force("left"),
            right_contact_force=self._sensor_force("right"),
            left_contact_count=self._contact_count("left"),
            right_contact_count=self._contact_count("right"),
            left_contact_presence_fraction=continuity_left,
            right_contact_presence_fraction=continuity_right,
            object_height_delta=height,
            object_linear_velocity=self.object.data.root_lin_vel_w,
            object_angular_velocity=self.object.data.root_ang_vel_w,
            hold_max_drift_m=self.max_hold_drift,
            challenge_translation_m=torch.where(disturbance, challenge_translation, torch.zeros_like(challenge_translation)),
            challenge_rotation_rad=torch.where(disturbance, challenge_rotation, torch.zeros_like(challenge_rotation)),
            current_penetration_m=self.current_penetration,
            maximum_penetration_m=self.maximum_penetration,
            penetration_measured=self.penetration_measured,
            penetration_limit_m=self.cfg.physx_penetration_max_m,
            force_closure_pass=self.force_closure_pass,
            force_closure_quality=self.force_closure_quality,
            ablation_phase=ablation_phase,
            ablation_stable_phase=ablation_stable_phase,
            ablation_active_contact_fraction=selected_active_fraction,
            ablation_inactive_contact_fraction=selected_inactive_fraction,
            actions=self.actions,
            previous_actions=self.previous_actions,
            joint_velocity=self.robot.data.joint_vel,
            lift_or_later=code >= PHASE_LIFT,
            hold_or_challenge=code >= PHASE_HOLD,
            disturbance_phase=disturbance,
            dropped=dropped,
            terminal_success=terminal_success,
            hard_gate_frontier_score=hard_gate_shaping_score,
            disturbance_direction_score=disturbance_direction_fraction,
            force_closure_ready=(
                hard_gate_prefixes["clear"] & gate_components["gravity_hold"]
                if self.cfg.force_closure_reward_requires_clearance
                else None
            ),
            hand_object_distance_m=hand_object_distance,
            approach_or_close=nominal_tracking_phase,
            proximity_scale_m=self.cfg.proximity_scale_m,
            nominal_tracking_score=nominal_tracking_score,
            nominal_tracking_phase=nominal_tracking_phase,
            palm_center_alignment_score=palm_alignment_score,
            lift_reward_target_m=self.cfg.lift_reward_target_m,
            lift_min_m=self.cfg.lift_min_m,
            gravity_drift_limit_m=self.cfg.gravity_drift_max_m,
            contact_presence_fraction_min=self.cfg.contact_presence_fraction_min,
            inactive_hand_contact_fraction_max=self.cfg.inactive_hand_contact_fraction_max,
            weights=reward_weights,
        )
        training_reward = self.cfg.training_reward_scale * reward
        self.extras["log"] = {f"reward/{name}": value.mean() for name, value in terms.items()}
        self.extras["log"]["reward/total_unscaled"] = reward.mean()
        self.extras["log"]["reward/total_scaled"] = training_reward.mean()
        for name, passed in gate_components.items():
            self.extras["log"][f"embedded/gate_{name}_fraction"] = passed.float().mean()
        for name, passed in hard_gate_prefixes.items():
            self.extras["log"][f"embedded/hard_prefix_{name}_fraction"] = (
                passed.float().mean()
            )
        self.extras["log"][
            "embedded/hard_disturbance_direction_prefix_fraction"
        ] = disturbance_direction_progress.mean()
        self.extras["log"][
            "embedded/hard_disturbance_direction_pass_fraction"
        ] = disturbance_direction_fraction.mean()
        self.extras["log"]["embedded/force_closure_evaluated_fraction"] = (
            self.force_closure_evaluated.float().mean()
        )
        if self.cfg.force_closure_reward_requires_clearance:
            self.extras["log"]["embedded/force_closure_reward_ready_fraction"] = (
                (hard_gate_prefixes["clear"] & gate_components["gravity_hold"])
                .float()
                .mean()
            )
        self.extras["log"]["embedded/terminal_success_fraction"] = terminal_success.float().mean()
        if float(self.cfg.palm_center_alignment_reward_weight) != 0.0:
            self.extras["log"]["reward/palm_center_alignment"] = (
                palm_alignment_score.mean()
            )
            self.extras["log"]["geometry/left_palm_center_cosine"] = (
                left_palm_cosine.mean()
            )
            self.extras["log"]["geometry/right_palm_center_cosine"] = (
                right_palm_cosine.mean()
            )
            self.extras["log"]["geometry/min_palm_center_cosine"] = torch.minimum(
                left_palm_cosine, right_palm_cosine
            ).mean()
        return training_reward

    def _metrics_report(self, env_id: int) -> dict[str, Any]:
        continuity = {
            "left": float((self.left_contact_steps[env_id] / self.contact_steps[env_id].clamp_min(1.0)).item()),
            "right": float((self.right_contact_steps[env_id] / self.contact_steps[env_id].clamp_min(1.0)).item()),
        }
        inactive = self.ablation_inactive_contact_steps[env_id] / self.ablation_steps[env_id].clamp_min(1.0)
        lift = self.lift_object_position[env_id, 2] - (
            self.initial_object_pos[env_id, 2] + self.scene.env_origins[env_id, 2]
        )
        arm_delta = torch.linalg.vector_norm(
            self.lift_q[env_id, self.arm_ids] - self.closure_q[env_id, self.arm_ids]
        )
        metrics = {
            "schema": "xhand_rl_embedded_episode_metrics_v2",
            "validator": "isaac_lab_embedded_rl_v2",
            "finite_state_pass": bool(
                torch.isfinite(self.robot.data.joint_pos[env_id]).all()
                and torch.isfinite(self.object.data.root_state_w[env_id]).all()
            ),
            "contact_presence_fraction": continuity,
            "lift_mode": "arm_joint_trajectory",
            "arm_joint_target_delta_l2_rad": float(arm_delta.item()),
            "robot_root_teleported_during_grasp_or_lift": False,
            "object_teleported_during_grasp_or_lift": False,
            "lift_displacement_m": float(lift.item()),
            "gravity_drift_m": float(self.gravity_drift[env_id].item()),
            "hold_max_drift_m": float(self.max_hold_drift[env_id].item()),
            "translation_disturbances_m": self.translation_displacements[env_id].detach().cpu().tolist(),
            "rotation_disturbances_rad": self.rotation_displacements[env_id].detach().cpu().tolist(),
            "physx_penetration_measured": bool(self.penetration_measured[env_id].item()),
            "maximum_physx_contact_penetration_m": float(self.maximum_penetration[env_id].item()),
            "force_closure_pass": bool(self.force_closure_pass[env_id].item()),
            "force_closure_quality": float(self.force_closure_quality[env_id].item()),
            "inactive_hand_contact_fraction": {
                "left_only": float(inactive[0].item()),
                "right_only": float(inactive[1].item()),
            },
            "single_hand_grasp_succeeded": {
                "left": bool(self.single_hand_success[env_id, 0].item()),
                "right": bool(self.single_hand_success[env_id, 1].item()),
            },
            "evaluated_policy_steps": int(self.episode_length_buf[env_id].item()),
            "final_audited_policy_step": int(self.last_audited_episode_step[env_id].item()),
        }
        if float(self.cfg.palm_center_alignment_reward_weight) != 0.0:
            alignment, left_cosine, right_cosine = self._palm_center_alignment()
            metrics["palm_center_alignment_score"] = float(alignment[env_id].item())
            metrics["left_palm_center_cosine"] = float(left_cosine[env_id].item())
            metrics["right_palm_center_cosine"] = float(right_cosine[env_id].item())
        thresholds = {
            "lift_min_m": self.cfg.lift_min_m,
            "gravity_drift_max_m": self.cfg.gravity_drift_max_m,
            "translation_disturbance_max_m": self.cfg.translation_disturbance_max_m,
            "rotation_disturbance_max_rad": self.cfg.rotation_disturbance_max_rad,
            "contact_presence_fraction_min": self.cfg.contact_presence_fraction_min,
            "inactive_hand_contact_fraction_max": self.cfg.inactive_hand_contact_fraction_max,
            "physx_penetration_max_m": self.cfg.physx_penetration_max_m,
        }
        metrics["physics_gates"] = evaluate_physics_metrics(metrics, thresholds)
        return metrics

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Isaac Lab invokes _get_dones() before _get_rewards(). Acceptance must
        # include the final ablation state, final penetration sample, and final
        # trajectory frame, so the post-physics audit belongs here.
        self._audit_post_physics_step()
        pos = self.object.data.root_pos_w - self.scene.env_origins
        object_escape = (pos[:, 2] < self.cfg.table_top_z_m - 0.05) | (
            torch.linalg.vector_norm(pos[:, :2] - self.object_start_pos[:, :2], dim=-1) > 0.35
        )
        finite = torch.isfinite(self.robot.data.joint_pos).all(dim=-1) & torch.isfinite(
            self.object.data.root_state_w
        ).all(dim=-1)
        terminated = (~finite) | (object_escape & self.cfg.terminate_on_object_escape)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        gate_pass = self._physics_gate_mask()
        terminal_ids = torch.nonzero(time_out, as_tuple=False).squeeze(-1)
        if terminal_ids.numel():
            self.last_terminal_audited_step[terminal_ids] = self.last_audited_episode_step[terminal_ids]
            self.last_terminal_phase[terminal_ids] = self._executed_phase_code()[terminal_ids]
            self.last_terminal_translation_count[terminal_ids] = torch.isfinite(
                self.translation_displacements[terminal_ids]
            ).sum(dim=-1)
            self.last_terminal_rotation_count[terminal_ids] = torch.isfinite(
                self.rotation_displacements[terminal_ids]
            ).sum(dim=-1)
            self.last_terminal_penetration_measured[terminal_ids] = self.penetration_measured[
                terminal_ids
            ]
            for env_id in terminal_ids.tolist():
                self.last_terminal_force_closure_evaluated[env_id] = (
                    self.force_closure_reports[env_id] is not None
                )
            self.last_terminal_gate_pass[terminal_ids] = gate_pass[terminal_ids]
        success = time_out & gate_pass
        ids = torch.nonzero(success, as_tuple=False).squeeze(-1)
        if ids.numel():
            self._success_serial_counter += 1
            self.completed_pregrasp_q[ids] = self.pregrasp_q[ids]
            self.completed_closure_q[ids] = self.closure_q[ids]
            self.completed_lift_q[ids] = self.lift_q[ids]
            self.completed_start_pose[ids] = self.last_start_pose[ids]
            self.completed_episode_steps[ids] = self.episode_length_buf[ids]
            self.completed_success_serial[ids] = self._success_serial_counter
            if self.trajectory_joint_q is not None:
                self.completed_trajectory_joint_q[ids] = self.trajectory_joint_q[ids]
                self.completed_trajectory_object_state[ids] = self.trajectory_object_state[ids]
                self.completed_trajectory_actions[ids] = self.trajectory_actions[ids]
                self.completed_trajectory_phase[ids] = self.trajectory_phase[ids]
            for env_id in ids.tolist():
                self.completed_reports[env_id] = self._metrics_report(env_id)
                self.completed_force_closure_reports[env_id] = self.force_closure_reports[env_id]
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        q = self.nominal_pregrasp[env_ids].clone()
        if not self.nominal_pose_lock:
            q += (2.0 * torch.rand_like(q) - 1.0) * self.cfg.reset_joint_noise_rad
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        q = torch.clamp(q, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        count = len(env_ids)
        root = self.object.data.default_root_state[env_ids].clone()
        if self.nominal_pose_lock and self.cfg.use_nominal_object_pose_for_reset:
            root[:, :3] = self.nominal_object_position + self.scene.env_origins[env_ids]
            root[:, 3:7] = self.nominal_object_quaternion
        elif self.nominal_pose_lock:
            # The locked nominal pose is a post-closure reference in the old
            # strict pipeline.  Start from the canonical table pose so the
            # close/lift trajectory can physically establish that pose.
            root[:, :3] = self.object_start_pos[env_ids] + self.scene.env_origins[env_ids]
            root[:, 3] = 1.0
            root[:, 4:7] = 0.0
        else:
            root[:, :3] = self.object_start_pos[env_ids] + self.scene.env_origins[env_ids]
            xy = (2.0 * torch.rand((count, 2), device=self.device) - 1.0) * self.cfg.reset_xy_noise_m
            yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * self.cfg.reset_yaw_noise_rad
            root[:, :2] += xy
            root[:, 3] = torch.cos(0.5 * yaw)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(0.5 * yaw)
        root[:, 7:] = 0.0
        self.object.write_root_state_to_sim(root, env_ids=env_ids)
        self.object.permanent_wrench_composer.reset(env_ids)
        self.initial_object_pos[env_ids] = root[:, :3] - self.scene.env_origins[env_ids]
        self.last_start_pose[env_ids] = root[:, :7]
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.integrated_residual[env_ids] = 0.0
        self.pregrasp_q[env_ids] = self.nominal_pregrasp[env_ids]
        self.closure_q[env_ids] = self.nominal_grasp[env_ids]
        self.lift_q[env_ids] = self.nominal_lift[env_ids]
        self.hold_robot_q[env_ids] = self.nominal_lift[env_ids]
        self.hold_object_state[env_ids] = 0.0
        self.hold_position[env_ids] = 0.0
        self.hold_quaternion[env_ids] = 0.0
        self.lift_object_position[env_ids] = 0.0
        self.previous_phase[env_ids] = -1
        self.last_audited_episode_step[env_ids] = -1
        self.current_wrench[env_ids] = 0.0
        self.contact_steps[env_ids] = 0.0
        self.left_contact_steps[env_ids] = 0.0
        self.right_contact_steps[env_ids] = 0.0
        self.max_hold_drift[env_ids] = 0.0
        self.gravity_drift[env_ids] = float("inf")
        self.translation_displacements[env_ids] = float("inf")
        self.rotation_displacements[env_ids] = float("inf")
        self.maximum_penetration[env_ids] = 0.0
        self.current_penetration[env_ids] = 0.0
        self.penetration_measured[env_ids] = False
        self.ablation_steps[env_ids] = 0.0
        self.ablation_inactive_contact_steps[env_ids] = 0.0
        self.ablation_active_contact_steps[env_ids] = 0.0
        self.single_hand_success[env_ids] = False
        self.force_closure_pass[env_ids] = False
        self.force_closure_evaluated[env_ids] = False
        self.force_closure_quality[env_ids] = 0.0
        self.hold_state_valid[env_ids] = False
        for env_id in env_ids.tolist():
            self.force_closure_reports[env_id] = None
