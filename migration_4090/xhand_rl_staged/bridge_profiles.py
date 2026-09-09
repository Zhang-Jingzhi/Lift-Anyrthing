"""Pure configuration profiles for the non-promotional lift bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class LiftBridgeProfile:
    name: str
    episode_length_s: float
    phase_fractions: tuple[float, float, float, float]
    target_height_m: float
    stable_hold_steps: int
    maximum_overshoot_m: float
    minimum_height_m: float
    linear_speed_max_m_s: float
    angular_speed_max_rad_s: float
    safety_linear_speed_max_m_s: float
    safety_angular_speed_max_rad_s: float
    action_group: str
    residual_activation: str
    residual_activation_close_fraction: float
    arm_approach_fraction_of_close: float
    finger_close_end_fraction_of_close: float
    residual_integration_end_fraction_of_hold: float
    residual_integration: float
    residual_limit_rad: float
    reset_xy_noise_m: float
    reset_yaw_noise_rad: float
    reset_joint_noise_rad: float
    height_progress_reward_weight: float
    height_tracking_reward_weight: float
    stability_reward_weight: float
    stable_hold_reward_weight: float
    contact_gate_reward_weight: float
    terminal_controlled_reward_weight: float
    terminal_stable_reward_weight: float
    overshoot_penalty_weight: float
    drop_penalty_weight: float
    linear_speed_penalty_weight: float
    angular_speed_penalty_weight: float
    unsafe_state_penalty_weight: float
    # Independent fixed-initial-pose ablation controls.  Defaults preserve
    # every historical bridge profile and its evaluation protocol.
    nominal_pose_lock: bool = False
    fixed_candidate_index: int | None = None
    palm_center_alignment_reward_weight: float = 0.0
    # Optional path-wise PhysX penetration guard.  ``None`` preserves the
    # historical bridge semantics; stable-lift profiles can opt in without
    # changing old controlled-lift checkpoints.
    penetration_limit_m: float | None = None
    # When enabled, residual actions are exposed only after the Stage-2
    # bilateral/continuity/distributed-contact conjunction is established.
    # This keeps the BODex approach and closure basin intact and makes a
    # contact loss fall back to the nominal controller immediately.
    contact_gated_residual: bool = False
    contact_gate_activation_threshold: float = 1.0

    @property
    def maximum_height_m(self) -> float:
        return self.target_height_m + self.maximum_overshoot_m

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "maximum_height_m": self.maximum_height_m}


def _profile(
    name: str,
    *,
    target_height_m: float,
    stable_hold_steps: int,
    stable: bool,
    residual_activation: str,
) -> LiftBridgeProfile:
    return LiftBridgeProfile(
        name=name,
        episode_length_s=3.2,
        phase_fractions=(0.15, 0.35, 0.25, 0.25),
        target_height_m=target_height_m,
        stable_hold_steps=stable_hold_steps,
        maximum_overshoot_m=0.020,
        minimum_height_m=-0.020,
        linear_speed_max_m_s=0.05,
        angular_speed_max_rad_s=0.50,
        safety_linear_speed_max_m_s=5.0,
        safety_angular_speed_max_rad_s=50.0,
        action_group="hands",
        residual_activation=residual_activation,
        residual_activation_close_fraction=0.0,
        arm_approach_fraction_of_close=0.0,
        finger_close_end_fraction_of_close=1.0,
        residual_integration_end_fraction_of_hold=1.0,
        residual_integration=0.0030,
        residual_limit_rad=0.12,
        reset_xy_noise_m=0.0030,
        reset_yaw_noise_rad=0.045,
        reset_joint_noise_rad=0.0045,
        height_progress_reward_weight=24.0,
        height_tracking_reward_weight=18.0,
        stability_reward_weight=16.0 if stable else 8.0,
        stable_hold_reward_weight=24.0 if stable else 8.0,
        contact_gate_reward_weight=14.0,
        terminal_controlled_reward_weight=120.0,
        terminal_stable_reward_weight=160.0 if stable else 40.0,
        overshoot_penalty_weight=80.0,
        drop_penalty_weight=30.0,
        linear_speed_penalty_weight=16.0 if stable else 8.0,
        angular_speed_penalty_weight=12.0 if stable else 6.0,
        unsafe_state_penalty_weight=120.0,
    )


BRIDGE_PROFILES = {
    profile.name: profile
    for profile in (
        _profile(
            "controlled_5mm_v1",
            target_height_m=0.005,
            stable_hold_steps=8,
            stable=False,
            residual_activation="hold",
        ),
        replace(
            _profile(
                "controlled_5mm_v2",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            # The first 14-iteration bridge improved bounded lifts but hurt
            # terminal retention for candidate 0.  Keep the same observation
            # and action ABI while making hold corrections smaller and
            # rewarding target retention/stability more than upward progress.
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "controlled_5mm_decoupled_settle_freeze25_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration_end_fraction_of_hold=0.25,
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "controlled_5mm_decoupled_settle_freeze50_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration_end_fraction_of_hold=0.50,
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "controlled_5mm_decoupled_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            # Use a real BODex-style pregrasp trajectory: approach with open
            # fingers first, then close the fingers only after the arm/wrist
            # target is reached.  The policy still acts only during hold.
            arm_approach_fraction_of_close=0.60,
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "controlled_5mm_decoupled_settle_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            # Three-part close: 45% open-hand arm approach, 35% finger
            # closure, then 20% grasp settling before lift begins.
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "controlled_5mm_slow_settle_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            # The nominal low-z candidates reach the target reliably, but
            # lose nearly all samples at terminal stability.  Slow every
            # physical transition and reserve 0.5 s at the end of close for
            # passive grasp settling before the 5 mm lift begins.
            episode_length_s=5.0,
            phase_fractions=(0.20, 0.40, 0.20, 0.20),
            arm_approach_fraction_of_close=0.35,
            finger_close_end_fraction_of_close=0.75,
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            reset_xy_noise_m=0.00075,
            reset_yaw_noise_rad=0.01125,
            reset_joint_noise_rad=0.001125,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "stable_5mm_decoupled_settle_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # Start from the safer three-part nominal and train only small
            # hold corrections.  Stability and terminal retention dominate;
            # upward progress is already supplied by paired-palm IK.
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration=0.0015,
            residual_limit_rad=0.08,
            height_progress_reward_weight=12.0,
            height_tracking_reward_weight=28.0,
            stability_reward_weight=28.0,
            stable_hold_reward_weight=48.0,
            contact_gate_reward_weight=12.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=160.0,
            overshoot_penalty_weight=120.0,
            drop_penalty_weight=80.0,
            linear_speed_penalty_weight=24.0,
            angular_speed_penalty_weight=18.0,
            unsafe_state_penalty_weight=160.0,
        ),
        replace(
            _profile(
                "stable_5mm_slow_settle_diverse_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # Preserve the full XY/yaw/joint reset diversity while giving
            # the nominal BODex controller substantially more time for the
            # arm approach, finger closure, and passive settling.  This is
            # an approach-dynamics ablation: residuals remain hold-only so
            # any improvement can be attributed to the nominal trajectory.
            episode_length_s=5.0,
            phase_fractions=(0.20, 0.40, 0.20, 0.20),
            arm_approach_fraction_of_close=0.35,
            finger_close_end_fraction_of_close=0.75,
            residual_integration=0.0015,
            residual_limit_rad=0.08,
            reset_xy_noise_m=0.003,
            reset_yaw_noise_rad=0.045,
            reset_joint_noise_rad=0.0045,
            height_progress_reward_weight=12.0,
            height_tracking_reward_weight=28.0,
            stability_reward_weight=28.0,
            stable_hold_reward_weight=48.0,
            contact_gate_reward_weight=12.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=160.0,
            overshoot_penalty_weight=120.0,
            drop_penalty_weight=80.0,
            linear_speed_penalty_weight=24.0,
            angular_speed_penalty_weight=18.0,
            unsafe_state_penalty_weight=160.0,
        ),
        replace(
            _profile(
                "stable_5mm_decoupled_settle_no_reset_noise_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # The BODex targets are expressed in the canonical object frame.
            # Until reset-time XY/yaw perturbations are transformed through a
            # matching arm IK, keep the object and nominal targets aligned.
            # Candidate selection remains round-robin across all four poses,
            # so this is not a fixed-pose or single-candidate ablation.
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration=0.0015,
            residual_limit_rad=0.08,
            reset_xy_noise_m=0.0,
            reset_yaw_noise_rad=0.0,
            reset_joint_noise_rad=0.0,
            penetration_limit_m=0.025,
            contact_gated_residual=True,
            contact_gate_activation_threshold=0.95,
            height_progress_reward_weight=12.0,
            height_tracking_reward_weight=28.0,
            stability_reward_weight=28.0,
            stable_hold_reward_weight=48.0,
            contact_gate_reward_weight=12.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=160.0,
            overshoot_penalty_weight=120.0,
            drop_penalty_weight=80.0,
            linear_speed_penalty_weight=24.0,
            angular_speed_penalty_weight=18.0,
            unsafe_state_penalty_weight=160.0,
        ),
        replace(
            _profile(
                "stable_5mm_decoupled_contact_gated_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # Keep the safer three-part BODex close/settle trajectory used by
            # ``stable_5mm_decoupled_settle_v1``.  The learned hand residual
            # is exposed only after the bilateral/contact-continuity gate is
            # established, so the adapter cannot alter approach or closure.
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration=0.00075,
            residual_limit_rad=0.030,
            penetration_limit_m=0.025,
            contact_gated_residual=True,
            contact_gate_activation_threshold=0.95,
            height_progress_reward_weight=10.0,
            height_tracking_reward_weight=32.0,
            stability_reward_weight=44.0,
            stable_hold_reward_weight=96.0,
            contact_gate_reward_weight=18.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=240.0,
            overshoot_penalty_weight=160.0,
            drop_penalty_weight=100.0,
            linear_speed_penalty_weight=36.0,
            angular_speed_penalty_weight=28.0,
            unsafe_state_penalty_weight=220.0,
        ),
        replace(
            _profile(
                "stable_5mm_decoupled_contact_gated_low_joint_noise_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # Preserve XY/yaw diversity while reducing reset finger/arm
            # perturbations.  The previous 4.5 mrad noise caused some BODex
            # pregrasp poses to start inside the sphere and fail during the
            # nominal approach before any residual could be useful.
            arm_approach_fraction_of_close=0.45,
            finger_close_end_fraction_of_close=0.80,
            residual_integration=0.00075,
            residual_limit_rad=0.030,
            reset_xy_noise_m=0.003,
            reset_yaw_noise_rad=0.045,
            reset_joint_noise_rad=0.001125,
            penetration_limit_m=0.025,
            contact_gated_residual=True,
            contact_gate_activation_threshold=0.95,
            height_progress_reward_weight=10.0,
            height_tracking_reward_weight=32.0,
            stability_reward_weight=44.0,
            stable_hold_reward_weight=96.0,
            contact_gate_reward_weight=18.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=240.0,
            overshoot_penalty_weight=160.0,
            drop_penalty_weight=100.0,
            linear_speed_penalty_weight=36.0,
            angular_speed_penalty_weight=28.0,
            unsafe_state_penalty_weight=220.0,
        ),
        replace(
            _profile(
                "controlled_5mm_v3",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="hold",
            ),
            # Phase diagnostics showed that 48/58 first unsafe events occurred
            # during approach, before bridge residuals are active.  Preserve
            # the four BODex modes as the diversity source while starting the
            # robustness curriculum at one quarter of the Stage-2 reset noise.
            residual_integration=0.0020,
            residual_limit_rad=0.10,
            reset_xy_noise_m=0.00075,
            reset_yaw_noise_rad=0.01125,
            reset_joint_noise_rad=0.001125,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "formation_5mm_close_wrist_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="close",
            ),
            # Diagnostic formation retry: preserve the BODex/IK nominal
            # trajectory and expose only distal wrist rows during the final
            # half of close.  The integrated correction is capped at 35 mrad
            # and then held fixed through lift/hold.
            action_group="distal_wrist",
            residual_activation_close_fraction=0.50,
            residual_integration=0.0020,
            residual_limit_rad=0.035,
            height_progress_reward_weight=20.0,
            height_tracking_reward_weight=24.0,
            stability_reward_weight=12.0,
            stable_hold_reward_weight=16.0,
            contact_gate_reward_weight=18.0,
            terminal_controlled_reward_weight=60.0,
            terminal_stable_reward_weight=40.0,
            overshoot_penalty_weight=100.0,
            drop_penalty_weight=60.0,
            linear_speed_penalty_weight=12.0,
            angular_speed_penalty_weight=10.0,
        ),
        replace(
            _profile(
                "formation_5mm_close_wrist_retry_v2",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="close",
            ),
            # Conservative retry after the v1 formation run showed PPO drift:
            # keep the same nominal BODex/controller trajectory and the same
            # six distal-wrist rows, but reduce the residual budget by ~43%.
            # This profile is intentionally separate so old evaluations and
            # checkpoints remain reproducible.
            action_group="distal_wrist",
            residual_activation_close_fraction=0.50,
            residual_integration=0.0010,
            residual_limit_rad=0.020,
            height_progress_reward_weight=16.0,
            height_tracking_reward_weight=28.0,
            stability_reward_weight=16.0,
            stable_hold_reward_weight=20.0,
            contact_gate_reward_weight=22.0,
            terminal_controlled_reward_weight=50.0,
            terminal_stable_reward_weight=60.0,
            overshoot_penalty_weight=120.0,
            drop_penalty_weight=80.0,
            linear_speed_penalty_weight=16.0,
            angular_speed_penalty_weight=14.0,
            unsafe_state_penalty_weight=140.0,
        ),
        replace(
            _profile(
                "formation_5mm_fixed_pose_palm_align_v1",
                target_height_m=0.005,
                stable_hold_steps=8,
                stable=False,
                residual_activation="close",
            ),
            # Independent ablation: remove all reset pose noise and always
            # start from the same BODex candidate (Candidate 2).  PPO still
            # receives a reproducible numeric seed; "no seed" here means no
            # initial-pose randomization, not removal of PPO stochasticity.
            nominal_pose_lock=True,
            fixed_candidate_index=2,
            palm_center_alignment_reward_weight=3.0,
            action_group="distal_wrist",
            residual_activation_close_fraction=0.25,
            residual_integration=0.0010,
            residual_limit_rad=0.015,
            reset_xy_noise_m=0.0,
            reset_yaw_noise_rad=0.0,
            reset_joint_noise_rad=0.0,
            height_progress_reward_weight=16.0,
            height_tracking_reward_weight=28.0,
            stability_reward_weight=16.0,
            stable_hold_reward_weight=20.0,
            contact_gate_reward_weight=22.0,
            terminal_controlled_reward_weight=50.0,
            terminal_stable_reward_weight=60.0,
            overshoot_penalty_weight=120.0,
            drop_penalty_weight=80.0,
            linear_speed_penalty_weight=16.0,
            angular_speed_penalty_weight=14.0,
            unsafe_state_penalty_weight=140.0,
        ),
        replace(
            _profile(
                "stable_5mm_wrist_guard_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="close",
            ),
            # Preserve the BODex finger closure and train only small,
            # late-stage wrist corrections.  Unlike the fixed-pose ablation,
            # this profile keeps all four candidates and their reset noise.
            action_group="distal_wrist",
            residual_activation_close_fraction=0.25,
            residual_integration=0.0010,
            residual_limit_rad=0.015,
            palm_center_alignment_reward_weight=1.5,
            penetration_limit_m=0.025,
            stability_reward_weight=32.0,
            stable_hold_reward_weight=64.0,
            terminal_controlled_reward_weight=50.0,
            terminal_stable_reward_weight=180.0,
            overshoot_penalty_weight=140.0,
            drop_penalty_weight=90.0,
            linear_speed_penalty_weight=28.0,
            angular_speed_penalty_weight=22.0,
            unsafe_state_penalty_weight=180.0,
        ),
        replace(
            _profile(
                "stable_5mm_hold_wrist_guard_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="hold",
            ),
            # The close-active wrist guard reduced contact quality and did
            # not improve stable holds.  Preserve the complete BODex grasp
            # and paired-palm IK lift, then expose only small distal-wrist
            # corrections during the first half of hold.  Freezing the
            # correction for the second half gives PPO a direct settling
            # objective without allowing late action chatter to invalidate
            # an otherwise controlled lift.
            action_group="distal_wrist",
            residual_integration_end_fraction_of_hold=0.50,
            residual_integration=0.0005,
            residual_limit_rad=0.008,
            palm_center_alignment_reward_weight=0.0,
            penetration_limit_m=0.025,
            height_progress_reward_weight=8.0,
            height_tracking_reward_weight=32.0,
            stability_reward_weight=40.0,
            stable_hold_reward_weight=80.0,
            contact_gate_reward_weight=14.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=220.0,
            overshoot_penalty_weight=160.0,
            drop_penalty_weight=100.0,
            linear_speed_penalty_weight=36.0,
            angular_speed_penalty_weight=28.0,
            unsafe_state_penalty_weight=200.0,
        ),
        replace(
            _profile(
                "stable_5mm_contact_gated_hold_v1",
                target_height_m=0.005,
                stable=True,
                stable_hold_steps=32,
                residual_activation="hold",
            ),
            # The learned action is completely suppressed until the Stage-2
            # contact conjunction is established.  It is then allowed only
            # during hold, with a small integrated budget over all hand rows.
            # If either hand loses contact, the integrated residual is reset
            # and the nominal BODex/IK target is restored on the same step.
            action_group="hands",
            residual_integration=0.00075,
            residual_limit_rad=0.030,
            palm_center_alignment_reward_weight=0.0,
            penetration_limit_m=0.025,
            contact_gated_residual=True,
            contact_gate_activation_threshold=0.95,
            height_progress_reward_weight=10.0,
            height_tracking_reward_weight=32.0,
            stability_reward_weight=44.0,
            stable_hold_reward_weight=96.0,
            contact_gate_reward_weight=18.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=240.0,
            overshoot_penalty_weight=160.0,
            drop_penalty_weight=100.0,
            linear_speed_penalty_weight=36.0,
            angular_speed_penalty_weight=28.0,
            unsafe_state_penalty_weight=220.0,
        ),
        replace(
            _profile(
                "stable_5mm_contact_gated_hold_freeze_v1",
                target_height_m=0.005,
                stable=True,
                stable_hold_steps=32,
                residual_activation="hold",
            ),
            # Keep the contact-conditioned hand adapter conservative: allow
            # it to settle the object during the first half of hold, then
            # freeze the integrated correction so late PPO chatter cannot
            # undo an otherwise valid grasp.
            action_group="hands",
            residual_integration_end_fraction_of_hold=0.50,
            residual_integration=0.0005,
            residual_limit_rad=0.015,
            palm_center_alignment_reward_weight=0.0,
            penetration_limit_m=0.025,
            contact_gated_residual=True,
            contact_gate_activation_threshold=0.95,
            height_progress_reward_weight=8.0,
            height_tracking_reward_weight=32.0,
            stability_reward_weight=48.0,
            stable_hold_reward_weight=120.0,
            contact_gate_reward_weight=18.0,
            terminal_controlled_reward_weight=40.0,
            terminal_stable_reward_weight=260.0,
            overshoot_penalty_weight=160.0,
            drop_penalty_weight=100.0,
            linear_speed_penalty_weight=36.0,
            angular_speed_penalty_weight=28.0,
            unsafe_state_penalty_weight=220.0,
        ),
        replace(
            _profile(
                "stable_5mm_v1",
                target_height_m=0.005,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            penetration_limit_m=0.025,
        ),
        _profile(
            "controlled_10mm_v1",
            target_height_m=0.010,
            stable_hold_steps=8,
            stable=False,
            residual_activation="lift",
        ),
        _profile(
            "stable_10mm_v1",
            target_height_m=0.010,
            stable_hold_steps=32,
            stable=True,
            residual_activation="lift",
        ),
        # Paired with the grasp-acquisition freeze: get the contact early, spend
        # the rest of the episode lifting.
        #
        # lift50_hold1s_v1 spends 40% of the episode approaching and 25% closing,
        # which under the freeze puts the release at step 672 of 906.  That
        # leaves 1.9 s for lift and hold and is why contact_continuity sits at
        # 0.05-0.16 while the contact itself is finally good: 5/3 groups per
        # side and distributed contacts 0.818, against 1/1 and 0.256 unfrozen.
        #
        # The object cannot move while frozen, so the approach does not need the
        # time it was given to avoid disturbing it.  Approach and closure drop to
        # 45% between them and lift and hold take 55%, about 4 s.
        replace(
            _profile(
                "lift50_freeze_v1",
                target_height_m=0.050,
                stable_hold_steps=125,
                stable=True,
                residual_activation="approach",
            ),
            action_group="all",
            episode_length_s=7.25,
            phase_fractions=(0.25, 0.20, 0.30, 0.25),
            maximum_overshoot_m=0.100,
            residual_integration=0.010,
            residual_limit_rad=0.30,
        ),
        # Lift off the table and hold, scored separately from height tracking.
        #
        # standoff_80mm_v1 conflates the two.  Its maximum_overshoot_m is 0.020,
        # inherited from the 10 mm profiles, so the band is 60-100 mm -- and 82.4%
        # of the rollouts that clear the neighbouring task's own bar (ball above
        # 80 mm at the end with both hands in contact) finish above 100 mm and are
        # scored as failures by our own gate.  The median successful lift is
        # 115 mm.  Penalising an 115 mm two-handed lift as an overshoot is not a
        # useful objective for a task whose stated requirement is lifting.
        #
        # So the target drops to 50 mm with a 100 mm overshoot allowance, which
        # accepts anything from 50 to 150 mm, and the hold requirement rises from
        # 32 steps to 125 -- a full second at this control rate, four times the
        # old window.  Height tracking is not abandoned, it is measured
        # separately; overshoot keeps its penalty weight so drifting upward still
        # costs, it just no longer disqualifies.
        replace(
            _profile(
                "lift50_hold1s_v1",
                target_height_m=0.050,
                stable_hold_steps=125,
                stable=True,
                residual_activation="approach",
            ),
            action_group="all",
            episode_length_s=7.25,
            phase_fractions=(0.40, 0.25, 0.20, 0.15),
            maximum_overshoot_m=0.100,
            residual_integration=0.010,
            residual_limit_rad=0.30,
        ),
        # standoff at the neighbouring task's own goal height.
        #
        # Measured 2026-09-08: under a 10 mm target the open-loop rollout holds
        # the ball above 20 mm with both hands in contact in 35.4% of episodes
        # but clears 80 mm in 0.8%, because nothing ever commands an 80 mm lift --
        # the lift targets raise the palms by 10 mm.  The neighbouring skrl task
        # sets goal_lift_height to 0.08 and judges stability by tilt, with no
        # hand-object penetration term anywhere in its environment; ours requires
        # 32 steps inside 0.05 m/s and 0.5 rad/s, which is the stricter test.
        #
        # So the height moves to theirs and the stability bounds stay ours: same
        # stable_hold_steps, same speed bounds, same overshoot allowance.
        replace(
            _profile(
                "standoff_80mm_v1",
                target_height_m=0.080,
                stable_hold_steps=32,
                stable=True,
                residual_activation="approach",
            ),
            action_group="all",
            episode_length_s=7.25,
            phase_fractions=(0.40, 0.25, 0.20, 0.15),
            residual_integration=0.010,
            residual_limit_rad=0.30,
        ),
        # The standoff formulation, mirroring how the neighbouring skrl task
        # actually works rather than how we assumed it worked.
        #
        # Everything measured on 2026-09-07 points at one interlock.  The BODex
        # pregrasp puts the palms 0.89-7.39 mm from the ball; the arms replay a
        # fixed trajectory to get there; the policy may only nudge fingers after
        # the grasp is formed.  Each single fix is cancelled by the other two:
        # paying for lift changed nothing (0.0215 vs 0.0234 over six seeds),
        # widening the residual to PHASE_CLOSE made it worse (0.0143 vs 0.0306),
        # and softening the arms to the neighbour's 800/60 traded penetration
        # (3.63 -> 2.81 mm) for contact (bilateral 0.571 -> 0.489) because at
        # that distance stiffness is what creates contact at all.
        #
        # The neighbour starts its hands 100-120 mm out (their standoffs are
        # 0.10 m left and 0.12 m right for narrow objects) and closes with arm
        # actions the policy owns, holding the pregrasp as an attractor it may
        # leave.  So they are not learning from less prior than us -- they hold
        # the same kind of prior as a guide where we hold it as a constraint.
        #
        # This profile does the same: start retracted (pass --retracted-pregrasp),
        # let the residual act from PHASE_APPROACH, and give it every joint
        # rather than fingers only.  Episode length and the softer gains follow
        # the neighbour too, since a policy that must drive the arms in needs
        # time and needs an arm that yields on contact.
        replace(
            _profile(
                "standoff_10mm_v1",
                target_height_m=0.010,
                stable_hold_steps=32,
                stable=True,
                residual_activation="approach",
            ),
            action_group="all",
            episode_length_s=7.25,
            phase_fractions=(0.40, 0.25, 0.20, 0.15),
            residual_integration=0.010,
            residual_limit_rad=0.30,
        ),
        # stable_10mm_v1 with the residual given room to matter.
        #
        # Measured 2026-09-07 over six seeds x 512 episodes, the retrained
        # lift-reward policy ties a zero action vector on every metric
        # (sustained 0.0215 vs 0.0234) even after lift_height and stable_lift
        # were paid for the first time.  Under stable_10mm_v1 the residual only
        # activates at PHASE_LIFT, touches hand joints alone, and is clamped to
        # 0.12 rad -- roughly +-7 degrees of finger adjustment applied after the
        # grasp has already been formed open-loop.  A grasp that is going to
        # fail is already failing by then.
        #
        # This opens the residual at PHASE_CLOSE, where the fingers are still
        # arriving, and widens the clamp and integration rate to match
        # stable_35mm_free_v1.  Everything else -- target, hold length, noise,
        # phase fractions -- is stable_10mm_v1 untouched, so a difference is
        # attributable to the residual's authority and nothing else.
        replace(
            _profile(
                "stable_10mm_free_v1",
                target_height_m=0.010,
                stable_hold_steps=32,
                stable=True,
                residual_activation="close",
            ),
            residual_integration=0.010,
            residual_limit_rad=0.15,
        ),
        replace(
            _profile(
                "stable_35mm_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            penetration_limit_m=0.003,
        ),
        replace(
            _profile(
                "stable_35mm_slow_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            episode_length_s=8.0,
            penetration_limit_m=0.003,
        ),
        replace(
            _profile(
                "stable_35mm_lownoise_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            reset_xy_noise_m=0.0010,
            reset_yaw_noise_rad=0.015,
            reset_joint_noise_rad=0.0015,
            penetration_limit_m=0.003,
        ),
        replace(
            _profile(
                "stable_35mm_slowapproach_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            # Paired with a retracted start, the approach has to cover ~90 mm.
            # At the usual 0.15 fraction of a 3.2 s episode that is ~190 mm/s and
            # the hands drive into the object instead of onto it: measured
            # 2026-09-07, retracting alone cut approach disturbance 23.8 -> 3.8 mm
            # but tripled contact penetration to 10.3 mm.  Here the approach gets
            # 3.2 s, about 28 mm/s.
            episode_length_s=8.0,
            phase_fractions=(0.40, 0.25, 0.20, 0.15),
            penetration_limit_m=0.003,
        ),
        replace(
            _profile(
                "stable_35mm_free_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="close",
            ),
            # The tight residual is the suspected cap: at 0.02 rad, active only
            # from the lift phase, the policy cannot touch the approach and
            # close phases where the object is already knocked to 0.9 m/s.
            # bridge_env only allows the "hands" action group, so the arms stay
            # controller-driven either way -- this widens what little the policy
            # does control.
            residual_limit_rad=0.15,
            residual_integration=0.010,
            penetration_limit_m=0.003,
        ),
        replace(
            _profile(
                "stable_35mm_nonoise_v1",
                target_height_m=0.035,
                stable_hold_steps=32,
                stable=True,
                residual_activation="lift",
            ),
            reset_xy_noise_m=0.0,
            reset_yaw_noise_rad=0.0,
            reset_joint_noise_rad=0.0,
            penetration_limit_m=0.003,
        ),
    )
}


def get_bridge_profile(name: str) -> LiftBridgeProfile:
    try:
        return BRIDGE_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown lift bridge profile {name!r}; choose from "
            f"{sorted(BRIDGE_PROFILES)}"
        ) from exc
