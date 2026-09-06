#!/usr/bin/env python3
"""Train one locked-object PPO policy with embedded physical gates."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--nominal-dataset", type=Path, required=True)
parser.add_argument("--nominal-sample-index", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--max-iterations", type=int, default=4000)
parser.add_argument("--seed", type=int, default=84)
parser.add_argument("--penetration-reward-weight", type=float)
parser.add_argument("--penetration-clear-reward-weight", type=float)
parser.add_argument("--proximity-reward-weight", type=float)
parser.add_argument("--terminal-success-weight", type=float)
parser.add_argument("--stable-lift-reward-weight", type=float)
parser.add_argument("--force-closure-reward-weight", type=float)
parser.add_argument("--hard-gate-frontier-reward-weight", type=float)
parser.add_argument("--disturbance-direction-reward-weight", type=float)
parser.add_argument(
    "--force-closure-shape-all-hold-contacts",
    action="store_true",
    default=None,
)
parser.add_argument(
    "--force-closure-requires-clearance",
    action="store_true",
    default=None,
    help="gate dense force-closure shaping on the same-trajectory strict clearance+gravity prefix",
)
parser.add_argument(
    "--disturbance-reward-requires-force-closure",
    action="store_true",
    default=None,
    help="gate disturbance-direction shaping on the same-trajectory stable force-closure prefix",
)
parser.add_argument("--gamma", type=float)
parser.add_argument(
    "--learning-rate",
    type=float,
    help="fixed PPO learning-rate override for an append-only reward-transition retry",
)
parser.add_argument("--init-noise-std", type=float)
parser.add_argument("--entropy-coef", type=float)
parser.add_argument("--freeze-policy-noise", action="store_true")
parser.add_argument(
    "--adaptive-prefix-lr-factor",
    type=float,
    default=0.25,
    help="multiply PPO learning rate after a new strict-prefix checkpoint is captured",
)
parser.add_argument("--instantaneous-residual-weight", type=float)
parser.add_argument("--residual-integration", type=float)
parser.add_argument("--residual-limit-rad", type=float)
parser.add_argument("--arm-action-scale-rad", type=float)
parser.add_argument("--hand-action-scale-rad", type=float)
parser.add_argument("--approach-fraction", type=float)
parser.add_argument("--close-fraction", type=float)
parser.add_argument("--lift-fraction", type=float)
parser.add_argument("--hold-fraction", type=float)
parser.add_argument("--disturbance-fraction", type=float)
parser.add_argument("--ablation-fraction", type=float)
parser.add_argument(
    "--residual-activation-phase",
    type=int,
    choices=(0, 1, 2, 3),
    help="first phase in which residual actions are applied (2=lifting, 3=hold)",
)
parser.add_argument("--resume-checkpoint", type=Path)
parser.add_argument(
    "--resume-prefix-state",
    type=Path,
    help=(
        "restore the exact Isaac robot/object/gate state paired with a hard "
        "prefix checkpoint; policy weights alone do not reproduce that basin"
    ),
)
parser.add_argument(
    "--reset-optimizer-on-resume",
    action="store_true",
    help="load policy/value weights but reset PPO optimizer state for a new shaping revision",
)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from .launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"train_{args.object}")
simulation_app = AppLauncher(args).app

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .configuration import configure_env
from .checkpoint_selection import select_collection_checkpoint
from .contracts import (
    EMBEDDED_BACKEND,
    KIT_SETTINGS_PROFILE,
    PPO_REVISION,
    REWARD_REVISION,
    SINGLE_GPU_KIT_ARGS,
    TRAINING_RUN_SCHEMA,
    TRAINING_REWARD_SCALE,
    latest_policy_checkpoint,
    resume_checkpoint_iteration,
    sha256_file,
    validate_manifest,
    validate_precheckpoint_restart,
)
from .env import XHandEmbeddedEnv, XHandEmbeddedEnvCfg
from .ppo_cfg import XHandEmbeddedPPORunnerCfg
from .training_capture import install_exact_success_capture


def _override_policy_noise(policy, noise_std: float) -> None:
    """Apply an explicit action-noise override after checkpoint loading.

    RSL-RL checkpoints contain the learned policy standard deviation.  Merely
    changing ``init_noise_std`` in the runner config therefore has no effect
    on a resumed run; the override must be applied to the loaded policy
    parameter itself.  The locked PPO configuration uses a state-independent
    scalar/log standard deviation, so both representations are supported.
    """

    import torch

    with torch.no_grad():
        noise_type = getattr(policy, "noise_std_type", "scalar")
        if noise_type == "scalar" and hasattr(policy, "std"):
            policy.std.fill_(float(noise_std))
            return
        if noise_type == "log" and hasattr(policy, "log_std"):
            policy.log_std.fill_(math.log(float(noise_std)))
            return
    raise RuntimeError(
        "--init-noise-std override requires a state-independent scalar/log "
        "RSL-RL policy standard deviation"
    )


def _freeze_policy_noise(policy) -> None:
    """Keep the selected exploration standard deviation fixed during PPO."""

    for name in ("std", "log_std"):
        parameter = getattr(policy, name, None)
        if parameter is not None:
            parameter.requires_grad_(False)
            return
    raise RuntimeError("--freeze-policy-noise requires a scalar/log policy std")


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    if not args.nominal_dataset.is_file():
        raise FileNotFoundError(args.nominal_dataset)
    nominal_payload = __import__("torch").load(
        args.nominal_dataset, map_location="cpu", weights_only=False
    )
    nominal_samples = (
        nominal_payload.get("samples", [])
        if isinstance(nominal_payload, dict)
        else nominal_payload
    )
    if not isinstance(nominal_samples, (list, tuple)):
        raise RuntimeError("nominal dataset does not contain a sample sequence")
    if not 0 <= args.nominal_sample_index < len(nominal_samples):
        raise IndexError("nominal sample index is outside the dataset")
    nominal = nominal_samples[args.nominal_sample_index]
    nominal_legacy_warm_start = bool(
        isinstance(nominal_payload, dict)
        and (
            nominal_payload.get("generation_backend") == "legacy_derived_warm_start_only"
            or (
                nominal_samples
                and nominal_samples[0].get("generation_backend")
                == "legacy_derived_warm_start_only"
            )
        )
    )
    nominal_legacy_reused_as_accepted = bool(
        isinstance(nominal_payload, dict)
        and nominal_payload.get("legacy_data_reused_as_accepted", False)
    )
    if nominal_legacy_warm_start and nominal_legacy_reused_as_accepted:
        raise RuntimeError("legacy warm-start declares accepted-data reuse")
    expected_kit_args = f"{SINGLE_GPU_KIT_ARGS} --portable-root {args.kit_portable_root}"
    if (
        args.kit_settings_profile != KIT_SETTINGS_PROFILE
        or args.kit_args != expected_kit_args
        or args.multi_gpu is not False
    ):
        raise RuntimeError("v2 training requires the locked single-GPU renderer settings")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint is None:
        validate_precheckpoint_restart(
            args.output,
            manifest_path=args.manifest,
            nominal_dataset=args.nominal_dataset,
            object_name=args.object,
            nominal_sample_index=args.nominal_sample_index,
            seed=args.seed,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            device=args.device or "cuda:0",
            kit_settings_profile=args.kit_settings_profile,
            renderer_multi_gpu=args.multi_gpu,
        )
    cfg = configure_env(
        XHandEmbeddedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        nominal_dataset=args.nominal_dataset,
        nominal_sample_index=args.nominal_sample_index,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        # Exact hard-success trajectories must survive the environment reset
        # that follows a terminal PPO rollout.  The buffers are GPU-resident
        # and bounded by num_envs * episode_length; nothing is written unless
        # all embedded gates actually pass.
        record_trajectory=True,
    )
    # When an interrupted retry is resumed, preserve the residual semantics
    # recorded by that attempt unless the caller explicitly overrides them.
    # This prevents a direct-residual retry from silently switching back to
    # integrated residuals after a process restart.
    previous = {}
    if args.resume_checkpoint is not None:
        provenance_path = args.output / "training_provenance.json"
        if provenance_path.is_file():
            try:
                previous = json.loads(provenance_path.read_text())
            except (OSError, TypeError, ValueError):
                previous = {}
            if args.instantaneous_residual_weight is None:
                cfg.instantaneous_residual_weight = float(
                    previous.get(
                        "instantaneous_residual_weight",
                        cfg.instantaneous_residual_weight,
                    )
                )
            if args.residual_integration is None:
                cfg.residual_integration = float(
                    previous.get("residual_integration", cfg.residual_integration)
                )
            if args.residual_activation_phase is None:
                cfg.residual_activation_phase = int(
                    previous.get("residual_activation_phase", cfg.residual_activation_phase)
                )
            if args.stable_lift_reward_weight is None:
                cfg.stable_lift_reward_weight = float(
                    previous.get(
                        "stable_lift_reward_weight", cfg.stable_lift_reward_weight
                    )
                )
            if args.penetration_clear_reward_weight is None:
                cfg.penetration_clear_reward_weight = float(
                    previous.get(
                        "penetration_clear_reward_weight",
                        cfg.penetration_clear_reward_weight,
                    )
                )
            if args.proximity_reward_weight is None:
                cfg.proximity_reward_weight = float(
                    previous.get(
                        "proximity_reward_weight", cfg.proximity_reward_weight
                    )
                )
            if args.force_closure_reward_weight is None:
                cfg.force_closure_reward_weight = float(
                    previous.get(
                        "force_closure_reward_weight", cfg.force_closure_reward_weight
                    )
                )
            if args.force_closure_shape_all_hold_contacts is None:
                cfg.force_closure_shape_all_hold_contacts = bool(
                    previous.get(
                        "force_closure_shape_all_hold_contacts",
                        cfg.force_closure_shape_all_hold_contacts,
                    )
                )
            if args.force_closure_requires_clearance is None:
                cfg.force_closure_reward_requires_clearance = bool(
                    previous.get(
                        "force_closure_reward_requires_clearance",
                        cfg.force_closure_reward_requires_clearance,
                    )
                )
            if args.disturbance_reward_requires_force_closure is None:
                cfg.disturbance_reward_requires_force_closure = bool(
                    previous.get(
                        "disturbance_reward_requires_force_closure",
                        cfg.disturbance_reward_requires_force_closure,
                    )
                )
            if args.hard_gate_frontier_reward_weight is None:
                cfg.hard_gate_frontier_reward_weight = float(
                    previous.get(
                        "hard_gate_frontier_reward_weight",
                        cfg.hard_gate_frontier_reward_weight,
                    )
                )
            if args.disturbance_direction_reward_weight is None:
                cfg.disturbance_direction_reward_weight = float(
                    previous.get(
                        "disturbance_direction_reward_weight",
                        cfg.disturbance_direction_reward_weight,
                    )
                )
            if args.residual_limit_rad is None:
                cfg.residual_limit_rad = float(
                    previous.get("residual_limit_rad", cfg.residual_limit_rad)
                )
            for argument, field in (
                ("approach_fraction", "approach_fraction"),
                ("close_fraction", "close_fraction"),
                ("lift_fraction", "lift_fraction"),
                ("hold_fraction", "hold_fraction"),
                ("disturbance_fraction", "disturbance_fraction"),
                ("ablation_fraction", "ablation_fraction"),
            ):
                if getattr(args, argument) is None and field in previous:
                    setattr(cfg, field, float(previous[field]))
    if args.penetration_reward_weight is not None:
        if args.penetration_reward_weight >= 0.0:
            raise ValueError("penetration reward weight must be negative")
        cfg.penetration_reward_weight = float(args.penetration_reward_weight)
    if args.penetration_clear_reward_weight is not None:
        if args.penetration_clear_reward_weight < 0.0:
            raise ValueError("penetration-clear reward weight must be non-negative")
        cfg.penetration_clear_reward_weight = float(args.penetration_clear_reward_weight)
    if args.proximity_reward_weight is not None:
        if args.proximity_reward_weight < 0.0:
            raise ValueError("proximity reward weight must be non-negative")
        cfg.proximity_reward_weight = float(args.proximity_reward_weight)
    if args.terminal_success_weight is not None:
        if args.terminal_success_weight < 0.0:
            raise ValueError("terminal success weight must be non-negative")
        cfg.terminal_success_weight = float(args.terminal_success_weight)
    if args.stable_lift_reward_weight is not None:
        if args.stable_lift_reward_weight < 0.0:
            raise ValueError("stable lift reward weight must be non-negative")
        cfg.stable_lift_reward_weight = float(args.stable_lift_reward_weight)
    if args.force_closure_reward_weight is not None:
        if args.force_closure_reward_weight < 0.0:
            raise ValueError("force closure reward weight must be non-negative")
        cfg.force_closure_reward_weight = float(args.force_closure_reward_weight)
    if args.force_closure_shape_all_hold_contacts is not None:
        cfg.force_closure_shape_all_hold_contacts = bool(
            args.force_closure_shape_all_hold_contacts
        )
    if args.force_closure_requires_clearance is not None:
        cfg.force_closure_reward_requires_clearance = bool(
            args.force_closure_requires_clearance
        )
    if args.disturbance_reward_requires_force_closure is not None:
        cfg.disturbance_reward_requires_force_closure = bool(
            args.disturbance_reward_requires_force_closure
        )
    if args.hard_gate_frontier_reward_weight is not None:
        if args.hard_gate_frontier_reward_weight < 0.0:
            raise ValueError("hard gate frontier reward weight must be non-negative")
        cfg.hard_gate_frontier_reward_weight = float(
            args.hard_gate_frontier_reward_weight
        )
    if args.disturbance_direction_reward_weight is not None:
        if args.disturbance_direction_reward_weight < 0.0:
            raise ValueError("disturbance direction reward weight must be non-negative")
        cfg.disturbance_direction_reward_weight = float(
            args.disturbance_direction_reward_weight
        )
    if args.instantaneous_residual_weight is not None:
        if args.instantaneous_residual_weight < 0.0:
            raise ValueError("instantaneous residual weight must be non-negative")
        cfg.instantaneous_residual_weight = float(args.instantaneous_residual_weight)
    if args.residual_integration is not None:
        if args.residual_integration < 0.0:
            raise ValueError("residual integration must be non-negative")
        cfg.residual_integration = float(args.residual_integration)
    if args.residual_limit_rad is not None:
        if args.residual_limit_rad <= 0.0:
            raise ValueError("residual limit must be positive")
        cfg.residual_limit_rad = float(args.residual_limit_rad)
    if args.arm_action_scale_rad is not None:
        if args.arm_action_scale_rad <= 0.0:
            raise ValueError("arm action scale must be positive")
        cfg.arm_action_scale_rad = float(args.arm_action_scale_rad)
    if args.hand_action_scale_rad is not None:
        if args.hand_action_scale_rad <= 0.0:
            raise ValueError("hand action scale must be positive")
        cfg.hand_action_scale_rad = float(args.hand_action_scale_rad)
    if args.residual_activation_phase is not None:
        cfg.residual_activation_phase = int(args.residual_activation_phase)
    for argument, field in (
        ("approach_fraction", "approach_fraction"),
        ("close_fraction", "close_fraction"),
        ("lift_fraction", "lift_fraction"),
        ("hold_fraction", "hold_fraction"),
        ("disturbance_fraction", "disturbance_fraction"),
        ("ablation_fraction", "ablation_fraction"),
    ):
        value = getattr(args, argument)
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{argument.replace('_', ' ')} must be non-negative")
            setattr(cfg, field, float(value))
    if abs(
        sum(
            float(getattr(cfg, field))
            for field in (
                "approach_fraction",
                "close_fraction",
                "lift_fraction",
                "hold_fraction",
                "disturbance_fraction",
                "ablation_fraction",
            )
        )
        - 1.0
    ) > 1.0e-6:
        raise ValueError("embedded episode fractions must sum to one")
    runner_cfg = XHandEmbeddedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg.run_name = args.object
    if nominal_legacy_warm_start:
        # The legacy-derived pose is a contact-preserving warm start, not an
        # accepted sample.  Keep the initial PPO exploration small enough to
        # refine that pose instead of immediately destroying its contacts.
        runner_cfg.policy.init_noise_std = 0.15
        runner_cfg.algorithm.entropy_coef = 0.001
    if args.init_noise_std is not None:
        if args.init_noise_std <= 0.0:
            raise ValueError("init noise std must be positive")
        runner_cfg.policy.init_noise_std = float(args.init_noise_std)
    if args.entropy_coef is not None:
        if args.entropy_coef < 0.0:
            raise ValueError("entropy coefficient must be non-negative")
        runner_cfg.algorithm.entropy_coef = float(args.entropy_coef)
    if args.gamma is not None:
        if not 0.0 < args.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        runner_cfg.algorithm.gamma = float(args.gamma)
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        runner_cfg.algorithm.learning_rate = float(args.learning_rate)
    if not 0.0 < args.adaptive_prefix_lr_factor <= 1.0:
        raise ValueError("adaptive prefix lr factor must be in (0, 1]")
    env = XHandEmbeddedEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(args.output), device=runner_cfg.device)
    start_iteration = 0
    resumed_iteration = None
    if args.resume_checkpoint is not None:
        runner.load(
            str(args.resume_checkpoint.resolve()),
            load_optimizer=not args.reset_optimizer_on_resume,
        )
        if args.init_noise_std is not None:
            _override_policy_noise(runner.alg.policy, float(args.init_noise_std))
        if args.freeze_policy_noise:
            _freeze_policy_noise(runner.alg.policy)
        resumed_iteration = resume_checkpoint_iteration(args.resume_checkpoint)
        start_iteration = resumed_iteration + 1
        runner.current_learning_iteration = start_iteration
    if args.resume_prefix_state is not None:
        if args.resume_checkpoint is None:
            raise RuntimeError("--resume-prefix-state requires --resume-checkpoint")
        prefix_state_path = args.resume_prefix_state.resolve()
        if not prefix_state_path.is_file():
            raise FileNotFoundError(prefix_state_path)
        # Validation of schema/object/shape is performed by the Isaac env after
        # construction.  Loading here also makes malformed or truncated state
        # fail before PPO starts and records a stable hash in provenance.
        prefix_state_payload = __import__("torch").load(
            prefix_state_path, map_location="cpu", weights_only=False
        )
        if not isinstance(prefix_state_payload, dict):
            raise RuntimeError("resume prefix state must contain a mapping")
        env.restore_training_prefix_state(prefix_state_path)
        restored_prefix_level = int(prefix_state_payload.get("prefix_level", 0))
        restored_prefix_schema = prefix_state_payload.get("schema")
    else:
        prefix_state_path = None
        restored_prefix_level = None
        restored_prefix_schema = None
    remaining = max(args.max_iterations - start_iteration, 0)
    output = args.output.resolve()
    if len(output.parents) < 3 or output.parent.parent.name != "runs":
        raise RuntimeError("training output must be ROOT/runs/OBJECT/ATTEMPT")
    if output.parent.name != args.object:
        raise RuntimeError("training output object directory does not match --object")
    generation_root = output.parents[2]
    reward_shaping_revision = (
        "strict_gate_aligned_v12_clearance_gated_force_closure"
        if cfg.force_closure_reward_requires_clearance
        else (
            "strict_gate_aligned_v11_surface_proximity_dense_force_closure"
            if cfg.disturbance_direction_reward_weight > 0.0
            else "strict_gate_aligned_v8_persistent_max_barrier"
        )
    )
    ppo_transition_revision = (
        "low_lr_strict_gate_alignment_v2"
        if args.learning_rate is not None
        and cfg.hard_gate_frontier_reward_weight > 0.0
        else (
            "low_lr_reward_transition_v1"
            if args.learning_rate is not None
            else "manifest_default"
        )
    )
    provenance = {
        "schema": TRAINING_RUN_SCHEMA,
        "backend": EMBEDDED_BACKEND,
        "object": args.object,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reward_revision": REWARD_REVISION,
        "reward_shaping_revision": reward_shaping_revision,
        "ppo_revision": PPO_REVISION,
        "training_reward_scale": TRAINING_REWARD_SCALE,
        "penetration_reward_weight": float(cfg.penetration_reward_weight),
        "penetration_clear_reward_weight": float(
            cfg.penetration_clear_reward_weight
        ),
        "proximity_reward_weight": float(cfg.proximity_reward_weight),
        "proximity_scale_m": float(cfg.proximity_scale_m),
        "terminal_success_weight": float(cfg.terminal_success_weight),
        "stable_lift_reward_weight": float(cfg.stable_lift_reward_weight),
        "force_closure_reward_weight": float(cfg.force_closure_reward_weight),
        "force_closure_shape_all_hold_contacts": bool(
            cfg.force_closure_shape_all_hold_contacts
        ),
        "force_closure_reward_requires_clearance": bool(
            cfg.force_closure_reward_requires_clearance
        ),
        "force_closure_shaping_revision": (
            "periodic_hold_wrench_quality_v1"
            if cfg.force_closure_shape_all_hold_contacts
            else "terminal_hold_only"
        ),
        "disturbance_reward_requires_force_closure": bool(
            cfg.disturbance_reward_requires_force_closure
        ),
        "hard_gate_frontier_reward_weight": float(
            cfg.hard_gate_frontier_reward_weight
        ),
        "disturbance_direction_reward_weight": float(
            cfg.disturbance_direction_reward_weight
        ),
        "gamma": float(runner_cfg.algorithm.gamma),
        "learning_rate": float(runner_cfg.algorithm.learning_rate),
        "learning_rate_override": args.learning_rate is not None,
        "adaptive_prefix_lr_factor": float(args.adaptive_prefix_lr_factor),
        "adaptive_prefix_lr_policy": "level_aware_floor_v1",
        "ppo_transition_revision": ppo_transition_revision,
        "nominal_dataset": str(args.nominal_dataset.resolve()),
        "nominal_dataset_sha256": sha256_file(args.nominal_dataset),
        "nominal_sample_index": args.nominal_sample_index,
        "nominal_legacy_warm_start": nominal_legacy_warm_start,
        "legacy_data_reused_as_accepted": nominal_legacy_reused_as_accepted,
        "warm_start_init_noise_std": float(runner_cfg.policy.init_noise_std),
        "warm_start_entropy_coef": float(runner_cfg.algorithm.entropy_coef),
        "arm_action_scale_rad": float(cfg.arm_action_scale_rad),
        "hand_action_scale_rad": float(cfg.hand_action_scale_rad),
        "instantaneous_residual_weight": float(cfg.instantaneous_residual_weight),
        "residual_integration": float(cfg.residual_integration),
        "residual_activation_phase": int(cfg.residual_activation_phase),
        "residual_limit_rad": float(cfg.residual_limit_rad),
        "approach_fraction": float(cfg.approach_fraction),
        "close_fraction": float(cfg.close_fraction),
        "lift_fraction": float(cfg.lift_fraction),
        "hold_fraction": float(cfg.hold_fraction),
        "disturbance_fraction": float(cfg.disturbance_fraction),
        "ablation_fraction": float(cfg.ablation_fraction),
        "reset_pose_mode": (
            "nominal_object_pose"
            if bool(cfg.use_nominal_object_pose_for_reset)
            else "canonical_table_pose"
        ),
        "hold_reference_revision": "learned_hold_posture_for_disturbances_v1",
        "reward_shaping_revision": reward_shaping_revision,
        "init_noise_std": float(runner_cfg.policy.init_noise_std),
        "entropy_coef": float(runner_cfg.algorithm.entropy_coef),
        "resume_policy_noise_override_applied": bool(
            args.resume_checkpoint is not None and args.init_noise_std is not None
        ),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_iterations": args.max_iterations,
        "device": cfg.sim.device,
        "kit_args": args.kit_args,
        "kit_portable_root": str(args.kit_portable_root),
        "kit_settings_profile": args.kit_settings_profile,
        "renderer_multi_gpu": args.multi_gpu,
        "resume_checkpoint_iteration": resumed_iteration,
        "resume_checkpoint": (
            str(args.resume_checkpoint.resolve())
            if args.resume_checkpoint is not None
            else None
        ),
        "resume_checkpoint_sha256": (
            sha256_file(args.resume_checkpoint)
            if args.resume_checkpoint is not None
            else None
        ),
        "resume_prefix_state": (
            str(prefix_state_path)
            if prefix_state_path is not None
            else None
        ),
        "resume_prefix_state_sha256": (
            sha256_file(prefix_state_path)
            if prefix_state_path is not None
            else None
        ),
        "resume_prefix_state_enabled": prefix_state_path is not None,
        "restored_prefix_level": restored_prefix_level,
        "restored_prefix_state_schema": restored_prefix_schema,
        "resume_optimizer_state_loaded": bool(
            args.resume_checkpoint is not None and not args.reset_optimizer_on_resume
        ),
        "start_iteration": start_iteration,
        "iterations_this_process": remaining,
        "rl_library": "rsl_rl",
        "rl_algorithm": "PPO",
        "physical_acceptance_location": "inside_environment_terminal_contract",
        "external_candidate_replay_validator": False,
        "exact_training_rollout_capture": True,
        "exact_checkpoint_timing": "before_ppo_update",
        "training_trajectory_recording": True,
        "strict_visual_mesh_accepted": False,
        "omitted_expensive_gates": manifest["omitted_expensive_gates"],
    }
    provenance_path = args.output / "training_provenance.json"
    if provenance_path.is_file():
        archived = args.output / f"training_provenance.before_resume_{time.time_ns()}.json"
        shutil.copy2(provenance_path, archived)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(args.manifest, args.output / "locked_embedded_manifest.json")
    if remaining:
        install_exact_success_capture(
            runner=runner,
            env=env,
            cfg=cfg,
            manifest=manifest,
            nominal=nominal,
            generation_root=generation_root,
            manifest_path=args.manifest.resolve(),
            output=args.output.resolve(),
            object_name=args.object,
            start_iteration=start_iteration,
            adaptive_prefix_lr_factor=args.adaptive_prefix_lr_factor,
        )
        # Every embedded gate depends on approach -> hold -> challenge order.
        # Random episode offsets would enter a disturbance without a valid hold
        # snapshot during the first rollout and corrupt the training signal.
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    if runner.writer is not None:
        runner.writer.flush()
    latest_checkpoint = latest_policy_checkpoint(args.output)
    if latest_checkpoint is None:
        raise RuntimeError("RSL-RL produced no v2 checkpoint")
    collection_checkpoint, checkpoint_selection = select_collection_checkpoint(args.output)
    provenance["completed_at"] = datetime.now(timezone.utc).isoformat()
    provenance["latest_checkpoint"] = str(latest_checkpoint.resolve())
    provenance["latest_checkpoint_sha256"] = sha256_file(latest_checkpoint)
    provenance["collection_checkpoint"] = str(collection_checkpoint.resolve())
    provenance["collection_checkpoint_sha256"] = sha256_file(collection_checkpoint)
    provenance["checkpoint_selection"] = checkpoint_selection
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "TRAINING_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "xhand_rl_embedded_training_complete_v2",
                "backend": EMBEDDED_BACKEND,
                "object": args.object,
                "reward_revision": REWARD_REVISION,
                "reward_shaping_revision": reward_shaping_revision,
                "ppo_revision": PPO_REVISION,
                "training_reward_scale": TRAINING_REWARD_SCALE,
                "penetration_reward_weight": float(cfg.penetration_reward_weight),
                "penetration_clear_reward_weight": float(
                    cfg.penetration_clear_reward_weight
                ),
                "proximity_reward_weight": float(cfg.proximity_reward_weight),
                "proximity_scale_m": float(cfg.proximity_scale_m),
                "terminal_success_weight": float(cfg.terminal_success_weight),
                "stable_lift_reward_weight": float(cfg.stable_lift_reward_weight),
                "force_closure_reward_weight": float(cfg.force_closure_reward_weight),
                "force_closure_shape_all_hold_contacts": bool(
                    cfg.force_closure_shape_all_hold_contacts
                ),
                "force_closure_reward_requires_clearance": bool(
                    cfg.force_closure_reward_requires_clearance
                ),
                "disturbance_reward_requires_force_closure": bool(
                    cfg.disturbance_reward_requires_force_closure
                ),
                "hard_gate_frontier_reward_weight": float(
                    cfg.hard_gate_frontier_reward_weight
                ),
                "disturbance_direction_reward_weight": float(
                    cfg.disturbance_direction_reward_weight
                ),
                "gamma": float(runner_cfg.algorithm.gamma),
                "learning_rate": float(runner_cfg.algorithm.learning_rate),
                "learning_rate_override": args.learning_rate is not None,
                "adaptive_prefix_lr_factor": float(args.adaptive_prefix_lr_factor),
                "adaptive_prefix_lr_policy": "level_aware_floor_v1",
                "ppo_transition_revision": ppo_transition_revision,
                "arm_action_scale_rad": float(cfg.arm_action_scale_rad),
                "hand_action_scale_rad": float(cfg.hand_action_scale_rad),
                "instantaneous_residual_weight": float(cfg.instantaneous_residual_weight),
                "residual_integration": float(cfg.residual_integration),
                "residual_activation_phase": int(cfg.residual_activation_phase),
                "residual_limit_rad": float(cfg.residual_limit_rad),
                "reset_pose_mode": (
                    "nominal_object_pose"
                    if bool(cfg.use_nominal_object_pose_for_reset)
                    else "canonical_table_pose"
                ),
                "hold_reference_revision": "learned_hold_posture_for_disturbances_v1",
                "reward_shaping_revision": reward_shaping_revision,
                "resume_optimizer_state_loaded": bool(
                    args.resume_checkpoint is not None
                    and not args.reset_optimizer_on_resume
                ),
                "checkpoint": provenance["collection_checkpoint"],
                "checkpoint_sha256": provenance["collection_checkpoint_sha256"],
                "latest_checkpoint": provenance["latest_checkpoint"],
                "latest_checkpoint_sha256": provenance["latest_checkpoint_sha256"],
                "checkpoint_selection": checkpoint_selection,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
