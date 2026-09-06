#!/usr/bin/env python3
"""Train one chunk of one curriculum stage; never promotes stages itself."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--bodex-bank", type=Path, required=True)
parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument(
    "--save-interval",
    type=int,
    default=25,
    help="save a policy checkpoint every N PPO iterations",
)
parser.add_argument(
    "--initialize-only",
    action="store_true",
    help="write the zero-residual BODex policy checkpoint without a PPO update",
)
parser.add_argument("--seed", type=int, default=20260829)
parser.add_argument("--resume-checkpoint", type=Path)
parser.add_argument("--load-optimizer", action="store_true")
parser.add_argument(
    "--fixed-candidate-index",
    type=int,
    default=-1,
    help=(
        "pin every reset to one zero-based BODex candidate; -1 keeps the "
        "default round-robin candidate schedule"
    ),
)
parser.add_argument(
    "--nominal-pose-lock",
    action="store_true",
    help="disable XY/yaw/joint reset noise for a fixed-initial-pose ablation",
)
parser.add_argument("--stage-contact-groups-per-side-min", type=int)
parser.add_argument("--stage-contact-groups-total-min", type=int)
parser.add_argument("--stage-penetration-max-m", type=float)
parser.add_argument("--stage-stable-hold-steps", type=int)
parser.add_argument("--stage-stability-height-tolerance-m", type=float)
parser.add_argument(
    "--stage-penetration-gate-mode",
    choices=("historical_max", "current"),
)
parser.add_argument(
    "--stage-stability-gate-mode",
    choices=("current", "historical_max"),
)
parser.add_argument(
    "--palm-center-alignment-reward-weight",
    type=float,
    default=None,
    help="optional BODex fixed-pose palm-center alignment shaping weight",
)
parser.add_argument(
    "--active-action-group-override",
    choices=("hands", "distal_wrist", "hands_and_distal_arms", "all"),
    help="conservative residual action mask override without changing formal stage gates",
)
parser.add_argument(
    "--residual-activation-phase",
    choices=("approach", "close", "hold"),
)
parser.add_argument("--residual-activation-close-fraction", type=float)
parser.add_argument(
    "--hand-residual-activation-phase",
    choices=("approach", "close", "hold"),
    help=(
        "optional later activation schedule for enabled hand rows; wrist rows "
        "continue to use --residual-activation-phase"
    ),
)
parser.add_argument("--hand-residual-activation-close-fraction", type=float)
parser.add_argument("--residual-integration", type=float)
parser.add_argument("--residual-limit-rad", type=float)
parser.add_argument("--reset-policy-noise-after-load", action="store_true")
parser.add_argument(
    "--actor-output-head-only",
    action="store_true",
    help="freeze the actor backbone and train only selected active output rows",
)
parser.add_argument(
    "--trainable-action-names",
    help="comma-separated actor output rows; defaults to the effective action mask",
)
parser.add_argument(
    "--candidate-context-only-actor",
    action="store_true",
    help=(
        "freeze the shared residual actor and train only the appended BODex "
        "candidate-context input columns; requires a resumed checkpoint and "
        "fresh optimizer state"
    ),
)
parser.add_argument(
    "--gate-memory-context-only-actor",
    action="store_true",
    help=(
        "freeze the shared residual actor and train only the four gate-memory "
        "inputs plus selected BODex candidate-context columns"
    ),
)
parser.add_argument(
    "--candidate-gated-memory-only-actor",
    action="store_true",
    help=(
        "freeze the existing actor and train only candidate-specific "
        "gate-memory interaction columns"
    ),
)
parser.add_argument(
    "--candidate-context-train-indices",
    help=(
        "comma-separated zero-based BODex candidate columns to update in "
        "candidate-context-only actor mode; defaults to every candidate"
    ),
)
parser.add_argument(
    "--zero-output-action-names-after-load",
    help=(
        "comma-separated joints whose actor output rows are newly unmasked; "
        "valid only with --resume-checkpoint and without --load-optimizer"
    ),
)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--init-noise-std", type=float)
parser.add_argument("--entropy-coef", type=float)
parser.add_argument(
    "--freeze-policy-noise",
    action=argparse.BooleanOptionalAction,
    default=None,
)
parser.add_argument("--bilateral-contact-reward-weight", type=float)
parser.add_argument("--terminal-success-weight", type=float)
parser.add_argument("--penetration-reward-weight", type=float)
parser.add_argument("--penetration-clear-reward-weight", type=float)
parser.add_argument("--proximity-reward-weight", type=float)
parser.add_argument("--contact-diversity-reward-weight", type=float)
parser.add_argument("--contact-continuity-reward-weight", type=float)
parser.add_argument("--stable-lift-reward-weight", type=float)
parser.add_argument("--stage-stability-reward-weight", type=float)
parser.add_argument("--stage-lift-hold-reward-weight", type=float)
parser.add_argument("--stage-gate-progress-reward-weight", type=float)
parser.add_argument("--stage-action-anchor-penalty-weight", type=float)
parser.add_argument("--stage-residual-anchor-penalty-weight", type=float)
parser.add_argument("--distributed-contact-reward-weight", type=float)
parser.add_argument("--terminal-contact-reward-weight", type=float)
parser.add_argument(
    "--terminal-contact-progress-mode",
    choices=("bilateral", "joint_gate"),
    default="bilateral",
)
parser.add_argument(
    "--distributed-reward-requires-current-bilateral-contact",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--stage-stability-requires-current-gate-progress",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--stage-lift-hold-min-stage", type=int, default=4)
parser.add_argument(
    "--terminate-on-stage-penetration-failure",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--candidate-repeat-factors",
    help="comma-separated positive repeats for deterministic weighted BODex sampling",
)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"staged_train_s{args.stage}_{args.object}")
simulation_app = AppLauncher(args).app

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_bodex_bimanual.contracts import load_bodex_bank, sha256_file
from migration_4090.xhand_rl_embedded.contracts import latest_policy_checkpoint, validate_manifest
from migration_4090.xhand_rl_embedded.env import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_HOLD,
)

from .configuration import configure_staged_env
from .contracts import STAGED_BACKEND, STAGED_TRAINING_SCHEMA
from .env import XHandStagedEnv, XHandStagedEnvCfg
from .policy_noise import (
    freeze_policy_noise,
    policy_noise_std,
    reset_policy_noise_std,
    restrict_actor_updates_to_action_rows,
    restrict_actor_updates_to_candidate_context,
    restrict_actor_updates_to_candidate_gated_memory,
    zero_initialize_residual_actor,
    zero_initialize_residual_actor_rows,
)
from .ppo_cfg import XHandStagedPPORunnerCfg
from .reward_profiles import (
    REWARD_WEIGHT_FIELDS,
    stage_entropy_coef,
    stage_freeze_policy_noise,
    stage_init_noise_std,
    stage_learning_rate,
)
from .sampling import parse_candidate_repeat_factors
from .stages import get_stage


def main() -> None:
    if args.num_envs <= 0 or (not args.initialize_only and args.iterations <= 0):
        raise ValueError("num-envs and training iterations must be positive")
    if args.save_interval <= 0:
        raise ValueError("save-interval must be positive")
    if args.initialize_only and args.resume_checkpoint is not None:
        raise ValueError("initialize-only mode cannot resume a checkpoint")
    effective_iterations = 0 if args.initialize_only else args.iterations
    init_noise_std = (
        float(args.init_noise_std)
        if args.init_noise_std is not None
        else stage_init_noise_std(args.stage)
    )
    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else stage_learning_rate(args.stage)
    )
    if learning_rate <= 0.0 or init_noise_std <= 0.0:
        raise ValueError("learning rate and policy noise must be positive")
    entropy_coef = (
        float(args.entropy_coef)
        if args.entropy_coef is not None
        else stage_entropy_coef(args.stage)
    )
    if entropy_coef < 0.0:
        raise ValueError("entropy coefficient must be non-negative")
    freeze_noise = (
        bool(args.freeze_policy_noise)
        if args.freeze_policy_noise is not None
        else stage_freeze_policy_noise(args.stage)
    )
    if args.reset_policy_noise_after_load and args.resume_checkpoint is None:
        raise ValueError("policy noise can only be reset after loading a checkpoint")
    adapter_modes = (
        args.candidate_context_only_actor,
        args.gate_memory_context_only_actor,
        args.candidate_gated_memory_only_actor,
    )
    if sum(bool(enabled) for enabled in adapter_modes) > 1:
        raise ValueError("actor observation-adapter modes are mutually exclusive")
    adapter_only_actor = any(adapter_modes)
    if args.actor_output_head_only and adapter_only_actor:
        raise ValueError("actor output-head and observation-adapter modes are mutually exclusive")
    if args.actor_output_head_only and args.resume_checkpoint is None:
        raise ValueError("output-head-only actor training requires a resumed checkpoint")
    if args.actor_output_head_only and args.load_optimizer:
        raise ValueError("output-head-only actor training requires fresh optimizer state")
    if adapter_only_actor and args.resume_checkpoint is None:
        raise ValueError("observation-adapter-only actor training requires a checkpoint")
    if adapter_only_actor and args.load_optimizer:
        raise ValueError(
            "observation-adapter-only actor training requires fresh optimizer state"
        )
    if args.candidate_context_train_indices and not adapter_only_actor:
        raise ValueError(
            "candidate context train indices require an observation-adapter actor mode"
        )
    zero_output_action_names = tuple(
        name.strip()
        for name in (args.zero_output_action_names_after_load or "").split(",")
        if name.strip()
    )
    if zero_output_action_names and args.resume_checkpoint is None:
        raise ValueError("actor output rows can only be reset after loading a checkpoint")
    if zero_output_action_names and args.load_optimizer:
        raise ValueError("cannot load optimizer state while zeroing newly active actor rows")
    trainable_action_names = tuple(
        name.strip()
        for name in (args.trainable_action_names or "").split(",")
        if name.strip()
    )
    if trainable_action_names and not args.actor_output_head_only:
        raise ValueError("trainable action names require --actor-output-head-only")
    if args.residual_activation_close_fraction is not None and not (
        0.0 <= args.residual_activation_close_fraction <= 1.0
    ):
        raise ValueError("residual activation close fraction must be in [0, 1]")
    if args.residual_integration is not None and args.residual_integration <= 0.0:
        raise ValueError("residual integration must be positive")
    if args.residual_limit_rad is not None and args.residual_limit_rad <= 0.0:
        raise ValueError("residual limit must be positive")
    if not 1 <= int(args.stage_lift_hold_min_stage) <= 6:
        raise ValueError("stage lift-hold minimum stage must be in [1, 6]")
    reward_overrides = {
        name: float(value)
        for name, value in {
            "bilateral_contact_reward_weight": args.bilateral_contact_reward_weight,
            "terminal_success_weight": args.terminal_success_weight,
            "penetration_reward_weight": args.penetration_reward_weight,
            "penetration_clear_reward_weight": args.penetration_clear_reward_weight,
            "proximity_reward_weight": args.proximity_reward_weight,
            "contact_diversity_reward_weight": args.contact_diversity_reward_weight,
            "contact_continuity_reward_weight": args.contact_continuity_reward_weight,
            "stable_lift_reward_weight": args.stable_lift_reward_weight,
            "stage_gate_progress_reward_weight": args.stage_gate_progress_reward_weight,
            "stage_action_anchor_penalty_weight": (
                args.stage_action_anchor_penalty_weight
            ),
            "stage_residual_anchor_penalty_weight": (
                args.stage_residual_anchor_penalty_weight
            ),
            "distributed_contact_reward_weight": (
                args.distributed_contact_reward_weight
            ),
            "terminal_contact_reward_weight": args.terminal_contact_reward_weight,
        }.items()
        if value is not None
    }
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    # Production training deliberately has no allow-test-fixture escape hatch.
    bank = load_bodex_bank(
        args.bodex_bank,
        expected_object=args.object,
        verify_source=False,
        intended_stage=args.stage,
    )
    candidate_repeat_factors = parse_candidate_repeat_factors(
        args.candidate_repeat_factors, candidate_count=len(bank["samples"])
    )
    candidate_context_train_indices = None
    if args.candidate_context_train_indices:
        try:
            candidate_context_train_indices = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in args.candidate_context_train_indices.split(",")
                        if value.strip()
                    }
                )
            )
        except ValueError as exc:
            raise ValueError(
                "candidate context train indices must be comma-separated integers"
            ) from exc
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = configure_staged_env(
        XHandStagedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        bodex_bank=args.bodex_bank,
        stage_id=args.stage,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=False,
        allow_test_bank=False,
        # ``fixed_candidate_index`` takes precedence inside the environment;
        # retain the round-robin label for the unfixed fallback schedule.
        candidate_selection="round_robin",
        candidate_repeat_factors=candidate_repeat_factors,
        fixed_candidate_index=args.fixed_candidate_index,
        nominal_pose_lock=bool(args.nominal_pose_lock),
        palm_center_alignment_reward_weight=(
            args.palm_center_alignment_reward_weight
        ),
        reward_overrides=reward_overrides,
        terminal_contact_progress_mode=args.terminal_contact_progress_mode,
        distributed_reward_requires_current_bilateral_contact=(
            args.distributed_reward_requires_current_bilateral_contact
        ),
        stage_contact_groups_per_side_min=(
            args.stage_contact_groups_per_side_min
        ),
        stage_contact_groups_total_min=args.stage_contact_groups_total_min,
        stage_penetration_max_m=args.stage_penetration_max_m,
        stage_stable_hold_steps=args.stage_stable_hold_steps,
        stage_stability_height_tolerance_m=(
            args.stage_stability_height_tolerance_m
        ),
        stage_penetration_gate_mode=args.stage_penetration_gate_mode,
        stage_stability_gate_mode=args.stage_stability_gate_mode,
    )
    if args.active_action_group_override is not None:
        cfg.active_action_group_override = args.active_action_group_override
    if args.residual_activation_phase is not None:
        cfg.residual_activation_phase = {
            "approach": PHASE_APPROACH,
            "close": PHASE_CLOSE,
            "hold": PHASE_HOLD,
        }[args.residual_activation_phase]
    if args.residual_activation_close_fraction is not None:
        cfg.residual_activation_close_fraction = float(
            args.residual_activation_close_fraction
        )
    if args.hand_residual_activation_phase is not None:
        cfg.hand_residual_activation_phase = {
            "approach": PHASE_APPROACH,
            "close": PHASE_CLOSE,
            "hold": PHASE_HOLD,
        }[args.hand_residual_activation_phase]
    if args.hand_residual_activation_close_fraction is not None:
        cfg.hand_residual_activation_close_fraction = float(
            args.hand_residual_activation_close_fraction
        )
    if args.residual_integration is not None:
        cfg.residual_integration = float(args.residual_integration)
    if args.residual_limit_rad is not None:
        cfg.residual_limit_rad = float(args.residual_limit_rad)
    if args.stage_stability_reward_weight is not None:
        cfg.stage_stability_reward_weight = float(args.stage_stability_reward_weight)
    if args.stage_lift_hold_reward_weight is not None:
        cfg.stage_lift_hold_reward_weight = float(args.stage_lift_hold_reward_weight)
    cfg.stage_stability_requires_current_gate_progress = bool(
        args.stage_stability_requires_current_gate_progress
    )
    cfg.stage_lift_hold_min_stage = int(args.stage_lift_hold_min_stage)
    cfg.terminate_on_stage_penetration_failure = bool(
        args.terminate_on_stage_penetration_failure
    )
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    runner_cfg.max_iterations = max(effective_iterations, 1)
    runner_cfg.save_interval = int(args.save_interval)
    runner_cfg.run_name = f"{args.object}_stage_{args.stage}"
    runner_cfg.algorithm.learning_rate = learning_rate
    runner_cfg.algorithm.entropy_coef = entropy_coef
    runner_cfg.policy.init_noise_std = init_noise_std
    env = XHandStagedEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=str(output), device=runner_cfg.device
    )
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise FileNotFoundError(args.resume_checkpoint)
        runner.load(str(args.resume_checkpoint.resolve()), load_optimizer=args.load_optimizer)
        if args.reset_policy_noise_after_load:
            reset_policy_noise_std(runner.alg.policy, init_noise_std)
        unknown_action_names = sorted(set(zero_output_action_names) - set(env.joint_names))
        if unknown_action_names:
            raise ValueError(f"unknown action names for actor-row reset: {unknown_action_names}")
        zeroed_output_action_indices = zero_initialize_residual_actor_rows(
            runner.alg.policy,
            [env.joint_names.index(name) for name in zero_output_action_names],
        )
        zero_initialized_residual_actor = False
    else:
        zero_initialize_residual_actor(runner.alg.policy)
        zeroed_output_action_indices = tuple(range(cfg.action_space))
        zero_initialized_residual_actor = True
    if freeze_noise:
        freeze_policy_noise(runner.alg.policy)
    actor_output_head_update = None
    if args.actor_output_head_only:
        unknown_trainable_names = sorted(
            set(trainable_action_names) - set(env.joint_names)
        )
        if unknown_trainable_names:
            raise ValueError(
                f"unknown trainable action names: {unknown_trainable_names}"
            )
        active_action_names = tuple(
            name
            for name, enabled in zip(env.joint_names, env.action_mask[0].tolist())
            if enabled > 0.5
        )
        selected_names = trainable_action_names or active_action_names
        inactive_trainable_names = sorted(set(selected_names) - set(active_action_names))
        if inactive_trainable_names:
            raise ValueError(
                "trainable action names must be enabled by the effective action mask: "
                f"{inactive_trainable_names}"
            )
        actor_output_head_update = restrict_actor_updates_to_action_rows(
            runner.alg.policy,
            [env.joint_names.index(name) for name in selected_names],
        )
    candidate_context_actor_update = None
    if adapter_only_actor:
        if args.candidate_gated_memory_only_actor:
            candidate_context_actor_update = (
                restrict_actor_updates_to_candidate_gated_memory(
                    runner.alg.policy,
                    context_width=len(bank["samples"]),
                    gate_memory_width=4,
                    trainable_candidate_indices=candidate_context_train_indices,
                )
            )
        else:
            candidate_context_actor_update = restrict_actor_updates_to_candidate_context(
                runner.alg.policy,
                len(bank["samples"]),
                trainable_context_indices=candidate_context_train_indices,
                gate_memory_width=(4 if args.gate_memory_context_only_actor else 0),
                intervening_width=(16 if args.gate_memory_context_only_actor else 0),
            )
    effective_initial_policy_noise_std = policy_noise_std(runner.alg.policy)
    attempt_index = len(list(output.glob("training_attempt_*.json")))
    provenance_path = output / f"training_attempt_{attempt_index:06d}.json"
    provenance = {
        "schema": STAGED_TRAINING_SCHEMA,
        "backend": STAGED_BACKEND,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "stage": env.stage_spec.to_dict(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "bodex_bank": str(args.bodex_bank.resolve()),
        "bodex_bank_sha256": sha256_file(args.bodex_bank),
        "bodex_bank_schema": bank["schema"],
        "bodex_stage_seed_profile": bank["stage_seed_profile"],
        "bodex_supported_curriculum_stages": bank[
            "supported_curriculum_stages"
        ],
        "bodex_candidate_count": len(bank["samples"]),
        "candidate_selection": (
            "fixed" if int(cfg.fixed_candidate_index) >= 0 else cfg.candidate_selection
        ),
        "fixed_candidate_index": int(cfg.fixed_candidate_index),
        "nominal_pose_lock": bool(cfg.nominal_pose_lock),
        "palm_center_alignment_reward_weight": float(
            cfg.palm_center_alignment_reward_weight
        ),
        "candidate_repeat_factors": list(env.candidate_repeat_factors),
        "observation_dimension": cfg.observation_space,
        "action_dimension": cfg.action_space,
        "formal_active_action_group": get_stage(args.stage).active_action_group,
        "effective_active_action_group": env.effective_active_action_group,
        "active_action_names": [
            name
            for name, enabled in zip(env.joint_names, env.action_mask[0].tolist())
            if enabled > 0.5
        ],
        "distributed_reward_requires_current_bilateral_contact": bool(
            cfg.distributed_reward_requires_current_bilateral_contact
        ),
        "terminal_contact_progress_semantics": (
            args.terminal_contact_progress_mode
        ),
        "num_envs": args.num_envs,
        "iterations": effective_iterations,
        "save_interval": int(args.save_interval),
        "initialize_only": bool(args.initialize_only),
        "seed": args.seed,
        "device": cfg.sim.device,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "init_noise_std": init_noise_std,
        "policy_noise_frozen": freeze_noise,
        "actor_output_head_only": bool(args.actor_output_head_only),
        "actor_output_head_update": actor_output_head_update,
        "candidate_context_only_actor": bool(args.candidate_context_only_actor),
        "gate_memory_context_only_actor": bool(
            args.gate_memory_context_only_actor
        ),
        "candidate_gated_memory_only_actor": bool(
            args.candidate_gated_memory_only_actor
        ),
        "candidate_context_actor_update": candidate_context_actor_update,
        "zero_initialized_residual_actor": zero_initialized_residual_actor,
        "zeroed_output_action_names_after_load": list(zero_output_action_names),
        "zeroed_output_action_indices_after_load": list(zeroed_output_action_indices),
        "reset_policy_noise_after_load": bool(args.reset_policy_noise_after_load),
        "effective_initial_policy_noise_std": effective_initial_policy_noise_std,
        "reward_profile": {
            name: float(getattr(cfg, name)) for name in REWARD_WEIGHT_FIELDS
        },
        "training_reward_scale": float(cfg.training_reward_scale),
        "stage_stability_reward_weight": float(cfg.stage_stability_reward_weight),
        "stage_lift_hold_reward_weight": float(cfg.stage_lift_hold_reward_weight),
        "stage_stability_requires_current_gate_progress": bool(
            cfg.stage_stability_requires_current_gate_progress
        ),
        "stage_lift_hold_min_stage": int(cfg.stage_lift_hold_min_stage),
        "terminate_on_stage_penetration_failure": bool(
            cfg.terminate_on_stage_penetration_failure
        ),
        "stage_penetration_gate_m": float(cfg.physx_penetration_max_m),
        "residual_activation_phase": int(cfg.residual_activation_phase),
        "residual_activation_close_fraction": float(
            cfg.residual_activation_close_fraction
        ),
        "hand_residual_activation_phase": int(
            cfg.hand_residual_activation_phase
        ),
        "hand_residual_activation_close_fraction": float(
            cfg.hand_residual_activation_close_fraction
        ),
        "residual_integration": float(cfg.residual_integration),
        "residual_limit_rad": float(cfg.residual_limit_rad),
        "resume_checkpoint": (
            str(args.resume_checkpoint.resolve()) if args.resume_checkpoint is not None else None
        ),
        "resume_checkpoint_sha256": (
            sha256_file(args.resume_checkpoint) if args.resume_checkpoint is not None else None
        ),
        "load_optimizer": bool(args.load_optimizer),
        "stage_promoted": False,
        "fixed_evaluation_pending": True,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.initialize_only:
        # RSL-RL normally creates this attribute in ``learn()`` while
        # preparing its writer.  Initialization-only checkpoints deliberately
        # skip learning, so establish the non-network logger type explicitly
        # before calling the canonical checkpoint writer.
        runner.logger_type = runner_cfg.logger
        runner.save(str(output / "model_0.pt"))
    else:
        runner.learn(
            num_learning_iterations=effective_iterations,
            init_at_random_ep_len=False,
        )
    if runner.writer is not None:
        runner.writer.flush()
    checkpoint = latest_policy_checkpoint(output)
    if checkpoint is None:
        raise RuntimeError("RSL-RL produced no staged checkpoint")
    provenance["completed_at"] = datetime.now(timezone.utc).isoformat()
    provenance["latest_checkpoint"] = str(checkpoint.resolve())
    provenance["latest_checkpoint_sha256"] = sha256_file(checkpoint)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "latest_training_result.json").write_text(
        json.dumps(
            {
                "schema": "xhand_rl_staged_training_chunk_result_v1",
                "object": args.object,
                "stage": args.stage,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "promotion_pending": True,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
