#!/usr/bin/env python3
"""Train one non-promotional Stage-2.5 controlled-lift bridge chunk."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--bodex-bank", type=Path, required=True)
parser.add_argument("--lift-targets", type=Path, required=True)
parser.add_argument(
    "--retracted-pregrasp",
    type=Path,
    help=(
        "per-candidate retracted start pose; without it the approach phase "
        "holds the pregrasp pose and the hands start 1-7 mm off the object"
    ),
)
parser.add_argument("--retract-distance-m", type=float, default=0.10)
parser.add_argument("--profile", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=14)
parser.add_argument("--seed", type=int, default=20262331)
parser.add_argument("--learning-rate", type=float, default=5.0e-5)
parser.add_argument("--init-noise-std", type=float, default=0.12)
parser.add_argument("--entropy-coef", type=float, default=0.001)
parser.add_argument("--candidate-repeat-factors")
parser.add_argument(
    "--trainable-candidate-indices",
    help=(
        "comma-separated candidate indices; valid only with "
        "a candidate-conditioned actor update scope"
    ),
)
parser.add_argument(
    "--actor-update-scope",
    choices=(
        "output_head",
        "candidate_gated_memory",
        "candidate_gated_action_rows",
        "full",
    ),
    default="output_head",
)
parser.add_argument(
    "--trainable-action-names",
    help=(
        "comma-separated joint names for output-head updates; defaults to all "
        "actions enabled by the selected bridge profile"
    ),
)
parser.add_argument(
    "--zero-output-action-names-after-load",
    help=(
        "comma-separated newly enabled joint rows to zero after loading the "
        "resume checkpoint; requires a fresh optimizer state"
    ),
)
# The lift bridge builds its environment with stage_id=2 (contact_continuity),
# whose reward profile leaves lift_height, stable_lift and force_closure at 0.0 --
# only stages 4-6 turn them on (reward_profiles.py:102) -- and the overrides below
# also zeroed terminal_success.  Measured 2026-09-07 across two training lines and
# 340 iterations, those four tensorboard series are identically 0.000: nothing in
# the objective ever paid for lifting.  The policy could only earn contact-shaping
# reward, whose cheapest source is squeezing harder, and every resulting checkpoint
# lost to a zero action vector on stable_terminal_state and sustained_micro_lift.
#
# These expose the weights instead of hardcoding them.  Defaults match stage 4/5 so
# the bridge actually optimises the thing it is evaluated on; pass 0.0 to reproduce
# the historical runs.
# bridge_evaluate and bridge_capture_visualization gained these on 2026-09-08
# and bridge_train did not, so the 80 mm runs trained at the stock 1500/120 and
# 50/2 while their own zero-action baseline ran at 800/60 and 4/2.  A policy
# compared across that gap cannot be read: a loss says nothing about whether the
# learning failed or the environment differed.  Same names and defaults as the
# evaluator so a recipe can be copied between them verbatim.
parser.add_argument("--finger-close-scale", type=float, default=1.0)
parser.add_argument("--no-self-collisions", action="store_true")
parser.add_argument("--object-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--arm-stiffness", type=float, default=None)
parser.add_argument("--arm-damping", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--hand-damping", type=float, default=None)
parser.add_argument("--lift-height-reward-weight", type=float, default=14.0)
parser.add_argument("--stable-lift-reward-weight", type=float, default=8.0)
parser.add_argument("--terminal-success-weight", type=float, default=30.0)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"lift_bridge_train_{args.object}_{args.profile}")
simulation_app = AppLauncher(args).app

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_bodex_bimanual.contracts import load_bodex_bank, sha256_file
from migration_4090.xhand_rl_embedded.contracts import (
    latest_policy_checkpoint,
    validate_manifest,
)
from migration_4090.xhand_rl_embedded.env import PHASE_APPROACH, PHASE_CLOSE, PHASE_HOLD, PHASE_LIFT

from .bridge_env import XHandLiftBridgeEnv
from .bridge_profiles import get_bridge_profile
from .configuration import configure_staged_env
from .env import XHandStagedEnvCfg
from .policy_noise import (
    freeze_policy_noise,
    restrict_actor_updates_to_candidate_gated_action_rows,
    reset_policy_noise_std,
    restrict_actor_updates_to_action_rows,
    restrict_actor_updates_to_candidate_gated_memory,
    zero_initialize_residual_actor_rows,
)
from .ppo_cfg import XHandStagedPPORunnerCfg
from .sampling import parse_candidate_indices, parse_candidate_repeat_factors


def main() -> None:
    if args.num_envs <= 0 or args.iterations <= 0:
        raise ValueError("num-envs and iterations must be positive")
    if min(args.learning_rate, args.init_noise_std) <= 0.0:
        raise ValueError("learning rate and policy noise must be positive")
    if args.entropy_coef < 0.0:
        raise ValueError("entropy coefficient must be non-negative")
    for path in (args.checkpoint, args.lift_targets, args.bodex_bank):
        if not path.is_file():
            raise FileNotFoundError(path)
    profile = get_bridge_profile(args.profile)
    effective_candidate_selection = (
        "fixed" if profile.fixed_candidate_index is not None else "round_robin"
    )
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    bank = load_bodex_bank(
        args.bodex_bank,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    candidate_repeat_factors = parse_candidate_repeat_factors(
        args.candidate_repeat_factors,
        candidate_count=len(bank["samples"]),
    )
    trainable_candidate_indices = parse_candidate_indices(
        args.trainable_candidate_indices,
        candidate_count=len(bank["samples"]),
    )
    if (
        trainable_candidate_indices is not None
        and args.actor_update_scope
        not in ("candidate_gated_memory", "candidate_gated_action_rows")
    ):
        raise ValueError(
            "trainable candidate indices require a candidate-conditioned scope"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = configure_staged_env(
        XHandStagedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        bodex_bank=args.bodex_bank,
        stage_id=2,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=False,
        allow_test_bank=False,
        candidate_selection="round_robin",
        candidate_repeat_factors=candidate_repeat_factors,
        fixed_candidate_index=(
            int(profile.fixed_candidate_index)
            if profile.fixed_candidate_index is not None
            else -1
        ),
        nominal_pose_lock=bool(profile.nominal_pose_lock),
        palm_center_alignment_reward_weight=(
            float(profile.palm_center_alignment_reward_weight)
        ),
        terminal_contact_progress_mode="joint_gate",
        distributed_reward_requires_current_bilateral_contact=True,
        reward_overrides={
            "lift_height_reward_weight": float(args.lift_height_reward_weight),
            "stable_lift_reward_weight": float(args.stable_lift_reward_weight),
            "terminal_success_weight": float(args.terminal_success_weight),
            "stage_gate_progress_reward_weight": 20.0,
            "stage_action_anchor_penalty_weight": 1.0,
            "stage_residual_anchor_penalty_weight": 8.0,
            "distributed_contact_reward_weight": 10.0,
            "terminal_contact_reward_weight": 20.0,
        },
    )
    cfg.episode_length_s = profile.episode_length_s
    (
        cfg.approach_fraction,
        cfg.close_fraction,
        cfg.lift_fraction,
        cfg.hold_fraction,
    ) = profile.phase_fractions
    cfg.residual_activation_phase = (
        PHASE_APPROACH
        if profile.residual_activation == "approach"
        else PHASE_CLOSE
        if profile.residual_activation == "close"
        else PHASE_HOLD
        if profile.residual_activation == "hold"
        else PHASE_LIFT
    )
    cfg.residual_activation_close_fraction = (
        profile.residual_activation_close_fraction
    )
    # profile.action_group had never been wired to anything: bridge_train,
    # bridge_evaluate and bridge_capture_visualization all built the environment
    # and then set the residual fields from the profile without ever touching
    # active_action_group_override, so every lift-bridge run used stage 2's
    # "hands" no matter what its profile declared.  The five profiles declaring
    # "distal_wrist" were silently running on hand joints alone.
    for _field, _value in (
        ("robot_self_collisions", False if args.no_self_collisions else None),
        ("object_max_depenetration_velocity", args.object_max_depenetration_velocity),
        ("arm_stiffness", args.arm_stiffness),
        ("arm_damping", args.arm_damping),
        ("hand_stiffness", args.hand_stiffness),
        ("hand_damping", args.hand_damping),
    ):
        if _value is not None:
            setattr(cfg, _field, float(_value))
    cfg.active_action_group_override = str(profile.action_group)
    cfg.residual_integration = profile.residual_integration
    cfg.residual_limit_rad = profile.residual_limit_rad
    cfg.arm_approach_fraction_of_close = (
        profile.arm_approach_fraction_of_close
    )
    cfg.finger_close_end_fraction_of_close = (
        profile.finger_close_end_fraction_of_close
    )
    cfg.residual_integration_end_fraction_of_hold = (
        profile.residual_integration_end_fraction_of_hold
    )
    cfg.reset_xy_noise_m = profile.reset_xy_noise_m
    cfg.reset_yaw_noise_rad = profile.reset_yaw_noise_rad
    cfg.reset_joint_noise_rad = profile.reset_joint_noise_rad
    cfg.terminate_on_object_escape = False
    cfg.bridge_terminate_on_unsafe = True
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    runner_cfg.max_iterations = args.iterations
    # Bridge experiments are short and can regress sharply after the first
    # useful gated-hold updates.  Keep every iteration checkpoint so model
    # selection can use the same multi-candidate evaluation contract instead
    # of being forced to accept only the final PPO iterate.
    runner_cfg.save_interval = 1
    runner_cfg.run_name = f"{args.object}_{profile.name}"
    runner_cfg.algorithm.learning_rate = args.learning_rate
    runner_cfg.algorithm.entropy_coef = args.entropy_coef
    runner_cfg.policy.init_noise_std = args.init_noise_std
    env = XHandLiftBridgeEnv(
        cfg,
        lift_targets=args.lift_targets,
        retracted_pregrasp=args.retracted_pregrasp,
        retract_distance_m=args.retract_distance_m,
        finger_close_scale=args.finger_close_scale,
        source_bodex_bank=args.bodex_bank,
        profile=profile,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped,
        runner_cfg.to_dict(),
        log_dir=str(output),
        device=runner_cfg.device,
    )
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    reset_policy_noise_std(runner.alg.policy, args.init_noise_std)
    freeze_policy_noise(runner.alg.policy)
    zero_output_action_names = tuple(
        name.strip()
        for name in (args.zero_output_action_names_after_load or "").split(",")
        if name.strip()
    )
    unknown_zero_names = sorted(
        set(zero_output_action_names) - set(env.joint_names)
    )
    if unknown_zero_names:
        raise ValueError(
            f"unknown action names for actor-row reset: {unknown_zero_names}"
        )
    zeroed_output_action_indices = ()
    if zero_output_action_names:
        zeroed_output_action_indices = zero_initialize_residual_actor_rows(
            runner.alg.policy,
            [env.joint_names.index(name) for name in zero_output_action_names],
        )
    trainable_action_names = tuple(
        name.strip()
        for name in (args.trainable_action_names or "").split(",")
        if name.strip()
    )
    active_action_names = tuple(
        name
        for name, enabled in zip(env.joint_names, env.action_mask[0].tolist())
        if enabled > 0.5
    )
    actor_update = None
    if args.actor_update_scope == "output_head":
        if trainable_action_names:
            unknown_trainable_names = sorted(
                set(trainable_action_names) - set(env.joint_names)
            )
            if unknown_trainable_names:
                raise ValueError(
                    "unknown trainable action names: "
                    f"{unknown_trainable_names}"
                )
            inactive_trainable_names = sorted(
                set(trainable_action_names) - set(active_action_names)
            )
            if inactive_trainable_names:
                raise ValueError(
                    "trainable action names must be enabled by the bridge "
                    f"profile: {inactive_trainable_names}"
                )
            active_action_indices = tuple(
                env.joint_names.index(name) for name in trainable_action_names
            )
        else:
            active_action_indices = tuple(
                int(index)
                for index in env.action_mask[0]
                .nonzero(as_tuple=False)
                .squeeze(-1)
                .tolist()
            )
        actor_update = restrict_actor_updates_to_action_rows(
            runner.alg.policy, active_action_indices
        )
    elif args.actor_update_scope == "candidate_gated_memory":
        actor_update = restrict_actor_updates_to_candidate_gated_memory(
            runner.alg.policy,
            context_width=len(bank["samples"]),
            gate_memory_width=4,
            trainable_candidate_indices=trainable_candidate_indices,
        )
    elif args.actor_update_scope == "candidate_gated_action_rows":
        if trainable_action_names:
            unknown_trainable_names = sorted(
                set(trainable_action_names) - set(env.joint_names)
            )
            if unknown_trainable_names:
                raise ValueError(
                    "unknown trainable action names: "
                    f"{unknown_trainable_names}"
                )
            inactive_trainable_names = sorted(
                set(trainable_action_names) - set(active_action_names)
            )
            if inactive_trainable_names:
                raise ValueError(
                    "trainable action names must be enabled by the bridge "
                    f"profile: {inactive_trainable_names}"
                )
            action_indices = tuple(
                env.joint_names.index(name) for name in trainable_action_names
            )
        else:
            action_indices = tuple(
                int(index)
                for index in env.action_mask[0]
                .nonzero(as_tuple=False)
                .squeeze(-1)
                .tolist()
            )
        actor_update = restrict_actor_updates_to_candidate_gated_action_rows(
            runner.alg.policy,
            context_width=len(bank["samples"]),
            gate_memory_width=4,
            action_indices=action_indices,
            trainable_candidate_indices=trainable_candidate_indices,
        )
    provenance = {
        "schema": "xhand_bodex_lift_bridge_training_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "non_promotional": True,
        "formal_stage_remains": 2,
        "object": args.object,
        "profile": profile.to_dict(),
        "manifest": str(args.manifest.resolve()),
        "bodex_bank": str(args.bodex_bank.resolve()),
        "bodex_bank_sha256": sha256_file(args.bodex_bank),
        "lift_targets": str(args.lift_targets.resolve()),
        "lift_targets_sha256": sha256_file(args.lift_targets),
        "resume_checkpoint": str(args.checkpoint.resolve()),
        "resume_checkpoint_sha256": sha256_file(args.checkpoint),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "seed": args.seed,
        "device": cfg.sim.device,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "arm_stiffness": float(cfg.arm_stiffness),
        "arm_damping": float(cfg.arm_damping),
        "hand_stiffness": float(cfg.hand_stiffness),
        "hand_damping": float(cfg.hand_damping),
        "object_max_depenetration_velocity": float(
            cfg.object_max_depenetration_velocity
        ),
        "lift_height_reward_weight": float(cfg.lift_height_reward_weight),
        "stable_lift_reward_weight": float(cfg.stable_lift_reward_weight),
        "terminal_success_weight": float(cfg.terminal_success_weight),
        "init_noise_std": args.init_noise_std,
        "candidate_selection": effective_candidate_selection,
        "candidate_repeat_factors": list(candidate_repeat_factors),
        "fixed_candidate_index": (
            int(profile.fixed_candidate_index)
            if profile.fixed_candidate_index is not None
            else None
        ),
        "nominal_pose_lock": bool(cfg.nominal_pose_lock),
        "palm_center_alignment_reward_weight": float(
            cfg.palm_center_alignment_reward_weight
        ),
        "trainable_candidate_indices": (
            list(trainable_candidate_indices)
            if trainable_candidate_indices is not None
            else None
        ),
        "actor_update_scope": args.actor_update_scope,
        "actor_update": actor_update,
        "observation_dimension": int(cfg.observation_space),
        "action_dimension": int(cfg.action_space),
        "active_action_count": int(env.action_mask.sum().item()),
        "active_action_names": list(active_action_names),
        "trainable_action_names": list(trainable_action_names) or None,
        "zeroed_output_action_names_after_load": list(zero_output_action_names),
        "zeroed_output_action_indices_after_load": list(zeroed_output_action_indices),
        "residual_activation_phase": int(cfg.residual_activation_phase),
        "residual_activation_close_fraction": float(
            cfg.residual_activation_close_fraction
        ),
        "bridge_terminate_on_unsafe": bool(cfg.bridge_terminate_on_unsafe),
    }
    provenance_path = output / "bridge_training_attempt.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    if runner.writer is not None:
        runner.writer.flush()
    checkpoint = latest_policy_checkpoint(output)
    if checkpoint is None:
        raise RuntimeError("RSL-RL produced no bridge checkpoint")
    provenance["completed_at"] = datetime.now(timezone.utc).isoformat()
    provenance["latest_checkpoint"] = str(checkpoint.resolve())
    provenance["latest_checkpoint_sha256"] = sha256_file(checkpoint)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "latest_training_result.json").write_text(
        json.dumps(
            {
                "schema": "xhand_bodex_lift_bridge_training_result_v1",
                "profile": profile.name,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "non_promotional": True,
                "fixed_evaluation_pending": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "profile": profile.name,
                "checkpoint": str(checkpoint.resolve()),
                "iterations": args.iterations,
            },
            indent=2,
        )
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
