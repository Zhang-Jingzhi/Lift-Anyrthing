#!/usr/bin/env python3
"""Deterministically evaluate a non-promotional lift bridge checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROCESS_STARTED_AT = time.perf_counter()


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
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=20262331)
parser.add_argument(
    "--zero-actions",
    action="store_true",
    help="evaluate the BODex/IK nominal trajectory without an RL residual",
)
parser.add_argument(
    "--zero-action-candidates",
    help="comma-separated candidate indices whose RL residual is suppressed",
)
# Arm and hand position-control gains.  Ours are 1500/120 and 50/2; the
# neighbouring skrl task drives the same robot at 800/60 and 30/2.  A stiffer
# arm drives through an obstacle rather than yielding, which matches the
# measured asymmetry: 3 mm of hand-object penetration when the ball is knocked
# clear before closure, 10-12 mm when a retracted start leaves it in place.
# Defaults keep the historical values, so passing nothing changes nothing.
parser.add_argument("--latch-fingers-on-release", action="store_true")
parser.add_argument("--freeze-object-until-bilateral", action="store_true")
parser.add_argument("--freeze-release-force-n", type=float, default=0.02)
parser.add_argument("--freeze-release-hold-steps", type=int, default=4)
parser.add_argument("--finger-close-scale", type=float, default=1.0)
parser.add_argument("--no-self-collisions", action="store_true")
parser.add_argument("--object-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--arm-stiffness", type=float, default=None)
parser.add_argument("--arm-damping", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--hand-damping", type=float, default=None)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"lift_bridge_eval_{args.object}_{args.profile}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_bodex_bimanual.contracts import sha256_file
from migration_4090.xhand_rl_embedded.contracts import validate_manifest
from migration_4090.xhand_rl_embedded.env import PHASE_APPROACH, PHASE_CLOSE, PHASE_HOLD, PHASE_LIFT

from .bridge_env import XHandLiftBridgeEnv
from .bridge_profiles import get_bridge_profile
from .configuration import configure_staged_env
from .env import XHandStagedEnvCfg
from .micro_lift import summarize_micro_lift_reports
from .ppo_cfg import XHandStagedPPORunnerCfg
from .throughput import evaluation_throughput


class EvaluationEnv(XHandLiftBridgeEnv):
    def __init__(self, *env_args: Any, **env_kwargs: Any):
        self.bridge_evaluation_reports: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            report = self.bridge_metrics_report(env_id)
            report["terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            self.bridge_evaluation_reports.append(report)
        return terminated, time_out


def main() -> None:
    if min(args.num_envs, args.episodes) <= 0:
        raise ValueError("num-envs and episodes must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    profile = get_bridge_profile(args.profile)
    effective_candidate_selection = (
        "fixed" if profile.fixed_candidate_index is not None else "round_robin"
    )
    zero_action_candidates = tuple(
        sorted(
            {
                int(value)
                for value in (args.zero_action_candidates or "").split(",")
                if value.strip()
            }
        )
    )
    if any(index < 0 or index >= 4 for index in zero_action_candidates):
        raise ValueError("zero-action candidate indices must be in [0, 3]")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
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
        candidate_repeat_factors=(1, 1, 1, 1),
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
    )
    for _field, _value in (
        ("robot_self_collisions", False if args.no_self_collisions else None),
        ("object_max_depenetration_velocity",
         args.object_max_depenetration_velocity),
        ("arm_stiffness", args.arm_stiffness),
        ("arm_damping", args.arm_damping),
        ("hand_stiffness", args.hand_stiffness),
        ("hand_damping", args.hand_damping),
    ):
        if _value is not None:
            setattr(cfg, _field, float(_value))
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
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = EvaluationEnv(
        cfg,
        lift_targets=args.lift_targets,
        retracted_pregrasp=args.retracted_pregrasp,
        retract_distance_m=args.retract_distance_m,
        finger_close_scale=args.finger_close_scale,
        freeze_object_until_bilateral=args.freeze_object_until_bilateral,
        freeze_release_force_n=args.freeze_release_force_n,
        freeze_release_hold_steps=args.freeze_release_hold_steps,
        latch_fingers_on_release=args.latch_fingers_on_release,
        source_bodex_bank=args.bodex_bank,
        profile=profile,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device
    )
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    runner.eval_mode()
    observation = wrapped.get_observations()
    rollout_started_at = time.perf_counter()
    maximum_steps = math.ceil(args.episodes / args.num_envs) * (
        env.max_episode_length + 2
    )
    steps = 0
    with torch.inference_mode():
        while (
            len(env.bridge_evaluation_reports) < args.episodes
            and steps < maximum_steps
        ):
            if args.zero_actions:
                actions = torch.zeros(
                    (args.num_envs, int(cfg.action_space)),
                    dtype=torch.float32,
                    device=cfg.sim.device,
                )
            else:
                actions = runner.alg.policy.act_inference(observation)
                if zero_action_candidates:
                    candidate_mask = torch.zeros(
                        args.num_envs, dtype=torch.bool, device=cfg.sim.device
                    )
                    for candidate_index in zero_action_candidates:
                        candidate_mask |= env.active_bank_index == candidate_index
                    actions[candidate_mask] = 0.0
            observation, _, _, _ = wrapped.step(actions)
            steps += 1
    rollout_wall_time_s = time.perf_counter() - rollout_started_at
    reports = env.bridge_evaluation_reports[: args.episodes]
    if len(reports) < args.episodes:
        raise RuntimeError(
            f"only completed {len(reports)} of {args.episodes} bridge episodes"
        )
    # The profile's stability contract, not the scorer's module defaults.  Until
    # 2026-09-08 only target_height_m was passed, so lift50_hold1s_v1's 125-step
    # hold was scored at the hard-coded 32 and every number taken from it was
    # measured against a window four times shorter than the one it declared.
    summary = summarize_micro_lift_reports(
        reports,
        target_height_m=profile.target_height_m,
        stable_hold_steps=profile.stable_hold_steps,
        linear_speed_max_m_s=profile.linear_speed_max_m_s,
        angular_speed_max_rad_s=profile.angular_speed_max_rad_s,
    )
    throughput = evaluation_throughput(
        episodes=args.episodes,
        overall_rates=summary["overall_rates"],
        end_to_end_wall_time_s=time.perf_counter() - PROCESS_STARTED_AT,
        rollout_wall_time_s=rollout_wall_time_s,
    )
    result = {
        "schema": "xhand_bodex_lift_bridge_evaluation_v1",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "non_promotional": True,
        "formal_stage_remains": 2,
        "object": args.object,
        "profile": profile.to_dict(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "bodex_bank": str(args.bodex_bank.resolve()),
        "bodex_bank_sha256": sha256_file(args.bodex_bank),
        "lift_targets": str(args.lift_targets.resolve()),
        "lift_targets_sha256": sha256_file(args.lift_targets),
        "deterministic_policy": True,
        "policy_mode": "zero_residual" if args.zero_actions else "checkpoint",
        "scoring_criteria": {
            "target_height_m": profile.target_height_m,
            "stable_hold_steps": profile.stable_hold_steps,
            "linear_speed_max_m_s": profile.linear_speed_max_m_s,
            "angular_speed_max_rad_s": profile.angular_speed_max_rad_s,
            "maximum_overshoot_m": profile.maximum_overshoot_m,
            "minimum_height_m": profile.minimum_height_m,
        },
        "arm_stiffness": float(cfg.arm_stiffness),
        "arm_damping": float(cfg.arm_damping),
        "hand_stiffness": float(cfg.hand_stiffness),
        "hand_damping": float(cfg.hand_damping),
        "zero_action_candidates": list(zero_action_candidates),
        "candidate_selection": effective_candidate_selection,
        "candidate_repeat_factors": [1, 1, 1, 1],
        "fixed_candidate_index": (
            int(profile.fixed_candidate_index)
            if profile.fixed_candidate_index is not None
            else None
        ),
        "nominal_pose_lock": bool(cfg.nominal_pose_lock),
        "palm_center_alignment_reward_weight": float(
            cfg.palm_center_alignment_reward_weight
        ),
        "active_action_names": [
            name
            for name, enabled in zip(env.joint_names, env.action_mask[0].tolist())
            if enabled > 0.5
        ],
        "active_action_count": int(env.action_mask.sum().item()),
        "residual_activation_phase": int(cfg.residual_activation_phase),
        "residual_activation_close_fraction": float(
            cfg.residual_activation_close_fraction
        ),
        "num_envs": args.num_envs,
        "episodes": args.episodes,
        "seed": args.seed,
        "throughput": throughput,
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": profile.name,
                "episodes": args.episodes,
                "overall_rates": result["overall_rates"],
                "candidate_rates": result["candidate_rates"],
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
