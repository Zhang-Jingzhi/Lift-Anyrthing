#!/usr/bin/env python3
"""Capture deterministic, episode-aligned Stage-policy trajectories for video."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--bodex-bank", type=Path, required=True)
parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=64)
parser.add_argument("--seed", type=int, default=20260830)
parser.add_argument(
    "--fixed-candidate-index",
    type=int,
    default=-1,
    help="pin visualization to one zero-based BODex candidate",
)
parser.add_argument(
    "--nominal-pose-lock",
    action="store_true",
    help="disable XY/yaw/joint reset noise during visualization",
)
parser.add_argument(
    "--candidate-repeat-factors",
    help="comma-separated positive repeats for deterministic weighted BODex sampling",
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
    "--active-action-group-override",
    choices=("hands", "distal_wrist", "hands_and_distal_arms", "all"),
)
parser.add_argument(
    "--residual-activation-phase",
    choices=("approach", "close", "hold"),
)
parser.add_argument("--residual-activation-close-fraction", type=float)
parser.add_argument(
    "--hand-residual-activation-phase",
    choices=("approach", "close", "hold"),
)
parser.add_argument("--hand-residual-activation-close-fraction", type=float)
parser.add_argument("--residual-integration", type=float)
parser.add_argument("--residual-limit-rad", type=float)
parser.add_argument(
    "--palm-center-alignment-reward-weight",
    type=float,
    default=None,
    help="enable the fixed-pose palm-center alignment metric path",
)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"staged_visualize_s{args.stage}_{args.object}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_bodex_bimanual.contracts import sha256_file
from migration_4090.xhand_rl_embedded.contracts import validate_manifest
from migration_4090.xhand_rl_embedded.env import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_HOLD,
    PHASE_LIFT,
)

from .configuration import configure_staged_env
from .env import XHandStagedEnv, XHandStagedEnvCfg
from .ppo_cfg import XHandStagedPPORunnerCfg


PHASE_NAMES = {
    PHASE_APPROACH: "pregrasp",
    PHASE_CLOSE: "closure",
    PHASE_LIFT: "lift",
    PHASE_HOLD: "hold",
}


class VisualizationEnv(XHandStagedEnv):
    def __init__(self, *env_args: Any, **env_kwargs: Any):
        self.visualization_episodes: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            episode_steps = int(self.episode_length_buf[env_id].item())
            phases = self.trajectory_phase[env_id, :episode_steps].detach().cpu()
            joint_q = self.trajectory_joint_q[env_id, :episode_steps].detach().cpu()
            object_state = self.trajectory_object_state[
                env_id, :episode_steps
            ].detach().cpu()
            env_origin = self.scene.env_origins[env_id].detach().cpu()
            states = []
            for step, (phase, q, state) in enumerate(
                zip(phases.tolist(), joint_q, object_state)
            ):
                phase_name = PHASE_NAMES.get(int(phase))
                if phase_name is None:
                    continue
                states.append(
                    {
                        "step": step,
                        "stage": phase_name,
                        "joint_positions": q.tolist(),
                        "object_position_world_m": (state[:3] - env_origin).tolist(),
                        "object_quaternion_world_wxyz": state[3:7].tolist(),
                        "object_linear_velocity_world_m_s": state[7:10].tolist(),
                        "object_angular_velocity_world_rad_s": state[10:13].tolist(),
                    }
                )
            report = self._metrics_report(env_id)
            completion_index = len(self.visualization_episodes)
            report["completion_index"] = completion_index
            report["terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            self.visualization_episodes.append(
                {
                    "completion_index": completion_index,
                    "candidate_index": int(self.active_bank_index[env_id].item()),
                    "candidate_id": report["candidate_id"],
                    "report": report,
                    "trajectory": {
                        "schema": "xhand_rl_staged_visualization_trajectory_v1",
                        "simulation_dt_s": float(self.cfg.sim.dt * self.cfg.decimation),
                        "joint_names": list(self.joint_names),
                        "states": states,
                    },
                }
            )
        return terminated, time_out


def main() -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if min(args.num_envs, args.episodes) <= 0:
        raise ValueError("num-envs and episodes must be positive")
    candidate_repeat_factors = (
        tuple(int(value.strip()) for value in args.candidate_repeat_factors.split(","))
        if args.candidate_repeat_factors
        else ()
    )
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    capture_count = int(args.episodes)
    cfg = configure_staged_env(
        XHandStagedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        bodex_bank=args.bodex_bank,
        stage_id=args.stage,
        num_envs=int(args.num_envs),
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=True,
        allow_test_bank=False,
        candidate_selection="round_robin",
        candidate_repeat_factors=candidate_repeat_factors,
        fixed_candidate_index=args.fixed_candidate_index,
        nominal_pose_lock=bool(args.nominal_pose_lock),
        palm_center_alignment_reward_weight=(
            args.palm_center_alignment_reward_weight
        ),
        stage_contact_groups_per_side_min=args.stage_contact_groups_per_side_min,
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
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = VisualizationEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device
    )
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    runner.eval_mode()
    observation = wrapped.get_observations()
    maximum_steps = math.ceil(capture_count / cfg.scene.num_envs) * (
        env.max_episode_length + 2
    )
    steps = 0
    with torch.inference_mode():
        while len(env.visualization_episodes) < capture_count and steps < maximum_steps:
            actions = runner.alg.policy.act_inference(observation)
            observation, _, _, _ = wrapped.step(actions)
            steps += 1
    episodes = env.visualization_episodes[:capture_count]
    if len(episodes) != capture_count:
        raise RuntimeError(
            f"captured {len(episodes)} of {capture_count} episodes"
        )
    args.output_dir.mkdir(parents=True)
    rows = []
    for episode_number, episode in enumerate(episodes):
        index = int(episode["candidate_index"])
        stem = f"episode_{episode_number:04d}_candidate_{index}"
        trajectory_path = args.output_dir / f"{stem}_trajectory.json"
        report_path = args.output_dir / f"{stem}_report.json"
        trajectory_path.write_text(json.dumps(episode["trajectory"], indent=2) + "\n")
        report_path.write_text(json.dumps(episode["report"], indent=2) + "\n")
        rows.append(
            {
                "episode_number": episode_number,
                "completion_index": int(episode["completion_index"]),
                "candidate_index": index,
                "candidate_id": episode["candidate_id"],
                "stage_pass": bool(episode["report"]["stage_pass"]),
                "trajectory": str(trajectory_path.resolve()),
                "report": str(report_path.resolve()),
            }
        )
    manifest_path = args.output_dir / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "xhand_rl_staged_visualization_capture_v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "object": args.object,
                "stage": args.stage,
                "seed": int(args.seed),
                "stage_spec": env.stage_spec.to_dict(),
                "deterministic_policy": True,
                "num_envs": int(cfg.scene.num_envs),
                "episodes_requested": capture_count,
                "candidate_repeat_factors": list(env.candidate_repeat_factors),
                "candidate_selection": (
                    "fixed"
                    if int(cfg.fixed_candidate_index) >= 0
                    else cfg.candidate_selection
                ),
                "actual_candidate_indices": sorted(
                    {int(episode["candidate_index"]) for episode in episodes}
                ),
                "fixed_candidate_index": int(cfg.fixed_candidate_index),
                "nominal_pose_lock": bool(cfg.nominal_pose_lock),
                "palm_center_alignment_reward_weight": float(
                    cfg.palm_center_alignment_reward_weight
                ),
                "formal_active_action_group": env.stage_spec.active_action_group,
                "active_action_group": env.effective_active_action_group,
                "active_action_count": int(env.action_mask.sum().item()),
                "distributed_reward_requires_current_bilateral_contact": bool(
                    cfg.distributed_reward_requires_current_bilateral_contact
                ),
                "candidate_observation_context": (
                    "four_way_one_hot_plus_candidate_gated_gate_memory"
                ),
                "observation_dimension": int(cfg.observation_space),
                "terminal_contact_progress_semantics": (
                    cfg.terminal_contact_progress_mode
                ),
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
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "bodex_bank": str(args.bodex_bank.resolve()),
                "bodex_bank_sha256": sha256_file(args.bodex_bank),
                "episodes": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"manifest": str(manifest_path.resolve()), "episodes": rows}, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
