#!/usr/bin/env python3
"""Capture deterministic trajectories from a lift-bridge checkpoint.

The ordinary bridge evaluator intentionally writes scalar reports only.  This
entry point uses the same profile/configuration/checkpoint path, but enables
the environment trajectory buffers so the exact rollout can be rendered by
``migration_4090.xhand_rl.render_trajectory``.
"""

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
parser.add_argument("--lift-targets", type=Path, required=True)
parser.add_argument("--profile", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=20262460)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"bridge_visualize_{args.object}_{args.profile}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_rl_embedded.contracts import validate_manifest
from migration_4090.xhand_rl_embedded.env import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_HOLD,
    PHASE_LIFT,
)
from .bridge_env import XHandLiftBridgeEnv
from .bridge_profiles import get_bridge_profile
from .configuration import configure_staged_env
from .env import XHandStagedEnvCfg
from .ppo_cfg import XHandStagedPPORunnerCfg


PHASE_NAMES = {
    PHASE_APPROACH: "pregrasp",
    PHASE_CLOSE: "closure",
    PHASE_LIFT: "lift",
    PHASE_HOLD: "hold",
}


class CaptureEnv(XHandLiftBridgeEnv):
    def __init__(self, *env_args: Any, **env_kwargs: Any):
        self.captured: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            episode_steps = int(self.episode_length_buf[env_id].item())
            episode_steps = max(1, min(episode_steps, self.max_episode_length))
            phases = self.trajectory_phase[env_id, :episode_steps].detach().cpu()
            joint_q = self.trajectory_joint_q[env_id, :episode_steps].detach().cpu()
            object_state = self.trajectory_object_state[
                env_id, :episode_steps
            ].detach().cpu()
            env_origin = self.scene.env_origins[env_id].detach().cpu()
            states: list[dict[str, Any]] = []
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
            report = self.bridge_metrics_report(env_id)
            report["terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            self.captured.append(
                {
                    "candidate_index": int(self.active_bank_index[env_id].item()),
                    "candidate_id": report["candidate_id"],
                    "report": report,
                    "trajectory": {
                        "schema": "xhand_bodex_bridge_visualization_trajectory_v1",
                        "simulation_dt_s": float(self.cfg.sim.dt * self.cfg.decimation),
                        "joint_names": list(self.joint_names),
                        "states": states,
                    },
                }
            )
        return terminated, time_out


def main() -> None:
    if args.num_envs <= 0 or args.episodes <= 0:
        raise ValueError("num-envs and episodes must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    profile = get_bridge_profile(args.profile)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    candidate_selection = "fixed" if profile.fixed_candidate_index is not None else "round_robin"
    cfg = configure_staged_env(
        XHandStagedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        bodex_bank=args.bodex_bank,
        stage_id=2,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=True,
        allow_test_bank=False,
        candidate_selection="round_robin",
        candidate_repeat_factors=(1, 1, 1, 1),
        fixed_candidate_index=(
            int(profile.fixed_candidate_index)
            if profile.fixed_candidate_index is not None
            else -1
        ),
        nominal_pose_lock=bool(profile.nominal_pose_lock),
        palm_center_alignment_reward_weight=float(profile.palm_center_alignment_reward_weight),
        terminal_contact_progress_mode="joint_gate",
        distributed_reward_requires_current_bilateral_contact=True,
    )
    cfg.episode_length_s = profile.episode_length_s
    cfg.approach_fraction, cfg.close_fraction, cfg.lift_fraction, cfg.hold_fraction = profile.phase_fractions
    from migration_4090.xhand_rl_embedded.env import PHASE_CLOSE, PHASE_HOLD, PHASE_LIFT

    cfg.residual_activation_phase = (
        PHASE_CLOSE
        if profile.residual_activation == "close"
        else PHASE_HOLD
        if profile.residual_activation == "hold"
        else PHASE_LIFT
    )
    cfg.residual_activation_close_fraction = profile.residual_activation_close_fraction
    cfg.residual_integration = profile.residual_integration
    cfg.residual_limit_rad = profile.residual_limit_rad
    cfg.arm_approach_fraction_of_close = profile.arm_approach_fraction_of_close
    cfg.finger_close_end_fraction_of_close = profile.finger_close_end_fraction_of_close
    cfg.residual_integration_end_fraction_of_hold = profile.residual_integration_end_fraction_of_hold
    cfg.reset_xy_noise_m = profile.reset_xy_noise_m
    cfg.reset_yaw_noise_rad = profile.reset_yaw_noise_rad
    cfg.reset_joint_noise_rad = profile.reset_joint_noise_rad
    cfg.terminate_on_object_escape = False
    env = CaptureEnv(
        cfg,
        lift_targets=args.lift_targets,
        source_bodex_bank=args.bodex_bank,
        profile=profile,
    )
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    runner.eval_mode()
    observation = wrapped.get_observations()
    maximum_steps = math.ceil(args.episodes / args.num_envs) * (env.max_episode_length + 2)
    steps = 0
    with torch.inference_mode():
        while len(env.captured) < args.episodes and steps < maximum_steps:
            actions = runner.alg.policy.act_inference(observation)
            observation, _, _, _ = wrapped.step(actions)
            steps += 1
    episodes = env.captured[: args.episodes]
    if len(episodes) < args.episodes:
        raise RuntimeError(f"captured {len(episodes)} of {args.episodes} episodes")
    args.output_dir.mkdir(parents=True)
    rows = []
    # A fixed-candidate diagnostic intentionally captures several rollouts of
    # the same BODex sample.  Candidate-only filenames silently overwrote all
    # but the final rollout, which made it impossible to inspect multiple
    # failure modes from nominal_pose_lock experiments.  Keep capture order in
    # the filename and manifest so every report remains paired with its exact
    # trajectory.
    for episode_index, episode in enumerate(episodes):
        index = int(episode["candidate_index"])
        stem = f"episode_{episode_index:03d}_candidate_{index}"
        trajectory_path = args.output_dir / f"{stem}_trajectory.json"
        report_path = args.output_dir / f"{stem}_report.json"
        trajectory_path.write_text(json.dumps(episode["trajectory"], indent=2) + "\n")
        report_path.write_text(json.dumps(episode["report"], indent=2) + "\n")
        rows.append(
            {
                "episode_index": episode_index,
                "candidate_index": index,
                "candidate_id": episode["candidate_id"],
                "stage_pass": bool(episode["report"].get("stage_pass", False)),
                "trajectory": str(trajectory_path.resolve()),
                "report": str(report_path.resolve()),
            }
        )
    manifest_path = args.output_dir / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "xhand_bodex_bridge_visualization_capture_v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "object": args.object,
                "profile": profile.to_dict(),
                "checkpoint": str(args.checkpoint.resolve()),
                "bodex_bank": str(args.bodex_bank.resolve()),
                "lift_targets": str(args.lift_targets.resolve()),
                "candidate_selection": candidate_selection,
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
