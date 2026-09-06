#!/usr/bin/env python3
"""Run a deterministic, non-promotional micro-lift diagnostic from Stage 2."""

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
parser.add_argument("--target-height-m", type=float, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=20260830)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"staged_micro_lift_{args.object}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from migration_4090.xhand_bodex_bimanual.contracts import sha256_file
from migration_4090.xhand_rl_embedded.contracts import validate_manifest

from .configuration import configure_staged_env
from .env import XHandStagedEnv, XHandStagedEnvCfg
from .micro_lift import (
    MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S,
    MICRO_LIFT_LINEAR_SPEED_MAX_M_S,
    MICRO_LIFT_MIN_HEIGHT_M,
    MICRO_LIFT_STABLE_HOLD_STEPS,
    controlled_height_upper_bound_m,
    summarize_micro_lift_reports,
)
from .ppo_cfg import XHandStagedPPORunnerCfg


class MicroLiftEnv(XHandStagedEnv):
    def __init__(self, *env_args: Any, **env_kwargs: Any):
        self.probe_reports: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)
        payload = torch.load(args.lift_targets, map_location="cpu", weights_only=False)
        if payload.get("schema") != "xhand_bodex_nonpromotional_micro_lift_targets_v1":
            raise RuntimeError("wrong micro-lift target schema")
        if payload.get("source_bodex_bank_sha256") != sha256_file(args.bodex_bank):
            raise RuntimeError("micro-lift targets do not match the BODex bank")
        if list(payload.get("candidate_ids", [])) != list(self.bodex_candidate_ids):
            raise RuntimeError("micro-lift target candidate order changed")
        key = f"{args.target_height_m:.6f}"
        rows = payload.get("targets_by_height", {}).get(key)
        if not isinstance(rows, list) or len(rows) != len(self.bodex_candidate_ids):
            raise RuntimeError(f"micro-lift targets lack height {key}")
        name_to_source = {
            name: index for index, name in enumerate(payload["joint_names"])
        }
        missing = [name for name in self.joint_names if name not in name_to_source]
        if missing:
            raise RuntimeError(f"micro-lift targets lack joints: {missing}")
        self.bodex_lift_bank = torch.tensor(
            [
                [row[name_to_source[name]] for name in self.joint_names]
                for row in rows
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.micro_lift_current_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.micro_lift_maximum_stable_steps = torch.zeros_like(
            self.micro_lift_current_stable_steps
        )
        self.micro_lift_maximum_linear_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self.micro_lift_maximum_angular_speed = torch.zeros(
            self.num_envs, device=self.device
        )
        self.micro_lift_physically_bounded = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_idx(self.robot._ALL_INDICES)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "micro_lift_current_stable_steps"):
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.micro_lift_current_stable_steps[env_ids] = 0
        self.micro_lift_maximum_stable_steps[env_ids] = 0
        self.micro_lift_maximum_linear_speed[env_ids] = 0.0
        self.micro_lift_maximum_angular_speed[env_ids] = 0.0
        self.micro_lift_physically_bounded[env_ids] = True

    def _update_micro_lift_stability(self) -> None:
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
            & (height >= MICRO_LIFT_MIN_HEIGHT_M)
            & (
                height
                <= controlled_height_upper_bound_m(args.target_height_m)
            )
        )
        self.micro_lift_physically_bounded &= bounded
        self.micro_lift_maximum_linear_speed = torch.maximum(
            self.micro_lift_maximum_linear_speed,
            torch.where(finite, linear_speed, self.micro_lift_maximum_linear_speed),
        )
        self.micro_lift_maximum_angular_speed = torch.maximum(
            self.micro_lift_maximum_angular_speed,
            torch.where(finite, angular_speed, self.micro_lift_maximum_angular_speed),
        )
        stable = (
            bounded
            & (height >= 0.8 * args.target_height_m)
            & (linear_speed <= MICRO_LIFT_LINEAR_SPEED_MAX_M_S)
            & (angular_speed <= MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S)
        )
        self.micro_lift_current_stable_steps = torch.where(
            stable,
            self.micro_lift_current_stable_steps + 1,
            torch.zeros_like(self.micro_lift_current_stable_steps),
        )
        self.micro_lift_maximum_stable_steps = torch.maximum(
            self.micro_lift_maximum_stable_steps,
            self.micro_lift_current_stable_steps,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_micro_lift_stability()
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            report = self._metrics_report(env_id)
            report["terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            report["final_lift_height_m"] = float(
                (
                    self.object.data.root_pos_w[env_id, 2]
                    - self.initial_object_pos[env_id, 2]
                    - self.scene.env_origins[env_id, 2]
                ).item()
            )
            report["micro_lift_physically_bounded"] = bool(
                self.micro_lift_physically_bounded[env_id].item()
            )
            report["maximum_micro_lift_stable_steps"] = int(
                self.micro_lift_maximum_stable_steps[env_id].item()
            )
            report["maximum_micro_lift_linear_speed_m_s"] = float(
                self.micro_lift_maximum_linear_speed[env_id].item()
            )
            report["maximum_micro_lift_angular_speed_rad_s"] = float(
                self.micro_lift_maximum_angular_speed[env_id].item()
            )
            self.probe_reports.append(report)
        return terminated, time_out


def main() -> None:
    if min(args.num_envs, args.episodes) <= 0 or args.target_height_m <= 0.0:
        raise ValueError("num-envs, episodes, and target height must be positive")
    for path in (args.checkpoint, args.lift_targets):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
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
    )
    # This is deliberately outside the immutable promotion curriculum.  It
    # inserts a short paired-palm lift between closure and hold while retaining
    # the Stage-2 observation ABI, action mask, gates, and champion policy.
    cfg.approach_fraction = 0.15
    cfg.close_fraction = 0.35
    cfg.lift_fraction = 0.25
    cfg.hold_fraction = 0.25
    cfg.disturbance_fraction = 0.0
    cfg.ablation_fraction = 0.0
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = MicroLiftEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device
    )
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    runner.eval_mode()
    observation = wrapped.get_observations()
    maximum_steps = math.ceil(args.episodes / args.num_envs) * (
        env.max_episode_length + 2
    )
    steps = 0
    with torch.inference_mode():
        while len(env.probe_reports) < args.episodes and steps < maximum_steps:
            actions = runner.alg.policy.act_inference(observation)
            observation, _, _, _ = wrapped.step(actions)
            steps += 1
    reports = env.probe_reports[: args.episodes]
    if len(reports) < args.episodes:
        raise RuntimeError(
            f"only completed {len(reports)} of {args.episodes} probe episodes"
        )
    summary = summarize_micro_lift_reports(
        reports, target_height_m=args.target_height_m
    )
    result = {
        "schema": "xhand_bodex_stage2_nonpromotional_micro_lift_probe_v3",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "non_promotional": True,
        "promotion_eligible": False,
        "formal_stage_remains": 2,
        "object": args.object,
        "deterministic_policy": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "bodex_bank": str(args.bodex_bank.resolve()),
        "bodex_bank_sha256": sha256_file(args.bodex_bank),
        "lift_targets": str(args.lift_targets.resolve()),
        "lift_targets_sha256": sha256_file(args.lift_targets),
        "target_height_m": args.target_height_m,
        "num_envs": args.num_envs,
        "episodes": args.episodes,
        "seed": args.seed,
        "phase_fractions": {
            "approach": cfg.approach_fraction,
            "close": cfg.close_fraction,
            "lift": cfg.lift_fraction,
            "hold": cfg.hold_fraction,
        },
        "controlled_lift_contract": {
            "maximum_height_m": controlled_height_upper_bound_m(
                args.target_height_m
            ),
            "minimum_height_m": MICRO_LIFT_MIN_HEIGHT_M,
            "terminal_linear_speed_max_m_s": (
                MICRO_LIFT_LINEAR_SPEED_MAX_M_S
            ),
            "terminal_angular_speed_max_rad_s": (
                MICRO_LIFT_ANGULAR_SPEED_MAX_RAD_S
            ),
            "stable_hold_steps": MICRO_LIFT_STABLE_HOLD_STEPS,
        },
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "target_height_m": args.target_height_m,
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
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
