#!/usr/bin/env python3
"""Run a deterministic fixed-window evaluation for one staged checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROCESS_STARTED = time.perf_counter()


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--bodex-bank", type=Path, required=True)
parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument(
    "--policy-mode",
    choices=("checkpoint", "zero_residual"),
    default="checkpoint",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--successful-state-output",
    type=Path,
    help=(
        "optional torch shard containing the final robot/object state for every "
        "successful audited episode"
    ),
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--seed", type=int, default=20260830)
parser.add_argument("--allow-test-bank", action="store_true")
parser.add_argument(
    "--fixed-candidate-index",
    type=int,
    default=-1,
    help="pin evaluation to one zero-based BODex candidate",
)
parser.add_argument(
    "--candidate-repeat-factors",
    help="comma-separated positive repeats for deterministic weighted BODex sampling",
)
parser.add_argument(
    "--nominal-pose-lock",
    action="store_true",
    help="disable XY/yaw/joint reset noise during evaluation",
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
    help="enable the fixed-pose palm-center alignment metric/reward path",
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
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"staged_eval_s{args.stage}_{args.object}")
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
)

from .configuration import configure_staged_env
from .contracts import STAGED_BACKEND, STAGED_EVALUATION_SCHEMA
from .env import XHandStagedEnv, XHandStagedEnvCfg
from .ppo_cfg import XHandStagedPPORunnerCfg


class EvaluationEnv(XHandStagedEnv):
    def __init__(
        self,
        *env_args: Any,
        capture_successful_states: bool = False,
        **env_kwargs: Any,
    ):
        self.evaluation_reports: list[dict[str, Any]] = []
        self.capture_successful_states = bool(capture_successful_states)
        self.successful_states: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            report = self._metrics_report(env_id)
            completion_index = len(self.evaluation_reports)
            report["completion_index"] = completion_index
            report["terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            self.evaluation_reports.append(report)
            if (
                self.capture_successful_states
                and report["stage_pass"]
                and not report["terminated_early"]
            ):
                self.successful_states.append(
                    {
                        "schema": "xhand_bodex_staged_success_state_v1",
                        "completion_index": completion_index,
                        "candidate_index": int(report["candidate_index"]),
                        "candidate_id": report["candidate_id"],
                        "joint_names": list(self.joint_names),
                        "final_full_body_q": self.robot.data.joint_pos[env_id]
                        .detach()
                        .cpu(),
                        "final_object_root_state_w": self.object.data.root_state_w[env_id]
                        .detach()
                        .cpu(),
                        "final_integrated_residual": self.integrated_residual[env_id]
                        .detach()
                        .cpu(),
                        "report": report,
                    }
                )
        return terminated, time_out


def main() -> None:
    if min(args.num_envs, args.episodes) <= 0:
        raise ValueError("num-envs and episodes must be positive")
    if args.policy_mode == "checkpoint" and args.checkpoint is None:
        raise ValueError("checkpoint policy mode requires --checkpoint")
    if args.policy_mode == "zero_residual" and args.checkpoint is not None:
        raise ValueError("zero-residual policy mode must not receive a checkpoint")
    if args.successful_state_output is not None and args.successful_state_output.exists():
        raise FileExistsError(args.successful_state_output)
    candidate_repeat_factors = (
        tuple(int(value.strip()) for value in args.candidate_repeat_factors.split(","))
        if args.candidate_repeat_factors
        else ()
    )
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
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
        allow_test_bank=bool(args.allow_test_bank),
        candidate_selection="round_robin",
        candidate_repeat_factors=candidate_repeat_factors,
        fixed_candidate_index=args.fixed_candidate_index,
        nominal_pose_lock=bool(args.nominal_pose_lock),
        palm_center_alignment_reward_weight=(
            args.palm_center_alignment_reward_weight
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
    runner_cfg = XHandStagedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = EvaluationEnv(
        cfg,
        capture_successful_states=args.successful_state_output is not None,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    if args.policy_mode == "checkpoint":
        runner = OnPolicyRunner(
            wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device
        )
        runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
        runner.eval_mode()
    else:
        runner = None
    observation = wrapped.get_observations()
    maximum_steps = math.ceil(args.episodes / args.num_envs) * (env.max_episode_length + 2)
    steps = 0
    rollout_started = time.perf_counter()
    with torch.inference_mode():
        while len(env.evaluation_reports) < args.episodes and steps < maximum_steps:
            actions = (
                runner.alg.policy.act_inference(observation)
                if runner is not None
                else torch.zeros(
                    (args.num_envs, cfg.action_space), device=env.device
                )
            )
            observation, _, _, _ = wrapped.step(actions)
            steps += 1
    rollout_wall_time_s = time.perf_counter() - rollout_started
    reports = env.evaluation_reports[: args.episodes]
    if len(reports) < args.episodes:
        raise RuntimeError(f"only completed {len(reports)} of {args.episodes} evaluation episodes")
    gate_names = list(env.stage_spec.required_gates)
    gate_rates = {
        name: sum(report["stage_gates"].get(name) is True for report in reports) / len(reports)
        for name in gate_names
    }
    successes = sum(
        report["stage_pass"] and not report["terminated_early"] for report in reports
    )
    end_to_end_wall_time_s = time.perf_counter() - PROCESS_STARTED
    result = {
        "schema": (
            STAGED_EVALUATION_SCHEMA
            if args.policy_mode == "checkpoint"
            else "xhand_rl_staged_nominal_evaluation_v1"
        ),
        "backend": STAGED_BACKEND,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "stage": args.stage,
        "stage_spec": env.stage_spec.to_dict(),
        "policy_mode": args.policy_mode,
        "checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "checkpoint_sha256": (
            sha256_file(args.checkpoint) if args.checkpoint is not None else None
        ),
        "bodex_bank": str(args.bodex_bank.resolve()),
        "bodex_bank_sha256": sha256_file(args.bodex_bank),
        "deterministic_policy": True,
        "candidate_selection": (
            "fixed" if int(cfg.fixed_candidate_index) >= 0 else cfg.candidate_selection
        ),
        "candidate_repeat_factors": list(env.candidate_repeat_factors),
        "actual_candidate_indices": sorted(
            {int(report["candidate_index"]) for report in reports}
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
        "terminal_contact_progress_semantics": cfg.terminal_contact_progress_mode,
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
        "num_envs": int(args.num_envs),
        "episodes": len(reports),
        "successes": successes,
        "success_rate": successes / len(reports),
        "end_to_end_wall_time_s": end_to_end_wall_time_s,
        "rollout_wall_time_s": rollout_wall_time_s,
        "episodes_per_hour_end_to_end": 3600.0
        * len(reports)
        / end_to_end_wall_time_s,
        "successes_per_hour_end_to_end": 3600.0
        * successes
        / end_to_end_wall_time_s,
        "gate_rates": gate_rates,
        "earlier_gates_no_regression": all(rate >= 0.80 for rate in gate_rates.values()),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.successful_state_output is not None:
        successful_states = [
            state
            for state in env.successful_states
            if int(state["completion_index"]) < len(reports)
        ]
        if len(successful_states) != successes:
            raise RuntimeError(
                "successful state capture count does not match audited successes"
            )
        args.successful_state_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "xhand_bodex_staged_success_state_shard_v1",
                "object": args.object,
                "stage": args.stage,
                "stage_spec": env.stage_spec.to_dict(),
                "seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "episodes": len(reports),
                "successes": successes,
                "candidate_selection": (
                    "fixed"
                    if int(cfg.fixed_candidate_index) >= 0
                    else cfg.candidate_selection
                ),
                "candidate_repeat_factors": list(env.candidate_repeat_factors),
                "fixed_candidate_index": int(cfg.fixed_candidate_index),
                "nominal_pose_lock": bool(cfg.nominal_pose_lock),
                "formal_active_action_group": env.stage_spec.active_action_group,
                "active_action_group": env.effective_active_action_group,
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
                "checkpoint": (
                    str(args.checkpoint.resolve())
                    if args.checkpoint is not None
                    else None
                ),
                "checkpoint_sha256": (
                    sha256_file(args.checkpoint)
                    if args.checkpoint is not None
                    else None
                ),
                "bodex_bank": str(args.bodex_bank.resolve()),
                "bodex_bank_sha256": sha256_file(args.bodex_bank),
                "samples": successful_states,
            },
            args.successful_state_output,
        )
        result["successful_state_output"] = str(
            args.successful_state_output.resolve()
        )
        result["successful_state_output_sha256"] = sha256_file(
            args.successful_state_output
        )
    else:
        result["successful_state_output"] = None
        result["successful_state_output_sha256"] = None
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("stage", "episodes", "successes", "success_rate", "gate_rates")}, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
