#!/usr/bin/env python3
"""Read-only embedded-gate evaluation for an intermediate PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--nominal-dataset", type=Path, required=True)
parser.add_argument("--nominal-sample-index", type=int, default=0)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--action-mode", choices=("stochastic", "deterministic", "zero"), required=True
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--completed-episodes", type=int, default=64)
parser.add_argument("--max-steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=87)
parser.add_argument("--output-json", type=Path)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from .launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"diagnose_{args.object}_{args.action_mode}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .configuration import configure_env
from .contracts import sha256_file, validate_manifest
from .env import XHandEmbeddedEnv, XHandEmbeddedEnvCfg
from .gates import PHYSICS_GATE_NAMES
from .ppo_cfg import XHandEmbeddedPPORunnerCfg


class DiagnosticEnv(XHandEmbeddedEnv):
    """Capture every terminal report before DirectRLEnv resets the episode."""

    def __init__(self, *env_args: Any, **env_kwargs: Any):
        self.diagnostic_terminal_reports: list[dict[str, Any]] = []
        super().__init__(*env_args, **env_kwargs)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        finished = torch.nonzero(terminated | time_out, as_tuple=False).squeeze(-1)
        for env_id in finished.tolist():
            report = self._metrics_report(env_id)
            report["diagnostic_terminated_early"] = bool(
                terminated[env_id].item() and not time_out[env_id].item()
            )
            self.diagnostic_terminal_reports.append(report)
        return terminated, time_out


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"finite_count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "finite_count": len(finite),
        "minimum": min(finite),
        "mean": sum(finite) / len(finite),
        "maximum": max(finite),
    }


def main() -> int:
    if args.num_envs <= 0 or args.completed_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("diagnostic counts must be positive")
    if args.nominal_sample_index < 0:
        raise ValueError("nominal sample index must be non-negative")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    for path in (args.nominal_dataset, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = configure_env(
        XHandEmbeddedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        nominal_dataset=args.nominal_dataset,
        nominal_sample_index=args.nominal_sample_index,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=False,
    )
    runner_cfg = XHandEmbeddedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = DiagnosticEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    runner.eval_mode()
    if args.action_mode == "stochastic":
        policy = runner.alg.policy.act
    elif args.action_mode == "deterministic":
        policy = runner.alg.policy.act_inference
    else:
        policy = lambda observation: torch.zeros(  # noqa: E731
            (observation.shape[0], cfg.action_space),
            dtype=observation.dtype,
            device=observation.device,
        )
    observations = wrapped.get_observations()
    steps = 0
    with torch.inference_mode():
        while len(env.diagnostic_terminal_reports) < args.completed_episodes and steps < args.max_steps:
            actions = policy(observations)
            observations, _, _, _ = wrapped.step(actions)
            steps += 1
    reports = env.diagnostic_terminal_reports[: args.completed_episodes]
    gate_counts = {
        name: sum(report["physics_gates"].get(name) is True for report in reports)
        for name in PHYSICS_GATE_NAMES
    }
    strict_successes = sum(
        not report["diagnostic_terminated_early"]
        and all(report["physics_gates"].values())
        for report in reports
    )
    scalar_fields = (
        "lift_displacement_m",
        "gravity_drift_m",
        "hold_max_drift_m",
        "maximum_physx_contact_penetration_m",
        "force_closure_quality",
    )
    metrics = {
        field: finite_summary([float(report[field]) for report in reports])
        for field in scalar_fields
    }
    metrics["left_contact_presence_fraction"] = finite_summary(
        [float(report["contact_presence_fraction"]["left"]) for report in reports]
    )
    metrics["right_contact_presence_fraction"] = finite_summary(
        [float(report["contact_presence_fraction"]["right"]) for report in reports]
    )
    metrics["left_only_inactive_hand_contact_fraction"] = finite_summary(
        [float(report["inactive_hand_contact_fraction"]["left_only"]) for report in reports]
    )
    metrics["right_only_inactive_hand_contact_fraction"] = finite_summary(
        [float(report["inactive_hand_contact_fraction"]["right_only"]) for report in reports]
    )
    metrics["maximum_translation_disturbance_m"] = finite_summary(
        [max(report["translation_disturbances_m"]) for report in reports]
    )
    metrics["maximum_rotation_disturbance_rad"] = finite_summary(
        [max(report["rotation_disturbances_rad"]) for report in reports]
    )
    completed = len(reports)
    result = {
        "schema": "xhand_rl_embedded_checkpoint_diagnostic_v2",
        "object": args.object,
        "nominal_dataset": str(args.nominal_dataset.resolve()),
        "nominal_dataset_sha256": sha256_file(args.nominal_dataset),
        "nominal_sample_index": args.nominal_sample_index,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "action_mode": args.action_mode,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_completed_episodes": args.completed_episodes,
        "completed_episodes": completed,
        "full_length_episodes": sum(
            not report["diagnostic_terminated_early"] for report in reports
        ),
        "early_terminated_episodes": sum(
            report["diagnostic_terminated_early"] for report in reports
        ),
        "strict_successes": strict_successes,
        "single_hand_success_counts": {
            side: sum(report["single_hand_grasp_succeeded"][side] is True for report in reports)
            for side in ("left", "right")
        },
        "gate_pass_counts": gate_counts,
        "gate_pass_fractions": {
            name: (count / completed if completed else 0.0)
            for name, count in gate_counts.items()
        },
        "metrics": metrics,
        "policy_steps": steps,
        "strict_visual_mesh_accepted": False,
        "production_artifacts_written": False,
        "ready": completed == args.completed_episodes,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output_json is not None:
        output = args.output_json.resolve()
        if output.exists():
            raise FileExistsError(f"diagnostic output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        temporary.write_text(serialized)
        os.replace(temporary, output)
    print(serialized, end="", flush=True)
    env.close()
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
