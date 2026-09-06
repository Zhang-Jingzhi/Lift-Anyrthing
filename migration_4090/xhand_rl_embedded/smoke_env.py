#!/usr/bin/env python3
"""Exercise a v2 episode without training or production writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--nominal-dataset", type=Path, required=True)
parser.add_argument("--steps", type=int, default=960)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--stable-lift-reward-weight", type=float, default=0.0)
parser.add_argument("--force-closure-reward-weight", type=float, default=4.0)
parser.add_argument("--hard-gate-frontier-reward-weight", type=float, default=0.0)
parser.add_argument("--disturbance-direction-reward-weight", type=float, default=0.0)
parser.add_argument(
    "--force-closure-shape-all-hold-contacts", action="store_true"
)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from .launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"smoke_{args.object}")
simulation_app = AppLauncher(args).app

import torch

from .configuration import configure_env
from .contracts import validate_manifest
from .env import XHandEmbeddedEnv, XHandEmbeddedEnvCfg


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    cfg = configure_env(
        XHandEmbeddedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        nominal_dataset=args.nominal_dataset,
        nominal_sample_index=0,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=86,
        record_trajectory=True,
    )
    cfg.stable_lift_reward_weight = float(args.stable_lift_reward_weight)
    cfg.force_closure_reward_weight = float(args.force_closure_reward_weight)
    cfg.hard_gate_frontier_reward_weight = float(
        args.hard_gate_frontier_reward_weight
    )
    cfg.disturbance_direction_reward_weight = float(
        args.disturbance_direction_reward_weight
    )
    cfg.force_closure_shape_all_hold_contacts = bool(
        args.force_closure_shape_all_hold_contacts
    )
    env = XHandEmbeddedEnv(cfg)
    observation, _ = env.reset()
    phases = set()
    reward = torch.zeros(args.num_envs, device=env.device)
    terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    truncated = torch.zeros_like(terminated)
    diagnostic_prefixes = (
        "embedded/hard_prefix_contact_fraction",
        "embedded/hard_prefix_lifted_fraction",
        "embedded/hard_prefix_clear_fraction",
        "embedded/hard_prefix_stable_force_closure_fraction",
        "embedded/hard_prefix_disturbed_fraction",
        "embedded/hard_prefix_ablated_fraction",
    )
    diagnostics_max = {name: 0.0 for name in diagnostic_prefixes}
    for _ in range(args.steps):
        phases.update(int(value) for value in env._phase_code().detach().cpu().tolist())
        observation, reward, terminated, truncated, extras = env.step(
            torch.zeros((args.num_envs, cfg.action_space), device=env.device)
        )
        logged = extras.get("log", {})
        for name in diagnostic_prefixes:
            value = logged.get(name)
            if value is not None:
                diagnostics_max[name] = max(
                    diagnostics_max[name], float(torch.as_tensor(value).item())
                )
    force_matrix = env.object_contacts.data.force_matrix_w
    terminal_expected = args.steps >= env.max_episode_length - 1
    terminal_runtime_checks = {
        "terminal_snapshot_present": bool(
            not terminal_expected
            or torch.all(env.last_terminal_audited_step == env.max_episode_length - 1).item()
        ),
        "terminal_phase_is_right_ablation": bool(
            not terminal_expected or torch.all(env.last_terminal_phase == 17).item()
        ),
        "all_translation_challenges_measured": bool(
            not terminal_expected
            or torch.all(env.last_terminal_translation_count == 6).item()
        ),
        "all_rotation_challenges_measured": bool(
            not terminal_expected or torch.all(env.last_terminal_rotation_count == 6).item()
        ),
        "physx_penetration_was_read": bool(
            not terminal_expected or torch.all(env.last_terminal_penetration_measured).item()
        ),
        "ineligible_zero_action_episode_failed_closed": bool(
            not terminal_expected or torch.all(~env.last_terminal_gate_pass).item()
        ),
    }
    result = {
        "schema": "xhand_rl_embedded_env_smoke_v2",
        "object": args.object,
        "steps": args.steps,
        "observation_shape": list(observation["policy"].shape),
        "reward_shape": list(reward.shape),
        "joint_count": len(env.robot.joint_names),
        "contact_filter_count": env.object_contacts.contact_physx_view.filter_count,
        "contact_force_matrix_shape": None if force_matrix is None else list(force_matrix.shape),
        "phase_codes_observed": sorted(phases),
        "penetration_measured": env.penetration_measured.detach().cpu().tolist(),
        "finite_translation_challenges": torch.isfinite(env.translation_displacements).sum(dim=-1).detach().cpu().tolist(),
        "finite_rotation_challenges": torch.isfinite(env.rotation_displacements).sum(dim=-1).detach().cpu().tolist(),
        "force_closure_evaluated": [report is not None for report in env.force_closure_reports],
        "terminal_audited_policy_step": env.last_terminal_audited_step.detach().cpu().tolist(),
        "terminal_phase_code": env.last_terminal_phase.detach().cpu().tolist(),
        "terminal_translation_challenge_count": env.last_terminal_translation_count.detach().cpu().tolist(),
        "terminal_rotation_challenge_count": env.last_terminal_rotation_count.detach().cpu().tolist(),
        "terminal_penetration_measured": env.last_terminal_penetration_measured.detach().cpu().tolist(),
        "terminal_force_closure_evaluated": env.last_terminal_force_closure_evaluated.detach().cpu().tolist(),
        "terminal_gate_pass": env.last_terminal_gate_pass.detach().cpu().tolist(),
        "zero_action_hard_prefix_max": diagnostics_max,
        "runtime_checks": terminal_runtime_checks,
        "ready": all(terminal_runtime_checks.values()),
        "terminated": terminated.detach().cpu().tolist(),
        "truncated": truncated.detach().cpu().tolist(),
        "production_artifacts_written": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
    simulation_app.close()
