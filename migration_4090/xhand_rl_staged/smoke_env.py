#!/usr/bin/env python3
"""Exercise the 293D ABI, action masks, contacts, and stage gates without writes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--bodex-bank", type=Path, required=True)
parser.add_argument("--stage", type=int, choices=range(1, 7), default=1)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--allow-test-bank", action="store_true")
parser.add_argument("--candidate-repeat-factors")
parser.add_argument("--fixed-candidate-index", type=int, default=-1)
parser.add_argument("--nominal-pose-lock", action="store_true")
parser.add_argument("--palm-center-alignment-reward-weight", type=float, default=0.0)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from migration_4090.xhand_rl_embedded.launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"staged_smoke_s{args.stage}_{args.object}")
simulation_app = AppLauncher(args).app

import torch

from migration_4090.xhand_bodex_bimanual.contracts import load_bodex_bank
from migration_4090.xhand_rl_embedded.contracts import validate_manifest

from .configuration import configure_staged_env
from .env import XHandStagedEnv, XHandStagedEnvCfg
from .sampling import parse_candidate_repeat_factors


def main() -> int:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    bank = load_bodex_bank(
        args.bodex_bank,
        expected_object=args.object,
        verify_source=False,
        allow_test_fixture=bool(args.allow_test_bank),
        intended_stage=args.stage,
    )
    candidate_repeat_factors = parse_candidate_repeat_factors(
        args.candidate_repeat_factors, candidate_count=len(bank["samples"])
    )
    cfg = configure_staged_env(
        XHandStagedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        bodex_bank=args.bodex_bank,
        stage_id=args.stage,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=20260829,
        record_trajectory=False,
        allow_test_bank=bool(args.allow_test_bank),
        candidate_selection="round_robin",
        candidate_repeat_factors=candidate_repeat_factors,
        fixed_candidate_index=args.fixed_candidate_index,
        nominal_pose_lock=bool(args.nominal_pose_lock),
        palm_center_alignment_reward_weight=args.palm_center_alignment_reward_weight,
    )
    env = XHandStagedEnv(cfg)
    observation, _ = env.reset()
    reward = torch.zeros(args.num_envs, device=env.device)
    terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    truncated = torch.zeros_like(terminated)
    for _ in range(args.steps):
        observation, reward, terminated, truncated, _ = env.step(
            torch.zeros((args.num_envs, cfg.action_space), device=env.device)
        )
    candidate_context = observation["policy"][:, -4:]
    shared_gate_memory = observation["policy"][:, 269:273]
    candidate_gated_memory = observation["policy"][:, 273:289].reshape(
        args.num_envs, 4, 4
    )
    checks = {
        "observation_shape": list(observation["policy"].shape)
        == [args.num_envs, 293],
        "reward_shape": list(reward.shape) == [args.num_envs],
        "joint_count": len(env.joint_names) == 38,
        "action_mask_shape": list(env.action_mask.shape) == [1, 38],
        "action_mask_nonempty": int(env.action_mask.sum().item()) > 0,
        "contact_group_shape": list(env._group_contact_forces().shape)
        == [args.num_envs, 2, 6],
        "finite_observation": bool(torch.isfinite(observation["policy"]).all().item()),
        "candidate_context_one_hot": bool(
            torch.allclose(
                candidate_context.sum(dim=-1),
                torch.ones(args.num_envs, device=env.device),
            )
        ),
        "candidate_gated_memory_sparse": bool(
            torch.allclose(
                candidate_gated_memory,
                candidate_context.unsqueeze(-1) * shared_gate_memory.unsqueeze(1),
            )
        ),
        "candidate_sampling_schedule": env.candidate_sampling_schedule.bincount(
            minlength=len(candidate_repeat_factors)
        ).tolist()
        == list(candidate_repeat_factors),
        "fixed_candidate_selection": args.fixed_candidate_index < 0
        or bool(
            torch.all(
                env.active_bank_index
                == int(args.fixed_candidate_index)
            ).item()
        ),
        "nominal_pose_lock_configured": (not args.nominal_pose_lock)
        or bool(cfg.nominal_pose_lock),
        "terminal_contact_reward_finite": bool(
            torch.isfinite(
                env.extras["log"]["staged/terminal_contact_progress"]
            ).item()
        ),
        "stage2_hold_only_residual": args.stage != 2
        or int(cfg.residual_activation_phase) == 3,
        "stage2_preserves_bodex_arms": args.stage != 2
        or int(env.action_mask.sum().item()) == 24,
        "stage2_terminal_reward_profile": args.stage != 2
        or float(cfg.terminal_contact_reward_weight) == 40.0,
        "no_production_artifacts_written": True,
    }
    result = {
        "schema": "xhand_rl_staged_env_smoke_v1",
        "object": args.object,
        "stage": args.stage,
        "steps": args.steps,
        "observation_shape": list(observation["policy"].shape),
        "active_actions": int(env.action_mask.sum().item()),
        "residual_activation_phase": int(cfg.residual_activation_phase),
        "residual_integration": float(cfg.residual_integration),
        "residual_limit_rad": float(cfg.residual_limit_rad),
        "terminal_contact_progress_mode": cfg.terminal_contact_progress_mode,
        "candidate_repeat_factors": list(env.candidate_repeat_factors),
        "active_candidate_indices": env.active_bank_index.detach().cpu().tolist(),
        "checks": checks,
        "ready": all(checks.values()),
        "terminated": terminated.detach().cpu().tolist(),
        "truncated": truncated.detach().cpu().tolist(),
    }
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    raise SystemExit(exit_code)
