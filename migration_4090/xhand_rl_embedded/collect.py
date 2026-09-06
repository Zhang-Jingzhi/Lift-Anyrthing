#!/usr/bin/env python3
"""Directly materialize successful embedded-gate PPO episodes."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--object", required=True)
parser.add_argument("--nominal-dataset", type=Path, required=True)
parser.add_argument("--nominal-sample-index", type=int, default=0)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--allow-adjacent-hard-success-checkpoint",
    action="store_true",
    help="allow only the saved checkpoints immediately before/after persisted training hard success",
)
parser.add_argument(
    "--allow-exact-training-hard-success-checkpoint",
    action="store_true",
    help="allow an immutable pre-update checkpoint persisted with its exact passing PPO rollout",
)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--target", type=int, default=100)
parser.add_argument("--num-envs", type=int, default=64)
# A policy step advances all vectorized environments.  Two million policy
# steps is an unnecessarily long first collection attempt (tens of millions
# of episodes' worth of simulation time); failed attempts are handled by the
# bridge with independent PPO retries.  Keep each collection attempt bounded
# so the supervisor can adapt instead of parking a GPU for multiple days.
parser.add_argument("--max-steps", type=int, default=160000)
parser.add_argument("--seed", type=int, default=85)
parser.add_argument(
    "--policy-std-scale",
    type=float,
    default=1.0,
    help="explicit collection-only multiplier for the persisted stochastic policy std",
)
parser.add_argument(
    "--attempt-id",
    help="unique append-only identifier used to correlate bridge timing and heartbeats",
)
parser.add_argument("--penetration-reward-weight", type=float)
parser.add_argument("--penetration-clear-reward-weight", type=float)
parser.add_argument("--proximity-reward-weight", type=float)
parser.add_argument("--stable-lift-reward-weight", type=float)
parser.add_argument("--force-closure-reward-weight", type=float)
parser.add_argument("--hard-gate-frontier-reward-weight", type=float)
parser.add_argument("--disturbance-direction-reward-weight", type=float)
parser.add_argument(
    "--force-closure-shape-all-hold-contacts",
    action="store_true",
    default=None,
)
parser.add_argument(
    "--force-closure-requires-clearance",
    action="store_true",
    default=None,
    help="gate dense force-closure shaping on the same-trajectory strict clearance+gravity prefix",
)
parser.add_argument(
    "--disturbance-reward-requires-force-closure",
    action="store_true",
    default=None,
)
parser.add_argument("--instantaneous-residual-weight", type=float)
parser.add_argument("--residual-integration", type=float)
parser.add_argument("--residual-limit-rad", type=float)
parser.add_argument("--arm-action-scale-rad", type=float)
parser.add_argument("--hand-action-scale-rad", type=float)
parser.add_argument(
    "--collection-lock",
    type=Path,
    default=Path("/tmp/xhand_rl_embedded_kit/global_isaac_collection.lock"),
    help="host-wide advisory lock held before Isaac/Omniverse startup",
)
parser.add_argument(
    "--residual-activation-phase",
    type=int,
    choices=(0, 1, 2, 3),
    help="first phase in which residual actions are applied (3=hold refinement)",
)
parser.add_argument("--allow-partial", action="store_true")
parser.add_argument(
    "--kit-teardown-cooldown-s",
    type=float,
    default=0.0,
    help="hold the global Kit lock after App.close so driver resources can quiesce",
)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.policy_std_scale <= 0.0:
    raise ValueError("policy std scale must be positive")

# The bridge serializes collectors, and this second guard covers already
# running bridge processes plus manual ``collect`` invocations.  Acquire the
# lock before AppLauncher creates a Kit instance; otherwise simultaneous Kit
# startup/teardown can trigger carb.tasking mutex assertions.  flock is
# released automatically if this process is interrupted or killed.
args.attempt_id = args.attempt_id or f"manual-{args.object}-seed-{args.seed}-{int(time.time())}-{os.getpid()}"
args.collection_lock.parent.mkdir(parents=True, exist_ok=True)
_collection_lock_wait_started = time.monotonic()
_collection_lock_handle = args.collection_lock.open("a+")
fcntl.flock(_collection_lock_handle.fileno(), fcntl.LOCK_EX)
_collection_lock_wait_s = time.monotonic() - _collection_lock_wait_started

# Record lock acquisition before Kit startup.  This makes a collector that is
# waiting on another Isaac process distinguishable from one that crashed before
# entering the rollout loop, and gives every attempt an unambiguous lineage.
_early_heartbeat_path = args.root / "runs" / args.object / "collection_heartbeat.jsonl"
_early_heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
with _early_heartbeat_path.open("a") as _handle:
    _handle.write(
        json.dumps(
            {
                "schema": "xhand_rl_embedded_collection_heartbeat_v1",
                "time": time.time(),
                "object": args.object,
                "attempt_id": args.attempt_id,
                "seed": args.seed,
                "policy_std_scale": args.policy_std_scale,
                "stage": "lock_acquired",
                "lock_wait_s": _collection_lock_wait_s,
                "final": False,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    _handle.flush()


def _release_collection_lock() -> None:
    try:
        fcntl.flock(_collection_lock_handle.fileno(), fcntl.LOCK_UN)
    finally:
        _collection_lock_handle.close()


atexit.register(_release_collection_lock)
from .launcher import configure_isolated_kit

configure_isolated_kit(args, label=f"collect_{args.object}")
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .configuration import configure_env
from .checkpoint_selection import validate_collection_checkpoint
from .contracts import validate_manifest, validate_root
from .env import XHandEmbeddedEnv, XHandEmbeddedEnvCfg
from .ppo_cfg import XHandEmbeddedPPORunnerCfg
from .storage import accepted_count
from .success_capture import materialize_completed_env_success


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    validate_root(args.root, args.manifest, manifest)
    if args.object not in manifest["objects"]:
        raise ValueError(f"unknown locked object: {args.object}")
    for path in (args.nominal_dataset, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_validation = validate_collection_checkpoint(
        args.checkpoint,
        allow_adjacent_hard_success=args.allow_adjacent_hard_success_checkpoint,
        allow_exact_training_hard_success=args.allow_exact_training_hard_success_checkpoint,
    )
    checkpoint_hash = checkpoint_validation["checkpoint_sha256"]
    payload = torch.load(args.nominal_dataset, map_location="cpu", weights_only=False)
    nominal = (payload["samples"] if isinstance(payload, dict) else payload)[args.nominal_sample_index]
    cfg = configure_env(
        XHandEmbeddedEnvCfg(),
        manifest=manifest,
        object_name=args.object,
        nominal_dataset=args.nominal_dataset,
        nominal_sample_index=args.nominal_sample_index,
        num_envs=args.num_envs,
        device=args.device or "cuda:0",
        seed=args.seed,
        record_trajectory=True,
    )
    # Collection must replay the residual semantics used to train the
    # selected checkpoint.  The bridge passes explicit values for new
    # retries; this fallback also handles manual collection invocations.
    checkpoint_provenance = args.checkpoint.parent / "training_provenance.json"
    provenance_candidates = [checkpoint_provenance]
    # Exact curriculum hard-success seed probes have no formal
    # training_provenance.json beside the checkpoint.  Reuse the immutable
    # curriculum attempt metadata so residual units match the policy that
    # produced the saved rollout.
    provenance_candidates.extend(
        sorted(args.checkpoint.parent.glob("curriculum_training_attempt_*.json"), reverse=True)
    )
    checkpoint_provenance = next((path for path in provenance_candidates if path.is_file()), None)
    if checkpoint_provenance is not None:
        try:
            trained = json.loads(checkpoint_provenance.read_text())
        except (OSError, TypeError, ValueError):
            trained = {}
        if args.instantaneous_residual_weight is None:
            cfg.instantaneous_residual_weight = float(
                trained.get(
                    "instantaneous_residual_weight",
                    cfg.instantaneous_residual_weight,
                )
            )
        if args.residual_integration is None:
            cfg.residual_integration = float(
                trained.get("residual_integration", cfg.residual_integration)
            )
        if args.residual_limit_rad is None:
            cfg.residual_limit_rad = float(
                trained.get("residual_limit_rad", cfg.residual_limit_rad)
            )
        if args.residual_activation_phase is None:
            cfg.residual_activation_phase = int(
                trained.get("residual_activation_phase", cfg.residual_activation_phase)
            )
        if args.stable_lift_reward_weight is None:
            cfg.stable_lift_reward_weight = float(
                trained.get(
                    "stable_lift_reward_weight", cfg.stable_lift_reward_weight
                )
            )
        if args.penetration_clear_reward_weight is None:
            cfg.penetration_clear_reward_weight = float(
                trained.get(
                    "penetration_clear_reward_weight",
                    cfg.penetration_clear_reward_weight,
                )
            )
        if args.proximity_reward_weight is None:
            cfg.proximity_reward_weight = float(
                trained.get("proximity_reward_weight", cfg.proximity_reward_weight)
            )
        if args.force_closure_reward_weight is None:
            cfg.force_closure_reward_weight = float(
                trained.get(
                    "force_closure_reward_weight", cfg.force_closure_reward_weight
                )
            )
        if args.force_closure_shape_all_hold_contacts is None:
            cfg.force_closure_shape_all_hold_contacts = bool(
                trained.get(
                    "force_closure_shape_all_hold_contacts",
                    cfg.force_closure_shape_all_hold_contacts,
                )
            )
        if args.force_closure_requires_clearance is None:
            cfg.force_closure_reward_requires_clearance = bool(
                trained.get(
                    "force_closure_reward_requires_clearance",
                    cfg.force_closure_reward_requires_clearance,
                )
            )
        if args.disturbance_reward_requires_force_closure is None:
            cfg.disturbance_reward_requires_force_closure = bool(
                trained.get(
                    "disturbance_reward_requires_force_closure",
                    cfg.disturbance_reward_requires_force_closure,
                )
            )
        if args.hard_gate_frontier_reward_weight is None:
            cfg.hard_gate_frontier_reward_weight = float(
                trained.get(
                    "hard_gate_frontier_reward_weight",
                    cfg.hard_gate_frontier_reward_weight,
                )
            )
        if args.disturbance_direction_reward_weight is None:
            cfg.disturbance_direction_reward_weight = float(
                trained.get(
                    "disturbance_direction_reward_weight",
                    cfg.disturbance_direction_reward_weight,
                )
            )
    if args.penetration_reward_weight is not None:
        if args.penetration_reward_weight >= 0.0:
            raise ValueError("penetration reward weight must be negative")
        cfg.penetration_reward_weight = float(args.penetration_reward_weight)
    if args.penetration_clear_reward_weight is not None:
        if args.penetration_clear_reward_weight < 0.0:
            raise ValueError("penetration-clear reward weight must be non-negative")
        cfg.penetration_clear_reward_weight = float(args.penetration_clear_reward_weight)
    if args.proximity_reward_weight is not None:
        if args.proximity_reward_weight < 0.0:
            raise ValueError("proximity reward weight must be non-negative")
        cfg.proximity_reward_weight = float(args.proximity_reward_weight)
    if args.stable_lift_reward_weight is not None:
        if args.stable_lift_reward_weight < 0.0:
            raise ValueError("stable lift reward weight must be non-negative")
        cfg.stable_lift_reward_weight = float(args.stable_lift_reward_weight)
    if args.force_closure_reward_weight is not None:
        if args.force_closure_reward_weight < 0.0:
            raise ValueError("force closure reward weight must be non-negative")
        cfg.force_closure_reward_weight = float(args.force_closure_reward_weight)
    if args.force_closure_shape_all_hold_contacts is not None:
        cfg.force_closure_shape_all_hold_contacts = bool(
            args.force_closure_shape_all_hold_contacts
        )
    if args.force_closure_requires_clearance is not None:
        cfg.force_closure_reward_requires_clearance = bool(
            args.force_closure_requires_clearance
        )
    if args.disturbance_reward_requires_force_closure is not None:
        cfg.disturbance_reward_requires_force_closure = bool(
            args.disturbance_reward_requires_force_closure
        )
    if args.hard_gate_frontier_reward_weight is not None:
        if args.hard_gate_frontier_reward_weight < 0.0:
            raise ValueError("hard gate frontier reward weight must be non-negative")
        cfg.hard_gate_frontier_reward_weight = float(
            args.hard_gate_frontier_reward_weight
        )
    if args.disturbance_direction_reward_weight is not None:
        if args.disturbance_direction_reward_weight < 0.0:
            raise ValueError("disturbance direction reward weight must be non-negative")
        cfg.disturbance_direction_reward_weight = float(
            args.disturbance_direction_reward_weight
        )
    if args.instantaneous_residual_weight is not None:
        if args.instantaneous_residual_weight < 0.0:
            raise ValueError("instantaneous residual weight must be non-negative")
        cfg.instantaneous_residual_weight = float(args.instantaneous_residual_weight)
    if args.residual_integration is not None:
        if args.residual_integration < 0.0:
            raise ValueError("residual integration must be non-negative")
        cfg.residual_integration = float(args.residual_integration)
    if args.residual_limit_rad is not None:
        if args.residual_limit_rad <= 0.0:
            raise ValueError("residual limit must be positive")
        cfg.residual_limit_rad = float(args.residual_limit_rad)
    if args.arm_action_scale_rad is not None:
        if args.arm_action_scale_rad <= 0.0:
            raise ValueError("arm action scale must be positive")
        cfg.arm_action_scale_rad = float(args.arm_action_scale_rad)
    if args.hand_action_scale_rad is not None:
        if args.hand_action_scale_rad <= 0.0:
            raise ValueError("hand action scale must be positive")
        cfg.hand_action_scale_rad = float(args.hand_action_scale_rad)
    if args.residual_activation_phase is not None:
        cfg.residual_activation_phase = int(args.residual_activation_phase)
    runner_cfg = XHandEmbeddedPPORunnerCfg()
    runner_cfg.seed = args.seed
    runner_cfg.device = cfg.sim.device
    env = XHandEmbeddedEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    if args.policy_std_scale != 1.0:
        with torch.no_grad():
            runner.alg.policy.std.mul_(float(args.policy_std_scale))
    runner.eval_mode()
    # Training hard successes are sampled from PPO's stochastic action
    # distribution. Collection must sample that same distribution instead of
    # silently switching to the deterministic mean action.
    policy = runner.alg.policy.act
    observations = wrapped.get_observations()
    seen_serial = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    heartbeat_path = args.root / "runs" / args.object / "collection_heartbeat.jsonl"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_started = time.monotonic()
    heartbeat_last = heartbeat_started
    heartbeat_steps = 0
    latest_diagnostics: dict[str, float] = {}
    diagnostic_max: dict[str, float] = {}
    diagnostic_min: dict[str, float] = {}
    diagnostic_keys = (
        "reward/bilateral_contact",
        "reward/contact_continuity",
        "reward/lift_height",
        "reward/hold_stability",
        "reward/gravity_margin",
        "reward/penetration",
        "reward/penetration_clear",
        "reward/force_closure",
        "reward/hard_gate_frontier",
        "reward/terminal_success",
        "embedded/gate_bilateral_contact_continuity_fraction",
        "embedded/gate_arm_joint_lift_fraction",
        "embedded/gate_lift_height_fraction",
        "embedded/gate_gravity_hold_fraction",
        "embedded/gate_translation_disturbance_fraction",
        "embedded/gate_rotation_disturbance_fraction",
        "embedded/gate_physx_penetration_fraction",
        "embedded/gate_formal_force_closure_fraction",
        "embedded/gate_single_hand_ablations_fraction",
        "embedded/terminal_success_fraction",
    )
    for step in range(args.max_steps):
        if accepted_count(args.root, args.object) >= args.target:
            break
        with torch.inference_mode():
            actions = policy(observations)
            observations, _, _, extras = wrapped.step(actions)
        log_values = extras.get("log", {}) if isinstance(extras, dict) else {}
        for key in diagnostic_keys:
            value = log_values.get(key)
            if value is None:
                continue
            try:
                scalar = float(value.item() if hasattr(value, "item") else value)
                latest_diagnostics[key] = scalar
                diagnostic_max[key] = max(diagnostic_max.get(key, scalar), scalar)
                diagnostic_min[key] = min(diagnostic_min.get(key, scalar), scalar)
            except (TypeError, ValueError, RuntimeError):
                continue
        heartbeat_steps = step + 1
        now = time.monotonic()
        # Keep collection externally observable without printing every Isaac
        # step.  This is append-only and is deliberately separate from the
        # immutable accepted-sample artifacts.
        if heartbeat_steps % 10000 == 0 or now - heartbeat_last >= 60.0:
            row = {
                "schema": "xhand_rl_embedded_collection_heartbeat_v1",
                "time": time.time(),
                "object": args.object,
                "attempt_id": args.attempt_id,
                "seed": args.seed,
                "policy_std_scale": args.policy_std_scale,
                "policy_steps": heartbeat_steps,
                "max_steps": args.max_steps,
                "target": args.target,
                "accepted": accepted_count(args.root, args.object),
                "num_envs": args.num_envs,
                "elapsed_s": now - heartbeat_started,
                "diagnostics": latest_diagnostics,
                "diagnostics_max": diagnostic_max,
                "diagnostics_min": diagnostic_min,
            }
            with heartbeat_path.open("a") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
            heartbeat_last = now
        changed = env.completed_success_serial > seen_serial
        ids = torch.nonzero(changed, as_tuple=False).squeeze(-1)
        for env_id in ids.tolist():
            seen_serial[env_id] = env.completed_success_serial[env_id]
            materialize_completed_env_success(
                root=args.root,
                manifest_path=args.manifest.resolve(),
                manifest=manifest,
                object_name=args.object,
                nominal=nominal,
                env=env,
                env_id=env_id,
                cfg=cfg,
                checkpoint=args.checkpoint.resolve(),
                checkpoint_sha256=checkpoint_hash,
                checkpoint_selection_mode=checkpoint_validation["mode"],
                training_selected_checkpoint=checkpoint_validation[
                    "training_selected_checkpoint"
                ],
                training_selected_checkpoint_sha256=checkpoint_validation[
                    "training_selected_checkpoint_sha256"
                ],
                training_hard_success_iteration=checkpoint_validation[
                    "training_hard_success_iteration"
                ],
                policy_action_mode="stochastic",
                policy_std_scale=float(args.policy_std_scale),
                method="rl_policy_embedded_physics",
            )
            if accepted_count(args.root, args.object) >= args.target:
                break
    # Always leave a final append-only status row, including the max-step
    # failure case, so the bridge supervisor can distinguish progress from a
    # process that exited before entering the rollout loop.
    with heartbeat_path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "xhand_rl_embedded_collection_heartbeat_v1",
                    "time": time.time(),
                    "object": args.object,
                    "attempt_id": args.attempt_id,
                    "seed": args.seed,
                    "policy_std_scale": args.policy_std_scale,
                    "policy_steps": heartbeat_steps,
                    "max_steps": args.max_steps,
                    "target": args.target,
                    "accepted": accepted_count(args.root, args.object),
                    "num_envs": args.num_envs,
                    "elapsed_s": time.monotonic() - heartbeat_started,
                    "diagnostics": latest_diagnostics,
                    "diagnostics_max": diagnostic_max,
                    "diagnostics_min": diagnostic_min,
                    "lock_wait_s": _collection_lock_wait_s,
                    "exit_reason": (
                        "target_reached"
                        if accepted_count(args.root, args.object) >= args.target
                        else "max_steps_reached"
                    ),
                    "final": True,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
    count = accepted_count(args.root, args.object)
    if count < args.target:
        env.close()
        if args.allow_partial:
            return
        raise RuntimeError(f"collector reached max steps with {count}/{args.target} embedded passes")
    run_root = args.root / "runs" / args.object
    (run_root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "xhand_rl_embedded_object_complete_v2",
                "object": args.object,
                "accepted": count,
                "target": args.target,
            },
            indent=2,
        )
        + "\n"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Isaac/PhysX releases some driver resources asynchronously.  Keep the
        # collection flock held through App.close and an optional cooldown so a
        # following collector/auditor cannot enter the known tcache/mutex race.
        simulation_app.close()
        if args.kit_teardown_cooldown_s > 0.0:
            time.sleep(args.kit_teardown_cooldown_s)
