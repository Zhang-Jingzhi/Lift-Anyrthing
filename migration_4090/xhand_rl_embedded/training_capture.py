"""Capture exact hard-success PPO rollouts before the policy update."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .contracts import sha256_file
from .success_capture import materialize_completed_env_success


def install_exact_success_capture(
    *,
    runner: Any,
    env: Any,
    cfg: Any,
    manifest: dict[str, Any],
    nominal: dict[str, Any],
    generation_root: Path,
    manifest_path: Path,
    output: Path,
    object_name: str,
    start_iteration: int,
    adaptive_prefix_lr_factor: float = 0.25,
) -> None:
    """Persist the precise pre-update policy and trajectory of every hard pass.

    RSL-RL normally saves only after PPO updates and only every 100 iterations.
    A one-rollout hard pass can therefore disappear before any replayable model
    exists.  This hook runs after collection but before ``alg.update``.
    """

    if not 0.0 < float(adaptive_prefix_lr_factor) <= 1.0:
        raise ValueError("adaptive_prefix_lr_factor must be in (0, 1]")
    original_update = runner.alg.update
    # Keep the hook usable with lightweight test doubles and older RSL-RL
    # wrappers that do not expose the optimizer learning-rate attribute.
    base_learning_rate = getattr(runner.alg, "learning_rate", None)
    if base_learning_rate is not None:
        base_learning_rate = float(base_learning_rate)
    seen_serial = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    prefix_marker_path = output / "hard_prefix_rollouts.jsonl"
    prior_prefix_levels: list[int] = []
    if prefix_marker_path.is_file():
        for line in prefix_marker_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                prior_prefix_levels.append(int(json.loads(line)["max_level"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    state = {
        "iteration": int(start_iteration),
        "best_prefix_level": max(prior_prefix_levels, default=0),
    }
    marker_path = output / "hard_success_rollouts.jsonl"

    def update_with_exact_success_capture():
        iteration = int(state["iteration"])
        changed = env.completed_success_serial > seen_serial
        ids = torch.nonzero(changed, as_tuple=False).squeeze(-1)
        if ids.numel():
            success_serial = int(env.completed_success_serial[ids].max().item())
            checkpoint = output / (
                f"hard_success_preupdate_iter_{iteration}_serial_{success_serial}.pt"
            )
            runner.current_learning_iteration = iteration
            runner.save(
                str(checkpoint),
                infos={
                    "schema": "xhand_rl_exact_hard_success_checkpoint_v1",
                    "object": object_name,
                    "iteration": iteration,
                    "success_serial": success_serial,
                    "saved_before_ppo_update": True,
                },
            )
            checkpoint_hash = sha256_file(checkpoint)
            artifacts: list[str] = []
            for env_id in ids.tolist():
                path = materialize_completed_env_success(
                    root=generation_root,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    object_name=object_name,
                    nominal=nominal,
                    env=env,
                    env_id=env_id,
                    cfg=cfg,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_hash,
                    checkpoint_selection_mode="exact_training_rollout_preupdate",
                    training_selected_checkpoint=str(checkpoint.resolve()),
                    training_selected_checkpoint_sha256=checkpoint_hash,
                    training_hard_success_iteration=iteration,
                    policy_action_mode="stochastic_training_rollout",
                    policy_std_scale=1.0,
                    method="rl_training_rollout_embedded_physics",
                )
                seen_serial[env_id] = env.completed_success_serial[env_id]
                if path is not None:
                    artifacts.append(str(path.resolve()))
            with marker_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "xhand_rl_exact_hard_success_rollout_v1",
                            "time": time.time(),
                            "object": object_name,
                            "iteration": iteration,
                            "success_serial": success_serial,
                            "successful_env_ids": ids.tolist(),
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": checkpoint_hash,
                            "saved_before_ppo_update": True,
                            "materialized_samples": artifacts,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
        prefix_stats = env.consume_training_prefix_statistics()
        prefix_level = int(prefix_stats["max_level"])
        if 0 < prefix_level < 6 and prefix_level > int(state["best_prefix_level"]):
            checkpoint = output / (
                f"hard_prefix_preupdate_iter_{iteration}_level_{prefix_level}.pt"
            )
            runner.current_learning_iteration = iteration
            runner.save(
                str(checkpoint),
                infos={
                    "schema": "xhand_rl_exact_hard_prefix_checkpoint_v1",
                    "object": object_name,
                    "iteration": iteration,
                    "max_level": prefix_level,
                    "mean_score": float(prefix_stats["mean_score"]),
                    "sample_count": int(prefix_stats["sample_count"]),
                    "saved_before_ppo_update": True,
                },
            )
            checkpoint_hash = sha256_file(checkpoint)
            prefix_state = prefix_stats.get("snapshot")
            prefix_state_path = None
            prefix_state_hash = None
            if isinstance(prefix_state, dict):
                prefix_state_path = output / (
                    f"hard_prefix_state_preupdate_iter_{iteration}_level_{prefix_level}.pt"
                )
                torch.save(prefix_state, prefix_state_path)
                prefix_state_hash = sha256_file(prefix_state_path)
            with prefix_marker_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "schema": "xhand_rl_exact_hard_prefix_rollout_v1",
                            "time": time.time(),
                            "object": object_name,
                            "iteration": iteration,
                            "max_level": prefix_level,
                            "mean_score": float(prefix_stats["mean_score"]),
                            "sample_count": int(prefix_stats["sample_count"]),
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": checkpoint_hash,
                            "prefix_state": (
                                str(prefix_state_path.resolve())
                                if prefix_state_path is not None
                                else None
                            ),
                            "prefix_state_sha256": prefix_state_hash,
                            "prefix_state_schema": (
                                prefix_state.get("schema")
                                if isinstance(prefix_state, dict)
                                else None
                            ),
                            "prefix_state_episode_length": (
                                int(prefix_state.get("episode_length", 0))
                                if isinstance(prefix_state, dict)
                                else None
                            ),
                            "saved_before_ppo_update": True,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
            state["best_prefix_level"] = prefix_level
            # A new strict prefix is a rare, valuable basin.  The normal PPO
            # update that immediately follows collection can erase it (the
            # TensorBoard traces show contact/lift spikes collapsing after a
            # single update).  Keep the optimizer/storage semantics intact,
            # but reduce the learning rate before that update and all later
            # updates in this lineage.  This is training stabilization only;
            # strict acceptance and the saved pre-update prefix are unchanged.
            if base_learning_rate is not None and hasattr(runner.alg, "optimizer"):
                # The next gate still has to be learned.  A single global
                # 1e-6 floor protects the contact basin but freezes the
                # downstream force-closure/disturbance refinement after a
                # level-3 capture.  Use a level-aware floor: retain more
                # update budget while acquiring contact/lift/clearance, then
                # tighten only after stable force closure has been captured.
                if prefix_level >= 4:
                    level_floor = 5.0e-7
                elif prefix_level >= 3:
                    level_floor = 1.0e-6
                else:
                    level_floor = 2.0e-6
                protected_learning_rate = max(
                    level_floor,
                    base_learning_rate * float(adaptive_prefix_lr_factor),
                )
                runner.alg.learning_rate = protected_learning_rate
                for parameter_group in runner.alg.optimizer.param_groups:
                    parameter_group["lr"] = protected_learning_rate
        state["iteration"] = iteration + 1
        return original_update()

    runner.alg.update = update_with_exact_success_capture
