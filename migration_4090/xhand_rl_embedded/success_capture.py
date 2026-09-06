"""Materialize an exact successful rollout already observed inside Isaac Lab.

Training and standalone collection share the same environment-side terminal
contract.  This module keeps the successful trajectory itself, instead of
reducing it to a scalar and later trying to reproduce it from a nearby model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import EMBEDDED_BACKEND
from .storage import materialize_success


def pose_matrix(position: list[float], quaternion_wxyz: list[float]) -> list[list[float]]:
    w, x, y, z = quaternion_wxyz
    q = np.asarray([w, x, y, z], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    output = np.eye(4)
    output[:3, :3] = rotation
    output[:3, 3] = position
    return output.tolist()


def merge_full_body(
    nominal: dict[str, Any],
    env_joint_names: list[str],
    key: str,
    learned: list[float],
) -> list[float]:
    source_names = list(nominal["joint_names"])
    base_values = nominal.get(key, nominal["full_body_q"])
    by_name = dict(zip(source_names, base_values))
    by_name.update(dict(zip(env_joint_names, learned)))
    return [float(by_name[name]) for name in source_names]


def materialize_completed_env_success(
    *,
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    object_name: str,
    nominal: dict[str, Any],
    env: Any,
    env_id: int,
    cfg: Any,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_selection_mode: str,
    training_selected_checkpoint: str,
    training_selected_checkpoint_sha256: str,
    training_hard_success_iteration: int | None,
    policy_action_mode: str,
    policy_std_scale: float,
    method: str,
) -> Path | None:
    """Persist one environment slot whose complete terminal gates passed."""

    report = copy.deepcopy(env.completed_reports[env_id])
    force_closure = copy.deepcopy(env.completed_force_closure_reports[env_id])
    if report is None or force_closure is None or not all(report["physics_gates"].values()):
        raise RuntimeError("environment emitted a success without complete embedded gates")
    if env.completed_trajectory_joint_q is None:
        raise RuntimeError("successful rollout was not recorded")

    pre = env.completed_pregrasp_q[env_id].detach().cpu().tolist()
    grasp = env.completed_closure_q[env_id].detach().cpu().tolist()
    lift = env.completed_lift_q[env_id].detach().cpu().tolist()
    start = env.completed_start_pose[env_id].detach().cpu().clone()
    episode_steps = int(env.completed_episode_steps[env_id].item())
    if episode_steps <= 0 or report.get("evaluated_policy_steps") != episode_steps:
        raise RuntimeError("environment emitted incomplete terminal trajectory metadata")

    origin = env.scene.env_origins[env_id].detach().cpu()
    start[:3] -= origin
    pre_full = merge_full_body(nominal, env.joint_names, "pregrasp_full_body_q", pre)
    grasp_full = merge_full_body(nominal, env.joint_names, "full_body_q", grasp)
    lift_full = merge_full_body(nominal, env.joint_names, "lift_full_body_q", lift)
    object_pose = pose_matrix(start[:3].tolist(), start[3:7].tolist())
    pose_hash = hashlib.sha256(
        json.dumps(
            {
                "pregrasp": pre_full,
                "grasp": grasp_full,
                "lift": lift_full,
                "object_pose": object_pose,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    object_states = (
        env.completed_trajectory_object_state[env_id, :episode_steps]
        .detach()
        .cpu()
        .clone()
    )
    object_states[:, :3] -= origin
    trajectory = {
        "schema": "xhand_rl_embedded_actual_trajectory_v2",
        "backend": EMBEDDED_BACKEND,
        "object": object_name,
        "simulation_dt_s": float(cfg.sim.dt),
        "policy_dt_s": float(cfg.sim.dt * cfg.decimation),
        "joint_names": list(env.joint_names),
        "states": {
            "joint_positions": env.completed_trajectory_joint_q[
                env_id, :episode_steps
            ].detach().cpu(),
            "object_root_state": object_states,
            "policy_actions": env.completed_trajectory_actions[
                env_id, :episode_steps
            ].detach().cpu(),
            "phase_code": env.completed_trajectory_phase[
                env_id, :episode_steps
            ].detach().cpu(),
        },
    }

    sample = copy.deepcopy(nominal)
    sample["joint_names"] = list(nominal["joint_names"])
    sample["pregrasp_full_body_q"] = pre_full
    sample["full_body_q"] = grasp_full
    sample["lift_full_body_q"] = lift_full
    sample["object_pose_world"] = object_pose
    sample["commanded_object_pose_world"] = copy.deepcopy(object_pose)
    sample["object_mesh_path"] = manifest["objects"][object_name]["mesh"]["path"]
    sample["object_name"] = object_name
    sample["method"] = method
    sample["generation_backend"] = EMBEDDED_BACKEND
    sample["reward_revision"] = manifest["algorithm"]["reward_revision"]
    sample["ppo_revision"] = manifest["algorithm"]["ppo_revision"]
    sample["training_reward_scale"] = manifest["algorithm"]["training_reward_scale"]
    sample["policy_checkpoint"] = str(checkpoint.resolve())
    sample["policy_checkpoint_sha256"] = checkpoint_sha256
    sample["policy_checkpoint_selection_mode"] = checkpoint_selection_mode
    sample["policy_training_selected_checkpoint"] = training_selected_checkpoint
    sample["policy_training_selected_checkpoint_sha256"] = (
        training_selected_checkpoint_sha256
    )
    sample["policy_training_hard_success_iteration"] = training_hard_success_iteration
    sample["policy_action_mode"] = policy_action_mode
    sample["policy_std_scale"] = float(policy_std_scale)
    sample["policy_candidate_pose_sha256"] = pose_hash
    sample["acceptance_profile"] = "embedded_physics_pass"
    sample["embedded_physics_pass"] = True
    sample["strict_visual_mesh_accepted"] = False
    sample["visual_mesh_audit_pending"] = True
    sample["physical_parameters"] = copy.deepcopy(manifest["physics"])
    return materialize_success(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        sample=sample,
        trajectory=trajectory,
        episode_report=report,
        force_closure_report=force_closure,
    )
