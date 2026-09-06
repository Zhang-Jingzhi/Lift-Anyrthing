#!/usr/bin/env python3
"""Export an exact RL prefix-state buffer into the renderer JSON schema.

This is intentionally a visualization-only conversion.  It does not mark the
prefix as an accepted sample and it does not modify any existing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(args.prefix_state, map_location="cpu", weights_only=False)
    required = (
        "object",
        "episode_length",
        "phase_boundaries",
        "trajectory_joint_q",
        "trajectory_object_state",
        "trajectory_actions",
        "trajectory_phase",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"prefix state is missing fields: {missing}")

    # Isaac Lab's articulation order is the depth-first USD order (arm joints
    # interleaved by side, then hand joints grouped by kinematic depth).  It is
    # not the XML declaration order of the visual URDF.  Keep this list
    # explicit so a visualization cannot silently permute the recorded q.
    joint_names = [
        "left_j1", "right_j1", "left_j2", "right_j2", "left_j3", "right_j3",
        "left_j4", "right_j4", "left_j5", "right_j5", "left_j6", "right_j6",
        "left_j7", "right_j7", "left_hand_index_bend_joint", "left_hand_mid_joint1",
        "left_hand_pinky_joint1", "left_hand_ring_joint1", "left_hand_thumb_bend_joint",
        "right_hand_index_bend_joint", "right_hand_mid_joint1", "right_hand_pinky_joint1",
        "right_hand_ring_joint1", "right_hand_thumb_bend_joint", "left_hand_index_joint1",
        "left_hand_mid_joint2", "left_hand_pinky_joint2", "left_hand_ring_joint2",
        "left_hand_thumb_rota_joint1", "right_hand_index_joint1", "right_hand_mid_joint2",
        "right_hand_pinky_joint2", "right_hand_ring_joint2", "right_hand_thumb_rota_joint1",
        "left_hand_index_joint2", "left_hand_thumb_rota_joint2", "right_hand_index_joint2",
        "right_hand_thumb_rota_joint2",
    ]
    q = payload["trajectory_joint_q"]
    obj = payload["trajectory_object_state"]
    actions = payload["trajectory_actions"]
    phase = payload["trajectory_phase"]
    episode_length = int(payload["episode_length"])
    if q.ndim != 2 or q.shape[1] != len(joint_names):
        raise RuntimeError(f"unexpected joint trajectory shape: {tuple(q.shape)}")
    if obj.ndim != 2 or obj.shape[1] != 13:
        raise RuntimeError(f"unexpected object trajectory shape: {tuple(obj.shape)}")
    if actions.shape != q.shape or phase.ndim != 1:
        raise RuntimeError("trajectory buffers have inconsistent shapes")
    count = min(episode_length, q.shape[0], obj.shape[0], actions.shape[0], phase.shape[0])
    origin = payload.get("env_origin", torch.zeros(3, dtype=torch.float32))
    origin = torch.as_tensor(origin, dtype=torch.float32)

    # Keep only the completed approach/close/lift/hold portion represented by
    # this level-4 prefix.  Prefixes at level 3/4 do not contain disturbance or
    # ablation states, and those later phases must not be fabricated.
    stage_by_code = {0: "pregrasp", 1: "closure", 2: "lift", 3: "hold"}
    states = []
    for step in range(count):
        code = int(phase[step].item())
        if code not in stage_by_code:
            continue
        state = obj[step]
        pos = (state[:3] - origin).tolist()
        quat = state[3:7].tolist()
        states.append(
            {
                "step": int(step),
                "stage": stage_by_code[code],
                "joint_positions": q[step].tolist(),
                "object_position_world_m": pos,
                "object_quaternion_world_wxyz": quat,
                "policy_actions": actions[step].tolist(),
            }
        )
    if not states or {row["stage"] for row in states} != {"pregrasp", "closure", "lift", "hold"}:
        raise RuntimeError("prefix does not contain all four renderable stages")

    out = {
        "schema": "xhand_rl_exact_prefix_visualization_trajectory_v1",
        "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v2",
        "object": payload["object"],
        "simulation_dt_s": 0.002,
        "policy_dt_s": 0.008,
        "joint_names": joint_names,
        "states": states,
        "source_prefix_state": str(args.prefix_state.resolve()),
        "source_prefix_level": int(payload["prefix_level"]),
        "source_episode_length": episode_length,
        "visualization_only": True,
        "accepted_sample": False,
        "note": "Exact recorded Isaac Lab states through hold; no disturbance/ablation states are fabricated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(args.output)
    print(f"states={len(states)} stages={sorted({row['stage'] for row in states})}")


if __name__ == "__main__":
    main()
