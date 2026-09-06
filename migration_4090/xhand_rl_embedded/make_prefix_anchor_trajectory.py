#!/usr/bin/env python3
"""Build a visualization-only trajectory from a prefix state's saved anchors.

Some early curriculum prefix snapshots contain the four anchor joint states but
not the per-step trajectory buffers.  This utility interpolates those anchors
for visualization and explicitly labels the result as reconstructed; it never
changes acceptance metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


JOINT_NAMES = [
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


def as_list(value):
    return torch.as_tensor(value, dtype=torch.float32).tolist()


def lerp(a, b, t: float):
    aa = torch.as_tensor(a, dtype=torch.float32)
    bb = torch.as_tensor(b, dtype=torch.float32)
    return ((1.0 - t) * aa + t * bb).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(args.prefix_state, map_location="cpu", weights_only=False)
    required = ("object", "prefix_level", "env_origin", "pregrasp_q", "closure_q", "lift_q", "hold_robot_q", "initial_object_pos", "lift_object_position", "hold_position", "hold_object_state")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"prefix state is missing fields: {missing}")
    q0 = as_list(payload["pregrasp_q"])
    q1 = as_list(payload["closure_q"])
    q2 = as_list(payload["lift_q"])
    q3 = as_list(payload["hold_robot_q"])
    if len(q0) != len(JOINT_NAMES) or any(len(q) != len(JOINT_NAMES) for q in (q1, q2, q3)):
        raise RuntimeError("unexpected 38-DoF anchor shape")

    origin = torch.as_tensor(payload["env_origin"], dtype=torch.float32)
    initial = torch.as_tensor(payload["initial_object_pos"], dtype=torch.float32)
    lift = torch.as_tensor(payload["lift_object_position"], dtype=torch.float32) - origin
    hold = torch.as_tensor(payload["hold_position"], dtype=torch.float32) - origin
    # The curriculum snapshot does not preserve a pregrasp quaternion; the
    # hold quaternion is the only exact orientation available, so keep it
    # constant rather than inventing rotational motion.
    quat = as_list(torch.as_tensor(payload["hold_object_state"], dtype=torch.float32)[3:7])

    states = []
    step = 0
    boundaries = [int(v) for v in payload.get("phase_boundaries", [60, 200, 320, 400])]
    if len(boundaries) < 4 or not (boundaries[0] > 0 and boundaries[1] > boundaries[0] and boundaries[2] > boundaries[1] and boundaries[3] > boundaries[2]):
        raise RuntimeError(f"invalid phase boundaries: {boundaries}")
    counts = (boundaries[0], boundaries[1] - boundaries[0], boundaries[2] - boundaries[1], boundaries[3] - boundaries[2])
    stages = (("pregrasp", q0, q0, initial, initial, counts[0]), ("closure", q0, q1, initial, initial, counts[1]), ("lift", q1, q2, initial, lift, counts[2]), ("hold", q2, q3, lift, hold, counts[3]))
    for stage, qa, qb, pa, pb, count in stages:
        for index in range(count):
            t = 0.0 if count == 1 else index / float(count - 1)
            pos = lerp(pa, pb, t)
            states.append({
                "step": step,
                "stage": stage,
                "joint_positions": lerp(qa, qb, t),
                "object_position_world_m": pos,
                "object_quaternion_world_wxyz": quat,
                "policy_actions": [0.0] * len(JOINT_NAMES),
            })
            step += 1
    output = {
        "schema": "xhand_rl_prefix_anchor_visualization_trajectory_v1",
        "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v2",
        "object": payload["object"],
        "simulation_dt_s": 0.002,
        "policy_dt_s": 0.008,
        "joint_names": JOINT_NAMES,
        "states": states,
        "source_prefix_state": str(args.prefix_state.resolve()),
        "source_prefix_level": int(payload["prefix_level"]),
        "visualization_only": True,
        "accepted_sample": False,
        "reconstructed_from_anchors": True,
        "note": "Anchor interpolation only; the original per-step trajectory was not stored in this early curriculum prefix snapshot.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)
    print(f"states={len(states)} source_level={payload['prefix_level']}")


if __name__ == "__main__":
    main()
