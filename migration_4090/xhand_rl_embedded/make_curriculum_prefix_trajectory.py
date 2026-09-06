#!/usr/bin/env python3
"""Build a clearly-labelled visualization trajectory from an RL curriculum prefix.

Curriculum prefixes store phase endpoint states rather than the formal trainer's
per-step buffers.  This tool interpolates only those recorded endpoints for a
visual diagnostic; it never marks the result accepted or changes source data.
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


def unit_quat(values):
    t = torch.as_tensor(values, dtype=torch.float32)
    return (t / torch.linalg.vector_norm(t).clamp_min(1e-8)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-state", type=Path, required=True)
    ap.add_argument("--nominal", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    x = torch.load(args.prefix_state, map_location="cpu", weights_only=False)
    required = ("object", "pregrasp_q", "closure_q", "lift_q", "hold_robot_q",
                "initial_object_pos", "lift_object_position", "hold_position",
                "hold_quaternion", "env_origin", "phase_boundaries")
    missing = [key for key in required if key not in x]
    if missing:
        raise RuntimeError(f"prefix missing fields: {missing}")
    nominal = torch.load(args.nominal, map_location="cpu", weights_only=False)
    sample = nominal["samples"][0]
    q_anchors = [x["pregrasp_q"], x["closure_q"], x["lift_q"], x["hold_robot_q"]]
    if any(tuple(q.shape) != (38,) for q in q_anchors):
        raise RuntimeError("curriculum endpoint q must all have shape [38]")
    origin = torch.as_tensor(x["env_origin"], dtype=torch.float32)
    p0 = torch.as_tensor(x["initial_object_pos"], dtype=torch.float32)
    p1 = torch.as_tensor(x["lift_object_position"], dtype=torch.float32) - origin
    p2 = torch.as_tensor(x["hold_position"], dtype=torch.float32) - origin
    q0 = torch.tensor(sample.get("nominal_object_quaternion_wxyz", [1, 0, 0, 0]), dtype=torch.float32)
    q2 = torch.as_tensor(x["hold_quaternion"], dtype=torch.float32)
    q2 = q2 / torch.linalg.vector_norm(q2).clamp_min(1e-8)
    bounds = list(map(int, x["phase_boundaries"]))
    # Curriculum schedule has approach, close, lift, hold endpoints at these
    # boundaries.  The final endpoint is excluded because the prefix is 399
    # steps in the recorded curriculum runs.
    n = max(1, min(int(x.get("episode_length", bounds[3] - 1)), bounds[3]))
    states = []
    for step in range(n):
        if step < bounds[0]:
            stage, a, b, u, pa, pb = "pregrasp", 0, 1, step / max(bounds[0] - 1, 1), p0, p0
        elif step < bounds[1]:
            stage, a, b, u, pa, pb = "closure", 0, 1, (step - bounds[0]) / max(bounds[1] - bounds[0] - 1, 1), p0, p0
        elif step < bounds[2]:
            stage, a, b, u, pa, pb = "lift", 1, 2, (step - bounds[1]) / max(bounds[2] - bounds[1] - 1, 1), p0, p1
        else:
            stage, a, b, u, pa, pb = "hold", 2, 3, (step - bounds[2]) / max(n - bounds[2] - 1, 1), p1, p2
        q = (1.0 - u) * q_anchors[a] + u * q_anchors[b]
        pos = (1.0 - u) * pa + u * pb
        quat = unit_quat((1.0 - u) * q0 + u * q2)
        states.append({
            "step": step,
            "stage": stage,
            "joint_positions": q.tolist(),
            "object_position_world_m": pos.tolist(),
            "object_quaternion_world_wxyz": quat,
            "policy_actions": [0.0] * 38,
        })
    out = {
        "schema": "xhand_rl_curriculum_endpoint_interpolation_visualization_v1",
        "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v2",
        "object": x["object"],
        "simulation_dt_s": 0.002,
        "policy_dt_s": 0.008,
        "joint_names": JOINT_NAMES,
        "states": states,
        "source_prefix_state": str(args.prefix_state.resolve()),
        "source_prefix_level": int(x["prefix_level"]),
        "source_episode_length": int(x.get("episode_length", n)),
        "visualization_only": True,
        "accepted_sample": False,
        "exact_per_step_recorded": False,
        "note": "Interpolates recorded curriculum endpoint q/object states for diagnosis; not a formal exact trajectory.",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
