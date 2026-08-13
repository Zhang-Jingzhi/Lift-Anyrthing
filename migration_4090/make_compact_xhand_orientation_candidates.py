#!/usr/bin/env python3
"""Retarget compact XHand poses while preserving verified palm orientations."""
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import torch

from generate_xhand_fullbody_grasps_v1 import fixed_solver


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "migration_4090/results/xhand_external_candidates_v1/cracker/baseline__ycb__cracker_box_formal_large_random_v1_002.pt"
COMPACT = ROOT / "migration_4090/results/xhand_compact_cracker_z_contact_scan_v1.pt"
OUTPUT = ROOT / "migration_4090/results/xhand_compact_cracker_orientation_v1.pt"
SUMMARY = ROOT / "migration_4090/results/xhand_compact_cracker_orientation_v1.json"
REFERENCE_INDEX = 14
COMPACT_INDEX = 5
TARGET_LINKS = ("L_tcp", "R_tcp")
ORIENTATION_WEIGHTS = (0.05, 0.1, 0.3, 1.0)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)["samples"]


def ik_q(sample, names):
    q = dict(zip(sample["joint_names"], sample["full_body_q"]))
    return np.asarray([q[name] for name in names], dtype=np.float32)


def solve_pose(robot, initial_q, target_poses, link_indices, ori_weight):
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_poses[i], jnp.asarray(link_indices[i]),
            pos_weight=10.0, ori_weight=float(ori_weight),
        )
        for i in range(2)
    ]
    factors.append(pk.costs.limit_cost(robot, joint_var, weight=10.0))
    problem = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze()
    return problem.solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(jnp.asarray(initial_q))]),
        linear_solver="dense_cholesky",
        verbose=False,
        termination=jaxls.TerminationConfig(max_iterations=128, early_termination=False),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
    )[joint_var]


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)
    solver, robot = fixed_solver()
    ik_names = list(robot.joints.actuated_names)
    link_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    reference = load(REFERENCE)[REFERENCE_INDEX]
    compact = load(COMPACT)[COMPACT_INDEX]
    q_reference = ik_q(reference, ik_names)
    q_initial = ik_q(compact, ik_names)
    reference_fk = robot.forward_kinematics(jnp.asarray(q_reference))
    reference_poses = [jaxlie.SE3(reference_fk[index]) for index in link_indices]
    positions = np.asarray(compact["target_tcp_positions_world"], dtype=np.float32)
    target_poses = [
        jaxlie.SE3.from_rotation_and_translation(reference_poses[i].rotation(), jnp.asarray(positions[i]))
        for i in range(2)
    ]
    full_index = {name: i for i, name in enumerate(compact["joint_names"])}
    samples, rows = [], []
    for weight in ORIENTATION_WEIGHTS:
        solved = np.asarray(solve_pose(robot, q_initial, target_poses, link_indices, weight))
        fk = robot.forward_kinematics(jnp.asarray(solved))
        achieved, pos_errors, ori_errors = [], [], []
        for i, link_index in enumerate(link_indices):
            actual = jaxlie.SE3(fk[link_index])
            achieved.append(np.asarray(actual.translation()).tolist())
            pos_errors.append(float(jnp.linalg.norm(actual.translation() - target_poses[i].translation())))
            ori_errors.append(float(jnp.linalg.norm((target_poses[i].rotation().inverse() @ actual.rotation()).log())))
        sample = copy.deepcopy(compact)
        full_q = np.asarray(sample["full_body_q"], dtype=np.float32)
        for value, name in zip(solved, ik_names):
            full_q[full_index[name]] = float(value)
        sample["full_body_q"] = full_q.tolist()
        sample["achieved_tcp_positions_world"] = achieved
        sample["ik_position_error_m"] = pos_errors
        sample["orientation_reference"] = {
            "dataset": str(REFERENCE),
            "sample_index": REFERENCE_INDEX,
            "orientation_weight": weight,
            "orientation_error_deg": (np.asarray(ori_errors) * 180.0 / np.pi).tolist(),
        }
        samples.append(sample)
        rows.append({
            "index": len(samples) - 1,
            "orientation_weight": weight,
            "position_error_mm": (np.asarray(pos_errors) * 1000.0).tolist(),
            "orientation_error_deg": (np.asarray(ori_errors) * 180.0 / np.pi).tolist(),
        })
    torch.save({"schema": "xhand_compact_orientation_candidates_v1", "samples": samples}, OUTPUT)
    SUMMARY.write_text(json.dumps({"schema": "xhand_compact_orientation_candidates_v1", "output": str(OUTPUT), "candidates": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
