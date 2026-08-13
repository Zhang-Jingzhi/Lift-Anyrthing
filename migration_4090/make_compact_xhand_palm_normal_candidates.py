#!/usr/bin/env python3
"""Build compact XHand candidates with opposed palm normals.

The first compact retargeting experiment copied the complete wrist rotations
from a wide grasp.  That over-constrained wrist roll after the object and both
hands were moved forward.  Here both TCP local +X axes point toward world -Y:
for the left hand +X is the inward palm direction, while for the right hand
-X is the inward direction.  The remaining axes use a neutral palm-down roll.
We scan two wrist clearances and three heights from the existing compact IK
solutions and let the pose factor trade a small amount of roll for reachability.
"""
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
COMPACT = ROOT / "migration_4090/results/xhand_compact_cracker_z_contact_scan_v1.pt"
OUTPUT = ROOT / "migration_4090/results/xhand_compact_cracker_palm_normal_v1.pt"
SUMMARY = ROOT / "migration_4090/results/xhand_compact_cracker_palm_normal_v1.json"
TARGET_LINKS = ("L_tcp", "R_tcp")
ORIENTATION_WEIGHTS = (0.05, 0.1, 0.3)


def load_samples(path):
    return torch.load(path, map_location="cpu", weights_only=False)["samples"]


def ik_q(sample, names):
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    return np.asarray([values[name] for name in names], dtype=np.float32)


def solve_pose(robot, initial_q, target_poses, link_indices, ori_weight):
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            target_poses[i],
            jnp.asarray(link_indices[i]),
            pos_weight=10.0,
            ori_weight=float(ori_weight),
        )
        for i in range(2)
    ]
    factors.append(pk.costs.limit_cost(robot, joint_var, weight=10.0))
    problem = jaxls.LeastSquaresProblem(factors, [joint_var]).analyze()
    return problem.solve(
        initial_vals=jaxls.VarValues.make(
            [joint_var.with_value(jnp.asarray(initial_q))]
        ),
        linear_solver="dense_cholesky",
        verbose=False,
        termination=jaxls.TerminationConfig(
            max_iterations=128, early_termination=False
        ),
        trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
    )[joint_var]


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)

    _, robot = fixed_solver()
    ik_names = list(robot.joints.actuated_names)
    link_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    compact_samples = load_samples(COMPACT)

    # Columns are the TCP local axes in world coordinates.  Both TCP +X axes
    # point along world -Y.  Local Z points down, leaving local Y along -X.
    canonical_rotation = jaxlie.SO3.from_matrix(
        jnp.asarray(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=jnp.float32,
        )
    )

    samples = []
    rows = []
    for compact_index, compact in enumerate(compact_samples):
        initial_q = ik_q(compact, ik_names)
        positions = np.asarray(compact["target_tcp_positions_world"], dtype=np.float32)
        target_poses = [
            jaxlie.SE3.from_rotation_and_translation(
                canonical_rotation, jnp.asarray(positions[i])
            )
            for i in range(2)
        ]
        full_index = {name: i for i, name in enumerate(compact["joint_names"])}

        for weight in ORIENTATION_WEIGHTS:
            solved = np.asarray(
                solve_pose(robot, initial_q, target_poses, link_indices, weight)
            )
            fk = robot.forward_kinematics(jnp.asarray(solved))
            achieved = []
            pos_errors = []
            ori_errors = []
            palm_axis_errors = []
            for i, link_index in enumerate(link_indices):
                actual = jaxlie.SE3(fk[link_index])
                achieved.append(np.asarray(actual.translation()).tolist())
                pos_errors.append(
                    float(
                        jnp.linalg.norm(
                            actual.translation() - target_poses[i].translation()
                        )
                    )
                )
                ori_errors.append(
                    float(
                        jnp.linalg.norm(
                            (target_poses[i].rotation().inverse() @ actual.rotation()).log()
                        )
                    )
                )
                actual_x = actual.rotation().as_matrix()[:, 0]
                palm_axis_errors.append(
                    float(jnp.arccos(jnp.clip(jnp.dot(actual_x, jnp.array([0.0, -1.0, 0.0])), -1.0, 1.0)))
                )

            sample = copy.deepcopy(compact)
            full_q = np.asarray(sample["full_body_q"], dtype=np.float32)
            for value, name in zip(solved, ik_names):
                full_q[full_index[name]] = float(value)
            sample["full_body_q"] = full_q.tolist()
            sample["achieved_tcp_positions_world"] = achieved
            sample["ik_position_error_m"] = pos_errors
            sample["palm_normal_target"] = {
                "tcp_local_axis": "+X",
                "world_direction": [0.0, -1.0, 0.0],
                "orientation_weight": weight,
                "orientation_error_deg": (
                    np.asarray(ori_errors) * 180.0 / np.pi
                ).tolist(),
                "palm_axis_error_deg": (
                    np.asarray(palm_axis_errors) * 180.0 / np.pi
                ).tolist(),
            }
            sample["compact_source_index"] = compact_index
            samples.append(sample)
            rows.append(
                {
                    "index": len(samples) - 1,
                    "compact_source_index": compact_index,
                    "wrist_z_offset_m": compact.get("compact_scan_z_offset_m"),
                    "wrist_side_m": float(abs(positions[0, 1])),
                    "orientation_weight": weight,
                    "position_error_mm": (
                        np.asarray(pos_errors) * 1000.0
                    ).tolist(),
                    "orientation_error_deg": (
                        np.asarray(ori_errors) * 180.0 / np.pi
                    ).tolist(),
                    "palm_axis_error_deg": (
                        np.asarray(palm_axis_errors) * 180.0 / np.pi
                    ).tolist(),
                }
            )

    torch.save(
        {"schema": "xhand_compact_palm_normal_candidates_v1", "samples": samples},
        OUTPUT,
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "schema": "xhand_compact_palm_normal_candidates_v1",
                "output": str(OUTPUT),
                "candidates": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
