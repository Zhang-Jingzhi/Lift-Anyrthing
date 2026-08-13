#!/usr/bin/env python3
"""Scan forward chassis clearance and palm roll for compact XHand grasps."""
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jaxlie
import numpy as np
import torch

from generate_xhand_fullbody_grasps_v1 import fixed_solver
from make_compact_xhand_palm_normal_candidates import ik_q, load_samples, solve_pose


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "migration_4090/results/xhand_compact_cracker_z_contact_scan_v1.pt"
OUTPUT = ROOT / "migration_4090/results/xhand_compact_cracker_tilted_palm_v1.pt"
SUMMARY = ROOT / "migration_4090/results/xhand_compact_cracker_tilted_palm_v1.json"
OBJECT_X_VALUES = (0.40, 0.50, 0.60)
SOURCE_INDICES = (0, 1)  # wrist z offsets +0.03 and +0.06 m, close side distance
PALM_ROLL_DEG = (-15.0, -25.0, -35.0)
ORIENTATION_WEIGHT = 0.3


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)
    _, robot = fixed_solver()
    ik_names = list(robot.joints.actuated_names)
    links = [robot.links.names.index(name) for name in ("L_tcp", "R_tcp")]
    sources = load_samples(SOURCE)
    canonical = jaxlie.SO3.from_matrix(
        jnp.asarray(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=jnp.float32,
        )
    )

    samples, rows = [], []
    for source_index in SOURCE_INDICES:
        source = sources[source_index]
        for object_x in OBJECT_X_VALUES:
            for roll_deg in PALM_ROLL_DEG:
                sample = copy.deepcopy(source)
                object_pose = np.asarray(sample["object_pose_world"], dtype=np.float32)
                object_pose[0, 3] = object_x
                positions = np.asarray(sample["target_tcp_positions_world"], dtype=np.float32)
                positions[:, 0] = object_x
                sample["object_pose_world"] = object_pose.tolist()
                sample["target_tcp_positions_world"] = positions.tolist()

                # Roll about TCP local X: inward palm normals are unchanged,
                # while local Z tilts toward the chassis (-world X).  This
                # matches the stable wide-grasp finger approach direction.
                rotation = canonical @ jaxlie.SO3.exp(
                    jnp.asarray([np.deg2rad(roll_deg), 0.0, 0.0], dtype=jnp.float32)
                )
                targets = [
                    jaxlie.SE3.from_rotation_and_translation(
                        rotation, jnp.asarray(positions[i])
                    )
                    for i in range(2)
                ]
                initial = ik_q(source, ik_names)
                solved = np.asarray(
                    solve_pose(robot, initial, targets, links, ORIENTATION_WEIGHT)
                )
                fk = robot.forward_kinematics(jnp.asarray(solved))
                achieved, pos_errors, ori_errors, axis_errors = [], [], [], []
                for i, link in enumerate(links):
                    actual = jaxlie.SE3(fk[link])
                    achieved.append(np.asarray(actual.translation()).tolist())
                    pos_errors.append(
                        float(jnp.linalg.norm(actual.translation() - targets[i].translation()))
                    )
                    ori_errors.append(
                        float(
                            jnp.linalg.norm(
                                (targets[i].rotation().inverse() @ actual.rotation()).log()
                            )
                        )
                    )
                    axis = actual.rotation().as_matrix()[:, 0]
                    axis_errors.append(
                        float(
                            jnp.arccos(
                                jnp.clip(
                                    jnp.dot(axis, jnp.asarray([0.0, -1.0, 0.0])),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )

                full_index = {name: i for i, name in enumerate(sample["joint_names"])}
                full_q = np.asarray(sample["full_body_q"], dtype=np.float32)
                for value, name in zip(solved, ik_names):
                    full_q[full_index[name]] = float(value)
                sample["full_body_q"] = full_q.tolist()
                sample["achieved_tcp_positions_world"] = achieved
                sample["ik_position_error_m"] = pos_errors
                sample["compact_tilted_palm"] = {
                    "source_index": source_index,
                    "object_x_m": object_x,
                    "palm_roll_deg": roll_deg,
                    "orientation_weight": ORIENTATION_WEIGHT,
                    "orientation_error_deg": (np.asarray(ori_errors) * 180.0 / np.pi).tolist(),
                    "palm_axis_error_deg": (np.asarray(axis_errors) * 180.0 / np.pi).tolist(),
                }
                samples.append(sample)
                rows.append(
                    {
                        "index": len(samples) - 1,
                        "source_index": source_index,
                        "wrist_z_offset_m": source.get("compact_scan_z_offset_m"),
                        "object_x_m": object_x,
                        "palm_roll_deg": roll_deg,
                        "position_error_mm": (np.asarray(pos_errors) * 1000.0).tolist(),
                        "orientation_error_deg": (np.asarray(ori_errors) * 180.0 / np.pi).tolist(),
                        "palm_axis_error_deg": (np.asarray(axis_errors) * 180.0 / np.pi).tolist(),
                    }
                )

    torch.save(
        {"schema": "xhand_compact_tilted_palm_candidates_v1", "samples": samples},
        OUTPUT,
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "schema": "xhand_compact_tilted_palm_candidates_v1",
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
