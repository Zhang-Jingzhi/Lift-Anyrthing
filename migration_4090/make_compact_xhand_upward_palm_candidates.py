#!/usr/bin/env python3
"""Generate compact grasps whose opposed palm normals also support upward."""
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
SOURCE = Path(
    os.environ.get(
        "XHAND_UPWARD_SOURCE",
        ROOT / "migration_4090/results/xhand_compact_cracker_z_contact_scan_v1.pt",
    )
)
OUTPUT_TAG = os.environ.get("XHAND_UPWARD_OUTPUT_TAG", "cracker_v1")
OUTPUT = ROOT / f"migration_4090/results/xhand_compact_{OUTPUT_TAG}_upward_palm.pt"
SUMMARY = ROOT / f"migration_4090/results/xhand_compact_{OUTPUT_TAG}_upward_palm.json"
SOURCE_INDICES = tuple(
    int(value)
    for value in os.environ.get("XHAND_UPWARD_SOURCE_INDICES", "0,1").split(",")
    if value.strip()
)
OBJECT_X = float(os.environ.get("XHAND_UPWARD_OBJECT_X", "0.4"))
UPWARD_TILT_DEG = (5.0, 10.0, 15.0, 20.0)
BACK_ROLL_DEG = (-15.0, -25.0)
ORIENTATION_WEIGHT = 0.3
PREGRASP_Y_OFFSET = float(os.environ.get("XHAND_UPWARD_PREGRASP_Y_OFFSET", "0.0"))
FINAL_Y_OFFSET = float(os.environ.get("XHAND_UPWARD_FINAL_Y_OFFSET", "0.0"))
WRIST_Z_OFFSET = float(os.environ.get("XHAND_UPWARD_WRIST_Z_OFFSET", "0.0"))


def target_rotation(side, upward_deg, back_roll_deg):
    """Return TCP axes with inward palm normal carrying an upward component."""
    angle = np.deg2rad(upward_deg)
    # Left inward normal is local +X. Right inward normal is local -X.
    z_sign = 1.0 if side == "left" else -1.0
    x_axis = np.asarray([0.0, -np.cos(angle), z_sign * np.sin(angle)])
    roll = abs(np.deg2rad(back_roll_deg))
    z_hint = np.asarray([-np.sin(roll), 0.0, -np.cos(roll)])
    z_axis = z_hint - np.dot(z_hint, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    matrix = np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)
    return jaxlie.SO3.from_matrix(jnp.asarray(matrix))


def set_hand_posture(sample):
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    for side in ("left", "right"):
        values[f"{side}_hand_index_bend_joint"] = 0.10
        for finger in ("index", "mid", "ring", "pinky"):
            values[f"{side}_hand_{finger}_joint1"] = 0.90
            values[f"{side}_hand_{finger}_joint2"] = 1.10
        values[f"{side}_hand_thumb_bend_joint"] = 1.00
        values[f"{side}_hand_thumb_rota_joint1"] = -0.30
        values[f"{side}_hand_thumb_rota_joint2"] = 1.20
    sample["full_body_q"] = [float(values[name]) for name in sample["joint_names"]]


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)
    _, robot = fixed_solver()
    ik_names = list(robot.joints.actuated_names)
    links = [robot.links.names.index(name) for name in ("L_tcp", "R_tcp")]
    sources = load_samples(SOURCE)
    samples, rows = [], []
    for source_index in SOURCE_INDICES:
        source = sources[source_index]
        positions = np.asarray(source["target_tcp_positions_world"], dtype=np.float32).copy()
        positions[:, 0] = OBJECT_X
        positions[0, 1] += FINAL_Y_OFFSET
        positions[1, 1] -= FINAL_Y_OFFSET
        positions[:, 2] += WRIST_Z_OFFSET
        for upward in UPWARD_TILT_DEG:
            for back_roll in BACK_ROLL_DEG:
                rotations = (
                    target_rotation("left", upward, back_roll),
                    target_rotation("right", upward, back_roll),
                )
                targets = [
                    jaxlie.SE3.from_rotation_and_translation(
                        rotations[i], jnp.asarray(positions[i])
                    )
                    for i in range(2)
                ]
                solved = np.asarray(
                    solve_pose(robot, ik_q(source, ik_names), targets, links, ORIENTATION_WEIGHT)
                )
                fk = robot.forward_kinematics(jnp.asarray(solved))
                achieved, pos_errors, ori_errors, inward_up = [], [], [], []
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
                    local_x = np.asarray(actual.rotation().as_matrix()[:, 0])
                    inward = local_x if i == 0 else -local_x
                    inward_up.append(float(inward[2]))

                sample = copy.deepcopy(source)
                object_pose = np.asarray(sample["object_pose_world"], dtype=np.float32)
                object_pose[0, 3] = OBJECT_X
                sample["object_pose_world"] = object_pose.tolist()
                sample["target_tcp_positions_world"] = positions.tolist()
                full_index = {name: i for i, name in enumerate(sample["joint_names"])}
                full_q = np.asarray(sample["full_body_q"], dtype=np.float32)
                for value, name in zip(solved, ik_names):
                    full_q[full_index[name]] = float(value)
                sample["full_body_q"] = full_q.tolist()
                sample["achieved_tcp_positions_world"] = achieved
                sample["ik_position_error_m"] = pos_errors
                set_hand_posture(sample)
                if PREGRASP_Y_OFFSET > 0.0:
                    # Approach from outside the object instead of spawning the
                    # full-size XHands directly at the final contact pose.
                    # This is essential for a rounded object: straight/open
                    # fingers at the final wrist pose can initially overlap
                    # the VHACD hull and receive an explosive correction.
                    pre_positions = positions.copy()
                    pre_positions[0, 1] += PREGRASP_Y_OFFSET
                    pre_positions[1, 1] -= PREGRASP_Y_OFFSET
                    pre_targets = [
                        jaxlie.SE3.from_rotation_and_translation(
                            rotations[i], jnp.asarray(pre_positions[i])
                        )
                        for i in range(2)
                    ]
                    pre_solved = np.asarray(
                        solve_pose(robot, solved, pre_targets, links, ORIENTATION_WEIGHT)
                    )
                    pre_fk = robot.forward_kinematics(jnp.asarray(pre_solved))
                    pre_full_q = np.asarray(sample["full_body_q"], dtype=np.float32)
                    for value, name in zip(pre_solved, ik_names):
                        pre_full_q[full_index[name]] = float(value)
                    sample["pregrasp_full_body_q"] = pre_full_q.tolist()
                    sample["pregrasp_target_tcp_positions_world"] = pre_positions.tolist()
                    sample["pregrasp_achieved_tcp_positions_world"] = [
                        np.asarray(jaxlie.SE3(pre_fk[link]).translation()).tolist()
                        for link in links
                    ]
                sample["compact_upward_palm"] = {
                    "source_index": source_index,
                    "object_x_m": OBJECT_X,
                    "wrist_z_offset_m": source.get("compact_scan_z_offset_m"),
                    "upward_tilt_deg": upward,
                    "back_roll_deg": back_roll,
                    "orientation_error_deg": (np.asarray(ori_errors) * 180.0 / np.pi).tolist(),
                    "achieved_inward_normal_z": inward_up,
                    "pregrasp_y_offset_m": PREGRASP_Y_OFFSET,
                    "final_y_offset_m": FINAL_Y_OFFSET,
                    "wrist_target_z_offset_m": WRIST_Z_OFFSET,
                }
                samples.append(sample)
                rows.append(
                    {
                        "index": len(samples) - 1,
                        **sample["compact_upward_palm"],
                        "position_error_mm": (np.asarray(pos_errors) * 1000.0).tolist(),
                    }
                )
    torch.save(
        {"schema": "xhand_compact_upward_palm_candidates_v1", "samples": samples},
        OUTPUT,
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "schema": "xhand_compact_upward_palm_candidates_v1",
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
