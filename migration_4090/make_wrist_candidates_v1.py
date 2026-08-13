#!/usr/bin/env python3
"""Create local wrist-perturbation candidates from one valid XHand IK pose."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import jax.numpy as jnp
import jaxlie

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pyroki_ik import PyrokiRetarget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ik-urdf", type=Path, required=True)
    ap.add_argument("--left-y", nargs="+", type=float, default=None)
    ap.add_argument("--right-y", nargs="+", type=float, required=True)
    ap.add_argument("--x-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--z-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--left-j7-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--right-j7-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--left-j6-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--right-j6-offset", nargs="+", type=float, default=[0.0])
    ap.add_argument("--random-init-seeds", nargs="*", type=int, default=[])
    ap.add_argument("--random-init-scale", type=float, default=0.35)
    args = ap.parse_args()

    payload = torch.load(args.dataset, map_location="cpu")
    base = payload["samples"][0]
    solver = PyrokiRetarget(str(args.ik_urdf), ["L_tcp", "R_tcp"])
    robot = solver.robot
    ik_names = list(robot.joints.actuated_names)
    full_names = list(base["joint_names"])
    q_map = dict(zip(full_names, base["full_body_q"]))
    q0 = np.asarray([q_map[n] for n in ik_names], dtype=np.float32)
    achieved = np.asarray(base["achieved_tcp_positions_world"], dtype=np.float32)
    left_ys = [float(achieved[0, 1])] if args.left_y is None else [float(y) for y in args.left_y]
    ys = [float(y) for y in args.right_y]
    xs = [float(x) for x in args.x_offset]
    zs = [float(z) for z in args.z_offset]
    left_j7_offsets = [float(v) for v in args.left_j7_offset]
    right_j7_offsets = [float(v) for v in args.right_j7_offset]
    left_j6_offsets = [float(v) for v in args.left_j6_offset]
    right_j6_offsets = [float(v) for v in args.right_j6_offset]
    name_to_ik = {n: i for i, n in enumerate(ik_names)}
    targets = []
    labels = []
    init_list = []
    for x in xs:
        for z in zs:
            for left_y in left_ys:
                for right_y in ys:
                    for left_j7 in left_j7_offsets:
                        for right_j7 in right_j7_offsets:
                            for left_j6 in left_j6_offsets:
                                for right_j6 in right_j6_offsets:
                                    target = achieved.copy()
                                    target[0] = [float(achieved[0, 0] + x), left_y, float(achieved[0, 2] + z)]
                                    target[1] = [float(achieved[1, 0] + x), right_y, float(achieved[1, 2] + z)]
                                    targets.append(target)
                                    q_init = q0.copy()
                                    if args.random_init_seeds:
                                        # Explore distinct arm/elbow IK basins while
                                        # retaining the hand posture and fixed joints.
                                        rng = np.random.default_rng(int(args.random_init_seeds[len(init_list) % len(args.random_init_seeds)]))
                                        q_init = q_init + rng.normal(0.0, float(args.random_init_scale), size=q_init.shape).astype(np.float32)
                                    if "left_j7" in name_to_ik:
                                        q_init[name_to_ik["left_j7"]] += left_j7
                                    if "right_j7" in name_to_ik:
                                        q_init[name_to_ik["right_j7"]] += right_j7
                                    if "left_j6" in name_to_ik:
                                        q_init[name_to_ik["left_j6"]] += left_j6
                                    if "right_j6" in name_to_ik:
                                        q_init[name_to_ik["right_j6"]] += right_j6
                                    init_list.append(q_init)
                                    labels.append({"left_y": left_y, "right_y": right_y, "x_offset": x, "z_offset": z, "left_j7_offset": left_j7, "right_j7_offset": right_j7, "left_j6_offset": left_j6, "right_j6_offset": right_j6})
    init = np.asarray(init_list, dtype=np.float32)
    solved = solver.solve_retarget(jnp.asarray(init), jnp.asarray(np.asarray(targets, dtype=np.float32)))
    solved = np.asarray(solved)
    poses = robot.forward_kinematics(jnp.asarray(solved))
    link_l = robot.links.names.index("L_tcp")
    link_r = robot.links.names.index("R_tcp")
    achieved_out = np.asarray(jaxlie.SE3(poses[:, [link_l, link_r], :]).translation())
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    out = []
    for i, q in enumerate(solved):
        err = np.linalg.norm(achieved_out[i] - np.asarray(targets[i]), axis=1)
        finite = bool(np.isfinite(q).all() and np.isfinite(err).all())
        limits = bool(np.all(q >= lower - 1e-4) and np.all(q <= upper + 1e-4))
        if not (finite and limits):
            continue
        sample = copy.deepcopy(base)
        full_q = np.zeros(len(full_names), dtype=np.float32)
        full_index = {n: j for j, n in enumerate(full_names)}
        for value, name in zip(q, ik_names):
            full_q[full_index[name]] = value
        sample["full_body_q"] = full_q.tolist()
        sample["target_tcp_positions_world"] = np.asarray(targets[i]).tolist()
        sample["achieved_tcp_positions_world"] = achieved_out[i].tolist()
        sample["ik_position_error_m"] = err.tolist()
        sample["optimization_candidate"] = labels[i]
        sample["geometry_pass"] = False
        sample["isaaclab_physical_validated"] = False
        out.append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "xhand_fullbody_grasp_pose_wrist_search_v1", "samples": out}, args.output)
    print({"output": str(args.output), "requested": len(targets), "accepted": len(out), "labels": labels})
    for i, sample in enumerate(out):
        print(i, sample["optimization_candidate"], sample["ik_position_error_m"])


if __name__ == "__main__":
    main()
