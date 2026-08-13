#!/usr/bin/env python3
"""Generate kinematic dual-XHand poses for scanned ObjectFlow assets."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
import jax.numpy as jnp
import jaxlie
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "external_assets/objectflow_20260812/ObjectFlow_3D_sim_assets_20260812"
sys.path.insert(0, str(ROOT))
from utils.pyroki_ik import PyrokiRetarget

IK_URDF = Path(os.environ.get("XHAND_FULLBODY_IK_URDF", "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"))


def load_pose(path, index):
    rows = list(csv.DictReader(path.open()))
    row = rows[max(0, min(index, len(rows) - 1))]
    matrix = np.asarray([[float(row[f"m{i}{j}"]) for j in range(4)] for i in range(4)], dtype=np.float32)
    return matrix, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--object-x", type=float, default=0.50)
    ap.add_argument("--object-y", type=float, default=0.0)
    ap.add_argument("--object-z", type=float, default=0.86,
                    help="height of the scanned object center in robot preview frame")
    ap.add_argument("--wrist-side", type=float, default=0.045,
                    help="clearance added outside the object half-width")
    ap.add_argument("--tcp-inset", type=float, default=0.0,
                    help="move each TCP inward from the nominal side surface")
    ap.add_argument("--wrist-z-offset", type=float, default=0.0,
                    help="additional TCP height relative to object center")
    ap.add_argument("--anchor-object", action="store_true", default=True,
                    help="anchor each transformed scan at the robot workspace center")
    ap.add_argument("--solve-ik", action="store_true")
    ap.add_argument("--hand-posture-template", type=Path, default=None,
                    help="optional .pt sample bank whose hand joint values are copied after TCP IK")
    ap.add_argument("--ik-initial-template", type=Path, default=None,
                    help="optional pose bank used as the IK warm start to preserve a validated wrist branch")
    ap.add_argument("--template-index", type=int, default=0)
    args = ap.parse_args()
    ep = ASSET_ROOT / args.episode
    mesh_path = ep / "model.obj"
    meta = json.loads((ep / "metadata.json").read_text())
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pose_path = ep / "pose_objectflow_interval.csv"
    pose_rows = list(csv.DictReader(pose_path.open()))
    indices = np.linspace(0, len(pose_rows) - 1, max(1, args.frames), dtype=int)
    solver = PyrokiRetarget(str(IK_URDF), ["L_tcp", "R_tcp"]) if args.solve_ik else None
    initial_template = None
    if solver is not None and args.ik_initial_template is not None:
        payload = torch.load(args.ik_initial_template, map_location="cpu", weights_only=False)
        source = payload["samples"][args.template_index]
        initial_template = {n: float(v) for n, v in zip(source["joint_names"], source["full_body_q"])}
    samples = []
    for idx in indices:
        pose, row = load_pose(pose_path, int(idx))
        # Convert OpenCV camera coordinates (+Y down, +Z forward) to a simple
        # robot preview frame (+Z up) while preserving the scanned object pose.
        cv_to_robot = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        rot = cv_to_robot @ pose[:3, :3]
        trans = cv_to_robot @ pose[:3, 3]
        local = mesh.copy()
        source_T = np.block([[rot, trans[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        local.apply_transform(source_T)
        if args.anchor_object:
            # ObjectFlow poses are camera-relative.  They do not define the
            # Tianji base/world origin, so align the scan centroid to a safe
            # tabletop workspace while retaining the observed orientation.
            scan_center = local.bounding_box.centroid
            anchor = np.array([args.object_x, args.object_y, args.object_z], dtype=np.float32)
            delta = anchor - scan_center
            local.apply_translation(delta)
            trans = trans + delta
        bounds = local.bounds
        center = bounds.mean(0)
        half_y = max(0.05, float(local.extents[1] * 0.5))
        target_z = float(center[2] + args.wrist_z_offset)
        side = max(0.02, half_y + args.wrist_side - args.tcp_inset)
        targets = [[float(center[0]), float(center[1] + side), target_z],
                   [float(center[0]), float(center[1] - side), target_z]]
        sample = {
            "schema": "objectflow_scanned_xhand_kinematic_pose_v1",
            "episode": args.episode,
            "frame_index": int(idx),
            "source_frame": int(row["source_frame"]),
            "object_mesh_path": str(mesh_path.resolve()),
            "object_pose_source_opencv": pose.tolist(),
            "object_pose_robot_preview": np.block([[rot, trans[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
            "object_extents_m": meta["extents_m"],
            "target_tcp_positions_world": targets,
            "ik_pending": not args.solve_ik,
            "physical_validation_pending": True,
            "source_quality": meta,
        }
        if solver is not None:
            q0 = np.zeros((solver.robot.joints.num_actuated_joints,), dtype=np.float32)
            if initial_template is not None:
                for j, name in enumerate(solver.robot.joints.actuated_names):
                    if name in initial_template:
                        q0[j] = initial_template[name]
            sample["_ik_initial_q"] = q0.tolist()
        samples.append(sample)
    if solver is not None:
        initial = jnp.asarray([s.pop("_ik_initial_q") for s in samples])
        target_batch = jnp.asarray([s["target_tcp_positions_world"] for s in samples])
        solved = np.asarray(solver.solve_retarget(initial, target_batch))
        links = [solver.robot.links.names.index(n) for n in ("L_tcp", "R_tcp")]
        posture = None
        if args.hand_posture_template is not None:
            tpl = torch.load(args.hand_posture_template, map_location="cpu", weights_only=False)
            ts = tpl["samples"][args.template_index]
            posture = {n: float(v) for n, v in zip(ts["joint_names"], ts["full_body_q"])}
        for sample, q in zip(samples, solved):
            q = q.copy()
            fk = solver.robot.forward_kinematics(jnp.asarray(q))
            achieved = [np.asarray(jaxlie.SE3(fk[i]).translation()).tolist() for i in links]
            if posture is not None:
                for j, name in enumerate(solver.robot.joints.actuated_names):
                    if "_hand_" in name and name in posture:
                        q[j] = posture[name]
            sample.update({"ik_pending": False, "full_body_ik_q": q.tolist(), "achieved_tcp_positions_world": achieved, "ik_position_error_m": [float(np.linalg.norm(np.asarray(achieved[i]) - np.asarray(sample["target_tcp_positions_world"][i]))) for i in range(2)]})
            if args.hand_posture_template is not None:
                sample["hand_posture_source"] = str(args.hand_posture_template.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "objectflow_scanned_xhand_pose_bank_v1", "samples": samples}, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "episode": args.episode, "samples": len(samples), "watertight": meta["watertight"]}, indent=2))


if __name__ == "__main__":
    main()
