#!/usr/bin/env python3
"""Generate Tianji + dual-XHand candidate banks for the compact six objects."""

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import jax
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import torch
import trimesh

from generate_xhand_fullbody_grasps_v1 import fixed_solver
from make_compact_xhand_palm_normal_candidates import ik_q
from make_compact_xhand_upward_palm_candidates import set_hand_posture, target_rotation


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "migration_4090/results/xhand_compact_6x8_v2/catalog.json"
TEMPLATE = ROOT / "migration_4090/results/xhand_compact_sphere_290mm_x0p50_v1_upward_palm.pt"
TEMPLATE_SAMPLE_INDEX = 7
TARGET_LINKS = ("L_tcp", "R_tcp")

# Measured with the branch-preserving IK configuration against the Isaac Gym
# asset.  Add the inverse residual to Pyroki targets so the real Isaac TCPs
# land at the desired opposed grasp locations.  Rows are left/right xyz.
ISAAC_BRANCH_TARGET_COMPENSATION_M = np.asarray(
    [
        [-0.0045, -0.0290, 0.0210],
        [-0.0035, 0.0220, 0.0210],
    ],
    dtype=np.float32,
)


def make_reusable_pose_solver(robot, link_indices, ori_weight=0.0):
    """JIT one pose problem while keeping Isaac-verified wrist branches.

    The external Tianji IK URDF and the Isaac asset agree closely on TCP
    translation, but their right-wrist orientation convention differs by a
    branch.  An unconstrained pose solve can therefore jump right_j5 by about
    pi radians: Pyroki reports a good target orientation while the real XHand
    points away from the object in Isaac.  Keep both j5-j7 triplets close to
    the verified sphere configuration and let shoulder/elbow joints provide
    the compact positional adjustment.
    """
    joint_var = robot.joint_var_cls(0)
    rest_weights = np.full(robot.joints.num_actuated_joints, 0.02, dtype=np.float32)
    name_to_index = {
        name: index for index, name in enumerate(robot.joints.actuated_names)
    }
    for side in ("left", "right"):
        for joint in ("j5", "j6", "j7"):
            rest_weights[name_to_index[f"{side}_{joint}"]] = 2.0
    for name, index in name_to_index.items():
        if "_hand_" in name:
            rest_weights[index] = 5.0
    rest_weights = jnp.asarray(rest_weights)

    @jax.jit
    def run(initial_q, target_parameters):
        target_poses = [jaxlie.SE3(target_parameters[i]) for i in range(2)]
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
        factors.append(
            pk.costs.rest_cost(
                joint_var,
                rest_pose=initial_q,
                weight=rest_weights,
            )
        )
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

    return run


def solve_targets(solver, initial_q, targets):
    parameters = jnp.stack([target.wxyz_xyz for target in targets], axis=0)
    return np.asarray(solver(jnp.asarray(initial_q), parameters))


def load_row(key: str, size_index: int):
    payload = json.loads(CATALOG.read_text())
    return next(
        row
        for row in payload["objects"]
        if row["key"] == key and int(row["size_index"]) == size_index
    )


def expand_q(sample, solved, ik_names):
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    values.update({name: float(value) for name, value in zip(ik_names, solved)})
    sample["full_body_q"] = [float(values[name]) for name in sample["joint_names"]]


def achieved_metrics(robot, q, targets, link_indices):
    fk = robot.forward_kinematics(jnp.asarray(q))
    achieved, pos_error, ori_error = [], [], []
    for target, link in zip(targets, link_indices):
        actual = jaxlie.SE3(fk[link])
        achieved.append(np.asarray(actual.translation()).tolist())
        pos_error.append(float(jnp.linalg.norm(actual.translation() - target.translation())))
        ori_error.append(
            float(
                jnp.linalg.norm(
                    (target.rotation().inverse() @ actual.rotation()).log()
                )
            )
        )
    return achieved, pos_error, ori_error


def compensate_targets_for_isaac(desired_positions):
    target_positions = (
        np.asarray(desired_positions, dtype=np.float32)
        + ISAAC_BRANCH_TARGET_COMPENSATION_M
    )
    return target_positions


def sampled_targets(method, rng, center, half_y):
    clearance_extra = float(os.environ.get("XHAND_FORMAL_CLEARANCE_EXTRA_M", "0.0"))
    wrist_z_offset = float(os.environ.get("XHAND_FORMAL_WRIST_Z_OFFSET_M", "0.0"))
    safe_z = max(float(center[2] - 0.020 + wrist_z_offset), 0.815)
    if method == "baseline":
        shared_clearance = float(rng.uniform(0.058, 0.066)) + clearance_extra
        clearance_delta = float(rng.uniform(-0.003, 0.003))
        clearances = np.asarray(
            [shared_clearance + clearance_delta, shared_clearance - clearance_delta]
        )
        shared_x = float(center[0] + rng.uniform(-0.004, 0.004))
        shared_z = float(max(safe_z + rng.uniform(-0.004, 0.005), 0.815))
        x = np.asarray([shared_x, shared_x])
        z = np.asarray([shared_z, shared_z])
        upward = rng.uniform(12.0, 24.0, size=2)
        back_roll = rng.uniform(-30.0, -18.0, size=2)
    else:
        shared_clearance = float(rng.uniform(0.058, 0.064)) + clearance_extra
        clearance_delta = float(rng.uniform(-0.002, 0.002))
        clearances = np.asarray(
            [shared_clearance + clearance_delta, shared_clearance - clearance_delta]
        )
        shared_x = float(center[0] + rng.uniform(-0.004, 0.004))
        shared_z = float(max(safe_z + rng.uniform(-0.003, 0.004), 0.815))
        x = np.asarray([shared_x, shared_x])
        z = np.asarray([shared_z, shared_z])
        shared_up = float(rng.uniform(18.0, 22.0))
        shared_roll = float(rng.uniform(-27.0, -23.0))
        upward = np.asarray([shared_up, shared_up])
        back_roll = np.asarray([shared_roll, shared_roll])
    desired_positions = np.asarray(
        [
            [x[0], half_y + clearances[0], z[0]],
            [x[1], -half_y - clearances[1], z[1]],
        ],
        dtype=np.float32,
    )
    positions = compensate_targets_for_isaac(desired_positions)
    rotations = (
        target_rotation("left", float(upward[0]), float(back_roll[0])),
        target_rotation("right", float(upward[1]), float(back_roll[1])),
    )
    targets = [
        jaxlie.SE3.from_rotation_and_translation(rotations[i], jnp.asarray(positions[i]))
        for i in range(2)
    ]
    return positions, targets, {
        "desired_isaac_tcp_positions_world": desired_positions.tolist(),
        "pyroki_target_compensation_m": ISAAC_BRANCH_TARGET_COMPENSATION_M.tolist(),
        "clearance_m": clearances.tolist(),
        "upward_tilt_deg": upward.tolist(),
        "back_roll_deg": back_roll.tolist(),
    }


def canonical_anchor_targets(center, half_y, compensate_for_isaac=False):
    """The verified sphere grasp expressed relative to any object's bounds."""
    wrist_z_offset = float(os.environ.get("XHAND_FORMAL_WRIST_Z_OFFSET_M", "0.0"))
    desired_positions = np.asarray(
        [
            [center[0], half_y + 0.060, max(float(center[2] - 0.020 + wrist_z_offset), 0.815)],
            [center[0], -half_y - 0.060, max(float(center[2] - 0.020 + wrist_z_offset), 0.815)],
        ],
        dtype=np.float32,
    )
    positions = (
        compensate_targets_for_isaac(desired_positions)
        if compensate_for_isaac
        else desired_positions
    )
    rotations = (
        target_rotation("left", 20.0, -25.0),
        target_rotation("right", 20.0, -25.0),
    )
    targets = [
        jaxlie.SE3.from_rotation_and_translation(rotations[i], jnp.asarray(positions[i]))
        for i in range(2)
    ]
    meta = {
        "desired_isaac_tcp_positions_world": desired_positions.tolist(),
        "clearance_m": [0.060, 0.060],
        "upward_tilt_deg": [20.0, 20.0],
        "back_roll_deg": [-25.0, -25.0],
        "canonical_anchor": True,
    }
    if compensate_for_isaac:
        meta["pyroki_target_compensation_m"] = (
            ISAAC_BRANCH_TARGET_COMPENSATION_M.tolist()
        )
    return positions, targets, meta


def perturb_hand_posture(sample, method, rng):
    set_hand_posture(sample)
    hand_delta_limit = float(os.environ.get("XHAND_FORMAL_HAND_DELTA_LIMIT", "0.035"))
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    if method == "baseline":
        for name in values:
            if "_hand_" in name:
                values[name] += float(rng.uniform(-hand_delta_limit, hand_delta_limit))
    else:
        for suffix in (
            "index_bend_joint",
            "index_joint1", "index_joint2",
            "mid_joint1", "mid_joint2",
            "ring_joint1", "ring_joint2",
            "pinky_joint1", "pinky_joint2",
            "thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2",
        ):
            delta = float(rng.uniform(-hand_delta_limit, hand_delta_limit))
            for side in ("left", "right"):
                values[f"{side}_hand_{suffix}"] += delta
    sample["full_body_q"] = [float(values[name]) for name in sample["joint_names"]]


def solve_candidate(robot, pose_solver, initial_q, link_indices, method, rng, center, half_y, anchor=False, reuse_initial_anchor=False):
    if anchor:
        positions, targets, meta = canonical_anchor_targets(
            center,
            half_y,
            compensate_for_isaac=not reuse_initial_anchor,
        )
        solved = initial_q.copy() if reuse_initial_anchor else solve_targets(pose_solver, initial_q, targets)
        achieved, pos_error, ori_error = achieved_metrics(
            robot, solved, targets, link_indices
        )
        lower = np.asarray(robot.joints.lower_limits)
        upper = np.asarray(robot.joints.upper_limits)
        if reuse_initial_anchor and np.isfinite(solved).all():
            score = max(pos_error) * 20.0 + max(ori_error) * 0.08
            meta["pyroki_gate_bypassed_for_isaac_verified_anchor"] = True
            return score, solved, positions, targets, achieved, pos_error, ori_error, meta
        if (
            np.isfinite(solved).all()
            and np.all(solved >= lower - 1e-4)
            and np.all(solved <= upper + 1e-4)
            and max(pos_error) <= 0.004
        ):
            score = max(pos_error) * 20.0 + max(ori_error) * 0.08
            return score, solved, positions, targets, achieved, pos_error, ori_error, meta
    proposal_count = 1 if method == "baseline" else 6
    best = None
    for _ in range(proposal_count):
        positions, targets, meta = sampled_targets(method, rng, center, half_y)
        solved = solve_targets(pose_solver, initial_q, targets)
        achieved, pos_error, ori_error = achieved_metrics(
            robot, solved, targets, link_indices
        )
        finite = bool(np.isfinite(solved).all())
        lower = np.asarray(robot.joints.lower_limits)
        upper = np.asarray(robot.joints.upper_limits)
        limits = bool(np.all(solved >= lower - 1e-4) and np.all(solved <= upper + 1e-4))
        symmetry = abs(abs(positions[0, 1]) - abs(positions[1, 1]))
        height_balance = abs(positions[0, 2] - positions[1, 2])
        score = max(pos_error) * 20.0 + max(ori_error) * 0.08 + symmetry + height_balance
        candidate = (score, solved, positions, targets, achieved, pos_error, ori_error, meta)
        if finite and limits and max(pos_error) <= 0.004 and (best is None or score < best[0]):
            best = candidate
    return best


def main():
    global ISAAC_BRANCH_TARGET_COMPENSATION_M
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", required=True, choices=("sphere", "cube", "cracker", "bleach", "pitcher", "drill"))
    parser.add_argument("--size-index", required=True, type=int, choices=range(8))
    parser.add_argument("--method", required=True, choices=("baseline", "bidex_v3"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026081201)
    parser.add_argument(
        "--disable-anchor",
        action="store_true",
        help=(
            "Do not emit the deterministic canonical smoke anchor. Formal "
            "generation uses this so every candidate is a fresh randomized pose."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.object == "cube":
        # For cube candidates the branch-preserving solution already matches
        # Isaac in Z.  Applying the +21 mm correction used by the thin YCB
        # objects makes the hands pre-lift the cube during closure and leaves
        # less than the required displacement for the actual lift phase.
        ISAAC_BRANCH_TARGET_COMPENSATION_M = (
            ISAAC_BRANCH_TARGET_COMPENSATION_M.copy()
        )
        ISAAC_BRANCH_TARGET_COMPENSATION_M[:, 2] = 0.0
    if args.output.exists():
        raise FileExistsError(args.output)
    row = load_row(args.object, args.size_index)
    mesh_path = Path(row["mesh_path"])
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    collision_volume_m3 = float(abs(mesh.volume))
    if collision_volume_m3 <= 0.0:
        raise RuntimeError(f"non-positive collision volume: {mesh_path}")
    target_mass_kg = 0.55
    recommended_density = float(target_mass_kg / collision_volume_m3)
    table_top = 0.71
    translation = np.asarray(
        [0.50, 0.0, table_top - float(mesh.bounds[0, 2])], dtype=np.float32
    )
    center = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float32) + translation
    half_y = float(mesh.extents[1] * 0.5)

    template = torch.load(TEMPLATE, map_location="cpu", weights_only=False)["samples"][TEMPLATE_SAMPLE_INDEX]
    _, robot = fixed_solver()
    ik_names = list(robot.joints.actuated_names)
    link_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    pose_solver = make_reusable_pose_solver(robot, link_indices)
    initial_q = ik_q(template, ik_names)
    samples = []
    attempts = 0
    while len(samples) < args.count and attempts < args.count * 12:
        candidate_seed = args.seed + attempts
        attempts += 1
        rng = np.random.default_rng(candidate_seed)
        result = solve_candidate(
            robot, pose_solver, initial_q, link_indices, args.method, rng, center, half_y,
            anchor=(attempts == 1 and not args.disable_anchor),
            reuse_initial_anchor=(
                attempts == 1 and args.object == "sphere" and not args.disable_anchor
            ),
        )
        if result is None:
            continue
        score, solved, positions, targets, achieved, pos_error, ori_error, meta = result
        sample = copy.deepcopy(template)
        sample["method"] = args.method
        sample["object_name"] = row["object_name"]
        sample["base_object_name"] = row["source_object_name"]
        sample["object_mesh_path"] = str(mesh_path)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = translation
        sample["object_pose_world"] = pose.tolist()
        sample["object_transform"] = {
            "scale_mode": "uniform",
            "uniform_scale": row["uniform_scale"],
            "canonical_rotation": row["canonical_rotation"],
            "extents_m": row["extents_m"],
            "size_index": args.size_index,
        }
        sample["physical_parameters"] = {
            "target_mass_kg": target_mass_kg,
            "collision_volume_m3": collision_volume_m3,
            "recommended_density_kg_m3": recommended_density,
            "friction": 2.0,
            "gravity_m_s2": 9.8,
        }
        expand_q(sample, solved, ik_names)
        if meta.get("canonical_anchor"):
            set_hand_posture(sample)
        else:
            perturb_hand_posture(sample, args.method, rng)
        sample["target_tcp_positions_world"] = positions.tolist()
        sample["achieved_tcp_positions_world"] = achieved
        sample["ik_position_error_m"] = pos_error
        sample["isaaclab_physical_validated"] = False
        sample["isaac_gym_physical_validated"] = False

        if meta.get("canonical_anchor") and args.object == "sphere":
            pre_positions = np.asarray(
                template["pregrasp_target_tcp_positions_world"], dtype=np.float32
            )
            sample["pregrasp_full_body_q"] = copy.deepcopy(
                template["pregrasp_full_body_q"]
            )
            sample["pregrasp_target_tcp_positions_world"] = pre_positions.tolist()
            sample["pregrasp_achieved_tcp_positions_world"] = copy.deepcopy(
                template["pregrasp_achieved_tcp_positions_world"]
            )
            pre_pos_error = np.linalg.norm(
                np.asarray(sample["pregrasp_achieved_tcp_positions_world"])
                - pre_positions,
                axis=1,
            )
        else:
            pre_positions = positions.copy()
            pre_positions[0, 1] += 0.12
            pre_positions[1, 1] -= 0.12
            pre_targets = [
                jaxlie.SE3.from_rotation_and_translation(
                    targets[i].rotation(), jnp.asarray(pre_positions[i])
                )
                for i in range(2)
            ]
            pre_solved = solve_targets(pose_solver, solved, pre_targets)
            pre_sample = copy.deepcopy(sample)
            expand_q(pre_sample, pre_solved, ik_names)
            sample["pregrasp_full_body_q"] = pre_sample["full_body_q"]
            sample["pregrasp_target_tcp_positions_world"] = pre_positions.tolist()
            pre_achieved, pre_pos_error, _ = achieved_metrics(
                robot, pre_solved, pre_targets, link_indices
            )
            sample["pregrasp_achieved_tcp_positions_world"] = pre_achieved
        sample["compact_v2"] = {
            "candidate_index": len(samples),
            "candidate_seed": candidate_seed,
            "size_index": args.size_index,
            "method": args.method,
            "paired_local_proposals": 1 if args.method == "baseline" else 6,
            "optimization_score": float(score),
            "orientation_error_deg": (np.asarray(ori_error) * 180.0 / np.pi).tolist(),
            "pregrasp_position_error_mm": (np.asarray(pre_pos_error) * 1000.0).tolist(),
            **meta,
        }
        samples.append(sample)

    if not samples:
        raise RuntimeError(f"generated no candidates after {attempts} attempts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "xhand_compact_candidate_bank_v2",
            "object": args.object,
            "method": args.method,
            "size_index": args.size_index,
            "samples": samples,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "requested_count": args.count, "count": len(samples), "attempts": attempts}, indent=2))


if __name__ == "__main__":
    main()
