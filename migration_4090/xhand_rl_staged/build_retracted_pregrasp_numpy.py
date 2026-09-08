#!/usr/bin/env python3
"""Solve a retracted pregrasp without jax.

build_retracted_pregrasp.py solves this with Pyroki.  This solves the same
problem with numpy and no jax: push each palm back along its own outward normal
and re-solve the seven arm joints on that side, leaving the BODex finger shape
untouched.

Written on 2026-09-08 under the mistaken belief that no interpreter on this
machine had jax.  It does --
/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro (jax 0.6.2,
jaxlie, pyroki); the directory is hidden, which is why a search for
*/miniforge3 missed it.  Kept because it is validated against the Pyroki output
(palm positions agree to 0.011-0.026 mm on three of four candidates, 2.36 mm on
the fourth) and needs only numpy, yourdfpy and torch, so it runs in the same
interpreter as the geometry measurements.  Prefer the Pyroki version when its
extra objectives matter: this one has no wrist-rest or palm-direction term and
so picks a different null-space solution.

Damped least squares on a finite-difference Jacobian.  Fourteen joints total
across four candidates, so the cost of not having an analytic Jacobian does not
matter, and the achieved error is reported per candidate rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from yourdfpy import URDF

# Both XHand palms face along local +X (CRITERIA_v2), so retracting steps along
# each palm frame's -X.
PALM_LOCAL_NORMAL = np.array([1.0, 0.0, 0.0])
TARGET_LINKS = ("L_tcp", "R_tcp")
SIDE_PREFIX = ("left", "right")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_pose(robot: URDF, values: dict, link: str) -> np.ndarray:
    robot.update_cfg(values)
    return np.asarray(robot.get_transform(link), dtype=np.float64)


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Six-vector: translation error, then rotation error as a rotation vector."""
    translation = target[:3, 3] - current[:3, 3]
    relative = target[:3, :3] @ current[:3, :3].T
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1.0e-9:
        rotation = np.zeros(3)
    else:
        axis = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        ) / (2.0 * np.sin(angle))
        rotation = axis * angle
    return np.concatenate((translation, rotation))


def solve_side(
    robot: URDF,
    values: dict,
    joints: list,
    link: str,
    target: np.ndarray,
    limits: dict,
    *,
    iterations: int = 200,
    damping: float = 0.02,
    step: float = 1.0e-5,
):
    """Damped least squares with the joint limits projected onto every step.

    Without the projection the seven-DOF null space happily parks left_j6 at
    1.06-1.19 rad against a 1.05 rad limit, and Isaac then clamps it, so the
    pose actually simulated is not the pose that was solved.  Clamping inside
    the loop lets the remaining joints take up the slack instead.
    """
    working = dict(values)
    q = np.array([working[name] for name in joints], dtype=np.float64)
    lower = np.array([limits[name][0] for name in joints], dtype=np.float64)
    upper = np.array([limits[name][1] for name in joints], dtype=np.float64)
    q = np.clip(q, lower, upper)
    for _ in range(iterations):
        for name, value in zip(joints, q):
            working[name] = float(value)
        error = pose_error(link_pose(robot, working, link), target)
        if np.linalg.norm(error[:3]) < 1.0e-5 and np.linalg.norm(error[3:]) < 1.0e-4:
            break
        jacobian = np.zeros((6, len(joints)))
        for index in range(len(joints)):
            probe = dict(working)
            probe[joints[index]] = float(q[index] + step)
            moved = pose_error(link_pose(robot, probe, link), target)
            jacobian[:, index] = (moved - error) / step
        # error points from current to target, so the damped least squares step
        # solves J dq = -error with a Tikhonov term.
        lhs = jacobian.T @ jacobian + (damping ** 2) * np.eye(len(joints))
        q = np.clip(q - np.linalg.solve(lhs, jacobian.T @ error), lower, upper)
    for name, value in zip(joints, q):
        working[name] = float(value)
    final = pose_error(link_pose(robot, working, link), target)
    return working, float(np.linalg.norm(final[:3])), float(np.linalg.norm(final[3:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--retract-m", type=float, nargs="+", default=(0.10,))
    parser.add_argument("--max-position-error-m", type=float, default=0.006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    distances = tuple(sorted({float(value) for value in args.retract_m}))

    bank = torch.load(args.bodex_bank, map_location="cpu", weights_only=False)
    robot = URDF.load(str(args.urdf))
    actuated = set(robot.actuated_joint_names)
    arm_joints = {
        prefix: [prefix + "_j" + str(index) for index in range(1, 8)]
        for prefix in SIDE_PREFIX
    }
    for prefix, names in arm_joints.items():
        missing = [name for name in names if name not in actuated]
        if missing:
            raise RuntimeError("arm joints missing from URDF: " + str(missing))
    limits = {
        joint.name: (float(joint.limit.lower), float(joint.limit.upper))
        for joint in robot.robot.joints
        if joint.type != "fixed" and joint.limit is not None
    }
    for names in arm_joints.values():
        missing = [name for name in names if name not in limits]
        if missing:
            raise RuntimeError("arm joints without limits: " + str(missing))

    poses = {"%.6f" % d: [] for d in distances}
    diagnostics = []
    for index, sample in enumerate(bank["samples"]):
        names = list(sample["joint_names"])
        pregrasp = dict(zip(names, [float(v) for v in sample["pregrasp_full_body_q"]]))
        record = {
            "candidate_index": index,
            "candidate_id": sample["candidate_id"],
            "distances": {},
        }
        for distance in distances:
            solved = dict(pregrasp)
            errors = []
            for prefix, link in zip(SIDE_PREFIX, TARGET_LINKS):
                pose = link_pose(robot, pregrasp, link)
                target = pose.copy()
                target[:3, 3] = pose[:3, 3] - distance * (
                    pose[:3, :3] @ PALM_LOCAL_NORMAL
                )
                solved, position_error, orientation_error = solve_side(
                    robot, solved, arm_joints[prefix], link, target, limits
                )
                errors.append((position_error, orientation_error))
            worst = max(error for error, _ in errors)
            ok = bool(worst <= args.max_position_error_m)
            record["distances"]["%.6f" % distance] = {
                "success": ok,
                "position_error_m": [error for error, _ in errors],
                "orientation_error_rad": [error for _, error in errors],
            }
            poses["%.6f" % distance].append([solved[name] for name in names])
            print(
                "candidate %d %-36s retract %5.1fmm  ok=%s  pos_err %7.4fmm  rot_err %7.4fmrad"
                % (
                    index,
                    sample["candidate_id"][:34],
                    distance * 1000,
                    ok,
                    worst * 1000,
                    max(e for _, e in errors) * 1000,
                )
            )
        diagnostics.append(record)

    payload = {
        "schema": "xhand_bodex_retracted_pregrasp_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "source_bodex_bank": str(args.bodex_bank.resolve()),
        "source_bodex_bank_sha256": sha256_file(args.bodex_bank),
        "candidate_ids": [s["candidate_id"] for s in bank["samples"]],
        "joint_names": list(bank["samples"][0]["joint_names"]),
        "retract_distances_m": list(distances),
        "retracted_pregrasp_full_body_q": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in poses.items()
        },
        "diagnostics": diagnostics,
        "solver": "numpy_damped_least_squares_finite_difference",
        "non_promotional": True,
    }
    torch.save(payload, args.output)
    summary = {k: v for k, v in payload.items() if k != "retracted_pregrasp_full_body_q"}
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
