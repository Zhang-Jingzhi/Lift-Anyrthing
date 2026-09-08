#!/usr/bin/env python3
"""Build paired-palm micro-lift IK targets without jax.

Same problem and same output schema as build_micro_lift_targets.py: take each
candidate's grasp pose, raise both palm frames by the requested height, and
re-solve the arms while the BODex finger shape is preserved, reusing the
damped-least-squares solver written for the retracted pregrasp.

Written on 2026-09-08 believing no interpreter here had jax; one does, at
/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro.  See
build_retracted_pregrasp_numpy for why this is kept and when to prefer the
Pyroki version.

The solver clamps to the joint limits on every step: without that the seven-DOF
null space parks left_j6 past its 1.05 rad limit and Isaac silently clamps it,
so the pose simulated is not the pose solved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from yourdfpy import URDF

from migration_4090.xhand_rl_staged.build_retracted_pregrasp_numpy import (
    SIDE_PREFIX,
    TARGET_LINKS,
    link_pose,
    sha256_file,
    solve_side,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--lift-height-m", type=float, nargs="+", default=(0.010,))
    parser.add_argument("--max-position-error-m", type=float, default=0.006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    heights = tuple(sorted({float(value) for value in args.lift_height_m}))
    if any(height <= 0.0 for height in heights):
        raise ValueError("lift heights must be positive")

    bank = torch.load(args.bodex_bank, map_location="cpu", weights_only=False)
    robot = URDF.load(str(args.urdf))
    arm_joints = {
        prefix: [prefix + "_j" + str(index) for index in range(1, 8)]
        for prefix in SIDE_PREFIX
    }
    limits = {
        joint.name: (float(joint.limit.lower), float(joint.limit.upper))
        for joint in robot.robot.joints
        if joint.type != "fixed" and joint.limit is not None
    }

    targets = {"%.6f" % h: [] for h in heights}
    diagnostics = []
    for index, sample in enumerate(bank["samples"]):
        names = list(sample["joint_names"])
        source = sample.get("controller_grasp_full_body_q", sample.get("full_body_q"))
        if source is None:
            raise RuntimeError("candidate %d has no grasp pose" % index)
        grasp = dict(zip(names, [float(v) for v in source]))
        record = {
            "candidate_index": index,
            "candidate_id": sample["candidate_id"],
            "controller_grasp_used": "controller_grasp_full_body_q" in sample,
            "heights": {},
        }
        for height in heights:
            solved = dict(grasp)
            errors = []
            for prefix, link in zip(SIDE_PREFIX, TARGET_LINKS):
                pose = link_pose(robot, grasp, link)
                target = pose.copy()
                target[2, 3] = pose[2, 3] + height
                solved, position_error, orientation_error = solve_side(
                    robot, solved, arm_joints[prefix], link, target, limits
                )
                errors.append((position_error, orientation_error))
            worst = max(error for error, _ in errors)
            if worst > args.max_position_error_m:
                raise RuntimeError(
                    "candidate %d lift IK failed at %.3f m: %.4f mm"
                    % (index, height, worst * 1000)
                )
            record["heights"]["%.6f" % height] = {
                "position_error_m": [error for error, _ in errors],
                "orientation_error_rad": [error for _, error in errors],
            }
            targets["%.6f" % height].append([solved[name] for name in names])
            print(
                "candidate %d %-36s lift %5.1fmm  pos_err %7.4fmm  rot_err %7.4fmrad"
                % (
                    index,
                    sample["candidate_id"][:34],
                    height * 1000,
                    worst * 1000,
                    max(e for _, e in errors) * 1000,
                )
            )
        diagnostics.append(record)

    payload = {
        "schema": "xhand_bodex_nonpromotional_micro_lift_targets_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "object": args.object,
        "source_bodex_bank": str(args.bodex_bank.resolve()),
        "source_bodex_bank_sha256": sha256_file(args.bodex_bank),
        "candidate_ids": [s["candidate_id"] for s in bank["samples"]],
        "joint_names": list(bank["samples"][0]["joint_names"]),
        "lift_heights_m": list(heights),
        "targets_by_height": targets,
        "diagnostics": diagnostics,
        "solver": "numpy_damped_least_squares_finite_difference",
        "non_promotional": True,
    }
    torch.save(payload, args.output)
    summary = {k: v for k, v in payload.items() if k != "targets_by_height"}
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
