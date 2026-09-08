#!/usr/bin/env python3
"""Move both palms toward the object and re-solve the arms.

The BODex grasp poses do not reach the sphere: measured 2026-09-08, their
minimum clearance to the exact mesh is 0.34, 0.96, 2.48 and 13.47 mm across the
four zero_overlap4 candidates, and against the convex hulls PhysX collides with
it is clear as well.  Executed faithfully the hands close on air, and every
contact the pipeline has produced came from self-collision artifacts deflecting
the fingers into the ball.

Closing the fingers further is the wrong lever for a 296 mm sphere -- it curls
them into the palm and leaves one or two contact groups per side where the
stage-2 criteria want at least two.  This moves the whole hand in instead, along
each palm's own normal, so the open BODex finger shape is preserved and only the
distance changes.

The advanced poses are written as controller_pregrasp_full_body_q and
controller_grasp_full_body_q, which the staged environment already prefers over
the raw fields, so the originals stay intact for audit.

Reports the achieved displacement per hand, measured by forward kinematics
rather than assumed, along with the IK residual and any joint pushed to a limit.
An advance of 1 mm is meaningless if the solver is allowed several mm of error,
so --max-position-error-m defaults to a tenth of the smallest advance.
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

import sys

sys.path.insert(
    0, "/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090/xhand_rl_staged"
)
import trimesh  # noqa: E402

from build_retracted_pregrasp_numpy import (  # noqa: E402
    PALM_LOCAL_NORMAL,
    SIDE_PREFIX,
    TARGET_LINKS,
    link_pose,
    solve_side,
)

sys.path.insert(0, "/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090")
from measure_bank_overlap import (  # noqa: E402
    SAMPLES_PER_LINK,
    hand_link_names,
    object_mesh,
)


def side_link_names(robot, side: str) -> list:
    """Hand links belonging to one side, by the URDF's own naming."""
    prefix = side.lower()
    initial = prefix[0].upper() + "_"
    return [
        name
        for name in hand_link_names(robot)
        if prefix in name.lower() or name.startswith(initial)
    ]


def surface_clearance_m(robot, links, obj, values: dict) -> float:
    """Smallest distance from these links to the object; negative means inside."""
    robot.update_cfg(values)
    worst = float("inf")
    for link in links:
        transform, name = robot.scene.graph.get(link)
        geometry = robot.scene.geometry.get(name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        points, _ = trimesh.sample.sample_surface(geometry, SAMPLES_PER_LINK)
        points = trimesh.transform_points(points, np.asarray(transform))
        inside = trimesh.proximity.signed_distance(obj, points)
        worst = min(worst, -float(np.max(inside)))
    return worst

POSE_FIELDS = (
    ("pregrasp_full_body_q", "controller_pregrasp_full_body_q"),
    ("full_body_q", "controller_grasp_full_body_q"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--advance-m",
        type=float,
        help="common distance each palm moves along its own inward normal",
    )
    parser.add_argument(
        "--target-clearance-m",
        type=float,
        help=(
            "seat every palm this far from the object surface, advancing each "
            "hand by its own measured clearance minus this value; negative "
            "values seat the hand slightly into the surface"
        ),
    )
    parser.add_argument("--max-position-error-m", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.advance_m is None) == (args.target_clearance_m is None):
        raise ValueError("pass exactly one of --advance-m or --target-clearance-m")
    if args.advance_m is not None and args.advance_m < 0.0:
        raise ValueError("advance must be non-negative; 0 writes the control bank")
    tolerance = args.max_position_error_m
    if tolerance is None:
        reference = args.advance_m if args.advance_m is not None else 0.005
        tolerance = max(reference * 0.1, 1.0e-4)
    if args.output.exists():
        raise FileExistsError(args.output)

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

    side_links = {side: side_link_names(robot, side) for side in SIDE_PREFIX}
    if args.advance_m is not None:
        print(
            "common advance %.1f mm, IK tolerance %.3f mm"
            % (args.advance_m * 1000.0, tolerance * 1000.0)
        )
    else:
        print(
            "seat every palm at %.1f mm clearance, IK tolerance %.3f mm"
            % (args.target_clearance_m * 1000.0, tolerance * 1000.0)
        )
    print(
        "%-6s %-10s %12s %12s %12s %10s"
        % ("cand", "pose", "commanded", "achieved", "IK residual", "at limit")
    )
    diagnostics = []
    worst_shortfall = 0.0
    for index, sample in enumerate(bank["samples"]):
        names = list(sample["joint_names"])
        record = {"candidate_index": index, "candidate_id": sample["candidate_id"]}
        obj = object_mesh(sample)
        for source_field, target_field in POSE_FIELDS:
            values = {n: float(v) for n, v in zip(names, sample[source_field])}
            solved = dict(values)
            per_hand = {}
            for prefix, link in zip(SIDE_PREFIX, TARGET_LINKS):
                start_pose = link_pose(robot, values, link)
                inward = start_pose[:3, :3] @ PALM_LOCAL_NORMAL
                residual = 0.0
                if args.advance_m is not None:
                    advance = args.advance_m
                    if advance > 0.0:
                        target = start_pose.copy()
                        target[:3, 3] = start_pose[:3, 3] + advance * inward
                        solved, residual, _ = solve_side(
                            robot, solved, arm_joints[prefix], link, target, limits
                        )
                else:
                    # The clearance is measured to the nearest point on the
                    # object, which is not generally along the palm normal, so
                    # advancing 1 mm closes less than 1 mm of gap -- about 0.85
                    # of it on this sphere.  A single step leaves the far
                    # candidates short by millimetres, so this measures and
                    # steps until the gap is where it was asked to be.
                    advance = 0.0
                    for _ in range(8):
                        clearance = surface_clearance_m(
                            robot, side_links[prefix], obj, solved
                        )
                        error = clearance - args.target_clearance_m
                        if abs(error) <= tolerance:
                            break
                        advance += error
                        target = start_pose.copy()
                        target[:3, 3] = start_pose[:3, 3] + advance * inward
                        solved, residual, _ = solve_side(
                            robot, solved, arm_joints[prefix], link, target, limits
                        )
                achieved_pose = link_pose(robot, solved, link)
                moved = float(
                    np.dot(achieved_pose[:3, 3] - start_pose[:3, 3], inward)
                )
                at_limit = [
                    name
                    for name in arm_joints[prefix]
                    if abs(solved[name] - limits[name][0]) < 1.0e-6
                    or abs(solved[name] - limits[name][1]) < 1.0e-6
                ]
                per_hand[prefix] = {
                    "commanded_m": advance,
                    "achieved_along_normal_m": moved,
                    "ik_residual_m": residual,
                    "joints_at_limit": at_limit,
                }
                worst_shortfall = max(worst_shortfall, abs(moved - advance))
                print(
                    "%-6d %-10s %9.3f mm %9.3f mm %9.4f mm %10s"
                    % (
                        index,
                        target_field.split("_")[1],
                        advance * 1000.0,
                        moved * 1000.0,
                        residual * 1000.0,
                        ",".join(at_limit) if at_limit else "-",
                    )
                )
            sample[target_field] = [solved[name] for name in names]
            record[target_field] = per_hand
        diagnostics.append(record)

    bank["controller_pose_advance_m"] = args.advance_m
    bank["controller_pose_target_clearance_m"] = args.target_clearance_m
    bank["controller_pose_advance_source_bank_sha256"] = sha256_file(args.bodex_bank)
    bank["controller_pose_advance_diagnostics"] = diagnostics
    bank["controller_pose_advance_created_at"] = datetime.now(timezone.utc).isoformat()
    torch.save(bank, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "advance_m": args.advance_m,
        "target_clearance_m": args.target_clearance_m,
        "ik_tolerance_m": tolerance,
        "worst_displacement_shortfall_m": worst_shortfall,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps({**summary, "diagnostics": diagnostics}, indent=2) + "\n"
    )
    # What matters is where the palms ended up, not the intermediate IK residual.
    # A hand can overshoot its commanded advance when the limit projection moves
    # the solution, which seats it slightly deeper -- harmless -- while falling
    # short leaves it off the surface, which is the failure this guards against.
    print()
    print("%-6s %-10s %10s %10s" % ("cand", "pose", "left gap", "right gap"))
    worst_gap_error = 0.0
    for index, sample in enumerate(bank["samples"]):
        obj = object_mesh(sample)
        names = list(sample["joint_names"])
        for _, target_field in POSE_FIELDS:
            values = {n: float(v) for n, v in zip(names, sample[target_field])}
            gaps = {
                side: surface_clearance_m(robot, side_links[side], obj, values)
                for side in SIDE_PREFIX
            }
            if args.target_clearance_m is not None:
                for gap in gaps.values():
                    worst_gap_error = max(
                        worst_gap_error, gap - args.target_clearance_m
                    )
            print(
                "%-6d %-10s %7.3f mm %7.3f mm"
                % (
                    index,
                    target_field.split("_")[1],
                    gaps["left"] * 1000.0,
                    gaps["right"] * 1000.0,
                )
            )
    summary["worst_clearance_shortfall_m"] = worst_gap_error
    args.output.with_suffix(".json").write_text(
        json.dumps({**summary, "diagnostics": diagnostics}, indent=2)
    )
    print()
    print(json.dumps(summary, indent=2))
    if args.target_clearance_m is not None and worst_gap_error > tolerance:
        raise SystemExit(
            "a palm sits %.4f mm further out than asked, over the %.4f mm tolerance"
            % (worst_gap_error * 1000.0, tolerance * 1000.0)
        )


if __name__ == "__main__":
    main()
