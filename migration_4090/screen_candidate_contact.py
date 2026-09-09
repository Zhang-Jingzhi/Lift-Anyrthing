#!/usr/bin/env python3
"""Screen candidates on whether each hand reaches the object, per hand.

The bank these candidates come from was selected for zero overlap, which accepts
a grasp that closes on nothing: measured on rl_bank_zero_overlap4, the per-hand
clearance at the grasp pose is 2.20/7.31, 1.20/1.19, 7.52/0.48 and 14.10/13.18 mm
(left/right), every one of them positive.  Executed faithfully -- self-collisions
off so the fingers hold their commanded pose -- those grasps produce bilateral
contact 0.000 and move the ball 3.3 mm.

The audit geometry was never the problem: the USD collision meshes and the URDF
visual meshes agree to 0.000 mm on all 96 links, so what the audit measures is
what PhysX collides with, up to the convex approximation.  The criterion was.
"No overlap" and "in contact" are different requirements, and only the second is
a grasp.

Reports each hand separately, because the two are not equidistant and a combined
minimum hides the hand that is short: candidate 2 reads 0.48 mm on a combined
measure and 7.52 mm on its left hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from yourdfpy import URDF

sys.path.insert(0, "/media/home/zhangjingzhi/TRO-Grasp-Reproduction/migration_4090")
from measure_bank_overlap import (  # noqa: E402
    SAMPLES_PER_LINK,
    hand_link_names,
    object_mesh,
)

SIDES = ("left", "right")


def side_links(robot: URDF, side: str) -> list:
    initial = side[0].upper() + "_"
    return [
        name
        for name in hand_link_names(robot)
        if side in name.lower() or name.startswith(initial)
    ]


def clearance_m(robot: URDF, links, obj, values: dict) -> float:
    """Smallest distance from these links to the object; negative is inside."""
    robot.update_cfg(values)
    worst = float("inf")
    for link in links:
        transform, name = robot.scene.graph.get(link)
        geometry = robot.scene.geometry.get(name)
        if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
            continue
        points, _ = trimesh.sample.sample_surface(geometry, SAMPLES_PER_LINK)
        points = trimesh.transform_points(points, np.asarray(transform))
        worst = min(worst, -float(np.max(trimesh.proximity.signed_distance(obj, points))))
    return worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument(
        "--contact-band-m",
        type=float,
        nargs=2,
        default=(-0.003, 0.000),
        help="a hand passes when its clearance falls inside this band",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    low, high = sorted(args.contact_band_m)

    bank = torch.load(args.bodex_bank, map_location="cpu", weights_only=False)
    robot = URDF.load(str(args.urdf))
    links = {side: side_links(robot, side) for side in SIDES}

    print(
        "contact band [%.1f, %.1f] mm; a hand outside it either misses the object "
        "or is buried in it" % (low * 1000.0, high * 1000.0)
    )
    print()
    print(
        "%-6s %-36s %11s %11s %8s"
        % ("cand", "id", "left gap", "right gap", "verdict")
    )
    rows = []
    passed = 0
    for index, sample in enumerate(bank["samples"]):
        names = list(sample["joint_names"])
        source = sample.get(
            "controller_grasp_full_body_q", sample.get("full_body_q")
        )
        values = dict(zip(names, [float(v) for v in source]))
        obj = object_mesh(sample)
        gaps = {
            side: clearance_m(robot, links[side], obj, values) for side in SIDES
        }
        ok = all(low <= gap <= high for gap in gaps.values())
        passed += bool(ok)
        rows.append(
            {
                "candidate_index": index,
                "candidate_id": sample["candidate_id"],
                "left_clearance_m": gaps["left"],
                "right_clearance_m": gaps["right"],
                "in_contact_band": bool(ok),
            }
        )
        print(
            "%-6d %-36s %8.2f mm %8.2f mm %8s"
            % (
                index,
                sample["candidate_id"][:36],
                gaps["left"] * 1000.0,
                gaps["right"] * 1000.0,
                "pass" if ok else "FAIL",
            )
        )
    print()
    print("%d of %d candidates have both hands in the contact band" % (passed, len(rows)))
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "bodex_bank": str(args.bodex_bank.resolve()),
                    "contact_band_m": [low, high],
                    "passed": passed,
                    "total": len(rows),
                    "candidates": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print("wrote", args.output)


if __name__ == "__main__":
    main()
