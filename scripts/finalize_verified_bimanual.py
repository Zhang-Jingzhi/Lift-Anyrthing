#!/usr/bin/env python3
"""Finalize one repeatedly verified left/right bimanual grasp."""

import argparse
import csv
import json
from pathlib import Path

import torch


PENETRATION_MM = 5.0
CONTACT_MM = 5.0
CLEARANCE_MM = 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/final_v1_lr_g98_fixture/"
            "cylinder_large_candidate4"
        ),
    )
    args = parser.parse_args()
    root = args.root
    both = torch.load(root / "both_repeat5.pt", map_location="cpu")
    left = torch.load(root / "left_repeat5.pt", map_location="cpu")
    right = torch.load(root / "right_repeat5.pt", map_location="cpu")
    geometry = torch.load(root / "geometry_repeat5.pt", map_location="cpu")

    rows = []
    strict_mask = []
    for index, metrics in enumerate(geometry):
        geometry_pass = (
            metrics["left"]["penetration_depth_mm"] <= PENETRATION_MM
            and metrics["right"]["penetration_depth_mm"]
            <= PENETRATION_MM
            and metrics["left"]["min_surface_distance_mm"] <= CONTACT_MM
            and metrics["right"]["min_surface_distance_mm"] <= CONTACT_MM
            and metrics["clearance_mm"] > CLEARANCE_MM
        )
        bimanual_required = (
            bool(both["gravity_success"][index])
            and not bool(left["gravity_success"][index])
            and not bool(right["gravity_success"][index])
        )
        strict = (
            bool(both["success"][index])
            and bimanual_required
            and geometry_pass
        )
        strict_mask.append(strict)
        rows.append(
            {
                "repeat": index,
                "strict_success": strict,
                "both_gravity_success": bool(
                    both["gravity_success"][index]
                ),
                "left_only_gravity_success": bool(
                    left["gravity_success"][index]
                ),
                "right_only_gravity_success": bool(
                    right["gravity_success"][index]
                ),
                "gravity_displacement_mm": float(
                    both["gravity_displacement"][index] * 1000
                ),
                "max_direction_displacement_mm": float(
                    both["max_direction_displacement"][index] * 1000
                ),
                "left_penetration_mm": metrics["left"][
                    "penetration_depth_mm"
                ],
                "right_penetration_mm": metrics["right"][
                    "penetration_depth_mm"
                ],
                "left_contact_distance_mm": metrics["left"][
                    "min_surface_distance_mm"
                ],
                "right_contact_distance_mm": metrics["right"][
                    "min_surface_distance_mm"
                ],
                "left_contact_points": metrics["left"][
                    "contact_point_count"
                ],
                "right_contact_points": metrics["right"][
                    "contact_point_count"
                ],
                "hand_clearance_mm": metrics["clearance_mm"],
            }
        )

    if not all(strict_mask):
        raise RuntimeError(f"Strict repeat verification failed: {strict_mask}")

    repeat_samples = []
    for index in range(len(strict_mask)):
        repeat_samples.append(
            {
                "object_name": "contactdb+cylinder_large",
                "left_robot_name": "allegro_left",
                "right_robot_name": "allegro_right",
                "left_q": both["left_q_final"][index],
                "right_q": both["right_q_final"][index],
                "seed_repeat_index": index,
                "metrics": rows[index],
            }
        )
    torch.save(repeat_samples, root / "verified_repeat_rollouts.pt")
    # The five runs originate from one seed. Preserve one representative as
    # one independent training datum rather than counting repeats as diversity.
    torch.save([repeat_samples[0]], root / "verified_unique_pairs.pt")

    with (root / "verification_rows.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gravity = [row["gravity_displacement_mm"] for row in rows]
    disturbance = [row["max_direction_displacement_mm"] for row in rows]
    penetration = [
        max(row["left_penetration_mm"], row["right_penetration_mm"])
        for row in rows
    ]
    summary = {
        "object_name": "contactdb+cylinder_large",
        "independent_pose_count": 1,
        "repeat_count": len(rows),
        "strict_repeat_success_count": sum(strict_mask),
        "strict_repeat_success_rate": sum(strict_mask) / len(rows),
        "gravity_m_s2": 9.8,
        "gravity_from_first_frame": True,
        "simultaneous_closure": True,
        "fixture_during_closure": True,
        "fixture_released_before_evaluation": True,
        "gravity_displacement_mm_range": [min(gravity), max(gravity)],
        "max_direction_displacement_mm_range": [
            min(disturbance),
            max(disturbance),
        ],
        "max_penetration_mm": max(penetration),
        "left_only_success_count": sum(
            row["left_only_gravity_success"] for row in rows
        ),
        "right_only_success_count": sum(
            row["right_only_gravity_success"] for row in rows
        ),
        "thresholds": {
            "gravity_displacement_mm": 20.0,
            "six_direction_displacement_mm": 20.0,
            "penetration_mm": PENETRATION_MM,
            "contact_distance_mm": CONTACT_MM,
            "hand_clearance_mm": CLEARANCE_MM,
        },
    }
    (root / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    report = f"""# Verified left/right bimanual grasp

## Result

- Object: `contactdb+cylinder_large`
- Independent grasp poses: **1**
- Repeat verification: **{sum(strict_mask)}/{len(rows)} strict successes**
- Both hands close simultaneously under `g=9.8 m/s^2`.
- An explicit pose fixture holds the object during closure and is released
  before gravity and disturbance evaluation.
- Left-only gravity success: **0/{len(rows)}**
- Right-only gravity success: **0/{len(rows)}**

## Strict metrics across five repeats

- Gravity displacement: **{min(gravity):.2f}–{max(gravity):.2f} mm**
- Six-direction maximum displacement: **{min(disturbance):.2f}–{max(disturbance):.2f} mm**
- Maximum realized penetration: **{max(penetration):.2f} mm**
- Both realized hands remain within **{CONTACT_MM:.1f} mm** of the object.
- Inter-hand clearance remains above **{CLEARANCE_MM:.1f} mm**.

The five rollouts verify repeatability but are not counted as five independent
training poses. `verified_unique_pairs.pt` therefore contains one representative
pose; `verified_repeat_rollouts.pt` retains all repeats for auditability.
"""
    (root / "VERIFICATION_REPORT.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
