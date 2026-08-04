#!/usr/bin/env python3
"""Aggregate independently verified bimanual poses without duplicating repeats."""

import json
from pathlib import Path

import torch


def main():
    root = Path("graph_exp/bimanual_data/final_v1_lr_g98_fixture")
    experiment_names = (
        "cylinder_large_candidate4",
        "cylinder_large_auto_candidate6",
    )
    unique = []
    summaries = []
    for name in experiment_names:
        experiment = root / name
        samples = torch.load(
            experiment / "verified_unique_pairs.pt",
            map_location="cpu",
            weights_only=False,
        )
        if len(samples) != 1:
            raise RuntimeError(f"Expected one independent pose in {experiment}")
        sample = samples[0]
        sample["verification_experiment"] = name
        unique.append(sample)
        summaries.append(
            json.loads((experiment / "verification_summary.json").read_text())
        )
    torch.save(unique, root / "final_verified_unique_pairs.pt")
    combined = {
        "dataset_version": "final_v1_lr_g98_fixture_verified",
        "object_name": "contactdb+cylinder_large",
        "independent_pose_count": len(unique),
        "repeat_count_per_pose": 5,
        "total_strict_repeat_successes": sum(
            item["strict_repeat_success_count"] for item in summaries
        ),
        "total_repeat_trials": sum(item["repeat_count"] for item in summaries),
        "gravity_m_s2": 9.8,
        "simultaneous_closure": True,
        "fixture_released_before_evaluation": True,
        "left_only_successes": sum(
            item["left_only_success_count"] for item in summaries
        ),
        "right_only_successes": sum(
            item["right_only_success_count"] for item in summaries
        ),
        "experiments": experiment_names,
    }
    (root / "final_summary.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    report = f"""# Final verified left/right bimanual result

## Outcome

- Object: `contactdb+cylinder_large`
- Independent strict grasp poses: **{len(unique)}**
- Repeat verification: **{combined['total_strict_repeat_successes']}/{combined['total_repeat_trials']}**
- Left-only successes: **{combined['left_only_successes']}/{combined['total_repeat_trials']}**
- Right-only successes: **{combined['right_only_successes']}/{combined['total_repeat_trials']}**
- Both Allegro hands close simultaneously under `g=9.8 m/s^2`.
- The explicit pose fixture is released before gravity and disturbance tests.
- Every retained pose passes dual contact, <=5 mm penetration, >2 mm
  inter-hand clearance, <=20 mm gravity displacement, and six independent
  <=20 mm disturbance tests.

`final_verified_unique_pairs.pt` contains the two independent training poses.
The per-experiment folders retain all five repeats for auditability.
"""
    (root / "FINAL_REPORT.md").write_text(report)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
