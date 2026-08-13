#!/usr/bin/env python3
"""Materialize the physically realized Tianji/XHand sphere grasp sample."""

import copy
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "migration_4090/results/xhand_compact_sphere_290mm_x0p50_v1_upward_palm.pt"
FULL_REPORT = ROOT / "migration_4090/results/xhand_sphere_290mm_strict_disturbance_fix_v1/full.json"
REPEATS = [
    FULL_REPORT,
    ROOT / "migration_4090/results/xhand_sphere_290mm_strict_disturbance_fix_v1/repeat2.json",
    ROOT / "migration_4090/results/xhand_sphere_290mm_strict_disturbance_fix_v1/repeat3.json",
]
OUT_DIR = ROOT / "migration_4090/results/xhand_sphere_290mm_strict_final_v2"
OUTPUT = OUT_DIR / "xhand_sphere_290mm_strict_success_v2.pt"
SUMMARY = OUT_DIR / "xhand_sphere_290mm_strict_success_v2.json"


def load_json(path):
    return json.loads(path.read_text())


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)
    source = torch.load(DATASET, map_location="cpu", weights_only=False)
    full_report = load_json(FULL_REPORT)
    repeat_reports = [load_json(path) for path in REPEATS]
    both = full_report["runs"]["both"]
    left = full_report["runs"]["left"]
    right = full_report["runs"]["right"]
    repeat_passes = [bool(report["runs"]["both"]["physical_pass"]) for report in repeat_reports]
    if not both["physical_pass"]:
        raise RuntimeError("fixed-disturbance both-hand report did not pass")
    if left["physical_pass"] or right["physical_pass"]:
        raise RuntimeError("single-hand ablation unexpectedly passed")
    if not all(repeat_passes):
        raise RuntimeError(f"repeat failure: {repeat_passes}")

    sample = copy.deepcopy(source["samples"][7])
    q_by_name = dict(zip(sample["joint_names"], sample["full_body_q"]))
    q_by_name.update(both["actual_dof_positions_after_closure"])
    sample["full_body_q"] = [float(q_by_name[name]) for name in sample["joint_names"]]
    closure_position = both["object_root_position_after_closure_world"]
    pose = sample["object_pose_world"]
    pose[0][3], pose[1][3], pose[2][3] = map(float, closure_position)
    sample["object_pose_world"] = pose
    sample["achieved_tcp_positions_world"] = [
        both["isaac_hand_ee_positions_after_closure_world"]["left_hand_ee_link"],
        both["isaac_hand_ee_positions_after_closure_world"]["right_hand_ee_link"],
    ]
    sample["isaaclab_physical_validated"] = True
    sample["isaac_gym_physical_validated"] = True
    sample["strict_validation"] = {
        "full_report": str(FULL_REPORT),
        "repeat_reports": [str(path) for path in REPEATS],
        "repeat_successes": int(sum(repeat_passes)),
        "repeat_count": len(repeat_passes),
        "repeat_success_rate": float(sum(repeat_passes) / len(repeat_passes)),
        "left_only_physical_pass": bool(left["physical_pass"]),
        "right_only_physical_pass": bool(right["physical_pass"]),
        "both_metrics": both,
        "left_only_metrics": left,
        "right_only_metrics": right,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "xhand_sphere_290mm_strict_success_v2",
            "samples": [sample],
        },
        OUTPUT,
    )
    summary = {
        "schema": "xhand_sphere_290mm_strict_success_v2",
        "output": str(OUTPUT),
        "object_name": sample["object_name"],
        "object_pose_world": sample["object_pose_world"],
        "repeat_success_rate": sample["strict_validation"]["repeat_success_rate"],
        "both_physical_pass": True,
        "left_only_physical_pass": False,
        "right_only_physical_pass": False,
        "lift_height_mm": both["lift_object_displacement_m"] * 1000.0,
        "gravity_displacement_mm": both["gravity_displacement_m"] * 1000.0,
        "six_direction_displacements_mm": [
            value * 1000.0 for value in both["six_direction_displacements_m"]
        ],
        "max_six_direction_displacement_mm": max(
            both["six_direction_displacements_m"]
        ) * 1000.0,
        "left_closure_contact_count": both["closure"]["left_contact_count"],
        "right_closure_contact_count": both["closure"]["right_contact_count"],
        "left_lift_contact_count": both["lifted"]["left_contact_count"],
        "right_lift_contact_count": both["lifted"]["right_contact_count"],
        "penetration_pass": both["penetration_pass"],
        "object_mass_kg": both["object_mass_kg"],
        "density_kg_m3": both["object_density_kg_m3"],
        "friction": both["object_friction"],
        "native_effort_limits": both["effort_limits"],
        "controller": both["controller"],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(OUTPUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
