#!/usr/bin/env python3
"""Package a repeat-verified, strictly bimanual Allegro grasp sample."""

import argparse
import csv
import json
from pathlib import Path

import torch


def millimetres(value):
    return float(value) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(
        args.verification_dir / "source_sample.pt",
        map_location="cpu",
        weights_only=False,
    )["samples"][0]
    repeats = torch.load(
        args.verification_dir / "repeat_final_samples.pt",
        map_location="cpu",
        weights_only=False,
    )["samples"]
    both = torch.load(
        args.verification_dir / "both_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    left_only = torch.load(
        args.verification_dir / "left_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    right_only = torch.load(
        args.verification_dir / "right_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    geometry = torch.load(
        args.verification_dir / "geometry_result_visual_surface.pt",
        map_location="cpu",
        weights_only=False,
    )
    force_audit = json.loads(
        (args.verification_dir / "repeat_force_closure.json").read_text()
    )

    count = len(repeats)
    if count != 3:
        raise RuntimeError(f"Expected three independent repeats, got {count}")
    force_rows = force_audit["samples"]
    rows = []
    for index in range(count):
        geom = geometry[index]
        force = force_rows[index]
        passed = bool(
            both["success"][index]
            and not left_only["success"][index]
            and not right_only["success"][index]
            and not left_only["gravity_success"][index]
            and not right_only["gravity_success"][index]
            and geom["left"]["penetration_depth_mm"] <= 2.0
            and geom["right"]["penetration_depth_mm"] <= 2.0
            and geom["clearance_mm"] >= 2.0
            and geom["left"]["contact_link_count"] >= 2
            and geom["right"]["contact_link_count"] >= 2
            and force["decoupled_force_closure_pass"]
        )
        rows.append(
            {
                "repeat": index + 1,
                "strict_bimanual_success": passed,
                "lift_mm": millimetres(both["lift_displacement_z"][index]),
                "gravity_displacement_mm": millimetres(
                    both["gravity_displacement"][index]
                ),
                "max_6dir_displacement_mm": millimetres(
                    both["max_direction_displacement"][index]
                ),
                "left_only_success": bool(left_only["success"][index]),
                "right_only_success": bool(right_only["success"][index]),
                "left_contact_points_2mm": geom["left"]["contact_point_count"],
                "right_contact_points_2mm": geom["right"]["contact_point_count"],
                "left_contact_links": geom["left"]["contact_link_count"],
                "right_contact_links": geom["right"]["contact_link_count"],
                "left_penetration_mm": geom["left"]["penetration_depth_mm"],
                "right_penetration_mm": geom["right"]["penetration_depth_mm"],
                "hand_clearance_mm": geom["clearance_mm"],
                "left_wrench_residual_max": force["left"][
                    "max_wrench_residual"
                ],
                "right_wrench_residual_max": force["right"][
                    "max_wrench_residual"
                ],
            }
        )

    if not all(row["strict_bimanual_success"] for row in rows):
        raise RuntimeError("At least one repeat failed the final hard gate")

    # Prefer a zero-penetration realization with the largest hand clearance.
    representative = max(
        range(count),
        key=lambda index: (
            -geometry[index]["left"]["penetration_depth_mm"]
            - geometry[index]["right"]["penetration_depth_mm"],
            geometry[index]["clearance_mm"],
        ),
    )
    representative_row = rows[representative]
    sample = {
        "object_name": source["object_name"],
        # Clone every slice so torch.save does not retain a much larger source
        # tensor storage behind a 22-value pose.
        "left_q": repeats[representative]["left_q"].clone(),
        "right_q": repeats[representative]["right_q"].clone(),
        "left_q_seed": source["left_q_seed"].clone(),
        "right_q_seed": source["right_q_seed"].clone(),
        "left_q_outer": source["left_q_outer"].clone(),
        "right_q_outer": source["right_q_outer"].clone(),
        "left_q_command": source["left_q_command"].clone(),
        "right_q_command": source["right_q_command"].clone(),
        "metrics": {
            "strict_success": True,
            "strict_bimanual_required": True,
            "repeat_success_rate": 1.0,
            "repeat_count": count,
            "representative_repeat": representative + 1,
            "lift_displacement_mm": representative_row["lift_mm"],
            "gravity_displacement_mm": representative_row[
                "gravity_displacement_mm"
            ],
            "max_direction_displacement_mm": representative_row[
                "max_6dir_displacement_mm"
            ],
            "left_contact_point_count_2mm": representative_row[
                "left_contact_points_2mm"
            ],
            "right_contact_point_count_2mm": representative_row[
                "right_contact_points_2mm"
            ],
            "left_contact_link_count": representative_row[
                "left_contact_links"
            ],
            "right_contact_link_count": representative_row[
                "right_contact_links"
            ],
            "left_penetration_depth_mm": representative_row[
                "left_penetration_mm"
            ],
            "right_penetration_depth_mm": representative_row[
                "right_penetration_mm"
            ],
            "hand_clearance_mm": representative_row["hand_clearance_mm"],
            "left_only_success": False,
            "right_only_success": False,
            "decoupled_force_closure_pass": True,
        },
        "provenance": {
            "candidate_index": source["candidate_index"],
            "left_source_index": source["left_source_index"],
            "right_source_index": source["right_source_index"],
            "opposition_roll_degrees": source["opposition_roll_degrees"],
        },
    }
    manifest = {
        "schema": "tro_grasp_bimanual_verified_v1",
        "generation_method": "bidexgrasp_inspired_hybrid_filter_v1",
        "description": (
            "Opposed left/right Allegro candidate selected by independent "
            "per-hand wrench coverage, then hard-gated in Isaac Gym."
        ),
        "num_unique_samples": 1,
        "repeat_count": count,
        "gravity_m_s2": 9.8,
        "robot_friction": 1.0,
        "object_friction": 1.0,
        "finger_effort_limit_nm": 0.7,
        "object_density_kg_m3": 500.0,
        "object_mass_kg": float(both["object_mass_kg"][representative]),
        "contact_mm": 2.0,
        "max_penetration_mm": 2.0,
        "min_hand_clearance_mm": 2.0,
        "min_contact_links_per_hand": 2,
        "max_gravity_displacement_mm": 10.0,
        "min_lift_mm": 30.0,
        "max_6dir_displacement_mm": 15.0,
        "gravity_settle_steps": 500,
        "lift_height_mm": 50.0,
        "lift_steps": 100,
        "independent_six_directions": True,
        "single_hand_ablation_required_to_fail": True,
        "local_wrench_audit": {
            "friction": force_audit["friction"],
            "contact_mm": force_audit["contact_mm"],
            "max_wrench_residual": force_audit["max_wrench_residual"],
            "note": "Geometric proxy; Isaac gravity/lift/disturbance is authoritative.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"manifest": manifest, "samples": [sample]},
        args.output_dir / "bimanual_dataset.pt",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "representative_metrics": sample["metrics"],
                "repeat_results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    with (args.output_dir / "repeat_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
