#!/usr/bin/env python3
"""Package a repeat-verified large-object tabletop bimanual sample."""

import argparse
import csv
import json
from pathlib import Path

import torch


def mm(value):
    return float(value) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_payload = torch.load(
        args.verification_dir / "source_sample.pt",
        map_location="cpu",
        weights_only=False,
    )
    source = source_payload["samples"][0]
    both = torch.load(args.verification_dir / "both_result.pt", weights_only=False)
    left = torch.load(args.verification_dir / "left_result.pt", weights_only=False)
    right = torch.load(args.verification_dir / "right_result.pt", weights_only=False)
    geometry = torch.load(
        args.verification_dir / "geometry_result.pt", weights_only=False
    )
    quality = json.loads((args.verification_dir / "local_quality.json").read_text())

    rows = []
    for index in range(3):
        geom = geometry[index]
        local = quality["samples"][index]
        strict = bool(
            both["success"][index]
            and not left["success"][index]
            and not right["success"][index]
            and geom["left"]["penetration_depth_mm"] <= 2.0
            and geom["right"]["penetration_depth_mm"] <= 2.0
            and geom["left"]["contact_link_count"] >= 3
            and geom["right"]["contact_link_count"] >= 3
            and local["decoupled_force_closure_pass"]
        )
        rows.append(
            {
                "repeat": index + 1,
                "strict_success": strict,
                "lift_mm": mm(both["lift_displacement_z"][index]),
                "gravity_displacement_mm": mm(both["gravity_displacement"][index]),
                "max_6dir_displacement_mm": mm(both["max_direction_displacement"][index]),
                "left_only_success": bool(left["success"][index]),
                "right_only_success": bool(right["success"][index]),
                "left_contact_points_2mm": geom["left"]["contact_point_count"],
                "right_contact_points_2mm": geom["right"]["contact_point_count"],
                "left_contact_links": geom["left"]["contact_link_count"],
                "right_contact_links": geom["right"]["contact_link_count"],
                "left_penetration_mm": geom["left"]["penetration_depth_mm"],
                "right_penetration_mm": geom["right"]["penetration_depth_mm"],
                "hand_clearance_mm": geom["clearance_mm"],
                "left_mean_wrench_residual": local["left"]["mean_wrench_residual"],
                "right_mean_wrench_residual": local["right"]["mean_wrench_residual"],
            }
        )
    if not all(row["strict_success"] for row in rows):
        raise RuntimeError(f"Repeat hard gate failed: {rows}")

    # Select the least-penetrating repeat, then prefer larger hand clearance.
    representative = max(
        range(3),
        key=lambda i: (
            -rows[i]["left_penetration_mm"] - rows[i]["right_penetration_mm"],
            rows[i]["hand_clearance_mm"],
        ),
    )
    representative_q = torch.load(
        args.verification_dir / "repeat_final_samples.pt", weights_only=False
    )["samples"][representative]
    base_metrics = source["metrics"]
    rep = rows[representative]
    sample = {
        "object_name": source["object_name"],
        "left_q": representative_q["left_q"].clone(),
        "right_q": representative_q["right_q"].clone(),
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
            "repeat_count": 3,
            "representative_repeat": representative + 1,
            "lift_displacement_mm": rep["lift_mm"],
            "gravity_displacement_mm": rep["gravity_displacement_mm"],
            "max_direction_displacement_mm": rep["max_6dir_displacement_mm"],
            "left_contact_point_count_2mm": rep["left_contact_points_2mm"],
            "right_contact_point_count_2mm": rep["right_contact_points_2mm"],
            "left_contact_link_count": rep["left_contact_links"],
            "right_contact_link_count": rep["right_contact_links"],
            "left_penetration_depth_mm": rep["left_penetration_mm"],
            "right_penetration_depth_mm": rep["right_penetration_mm"],
            "hand_clearance_mm": rep["hand_clearance_mm"],
            "left_only_success": False,
            "right_only_success": False,
            "local_cooperative_quality_pass": True,
            "side_approach_pass": bool(base_metrics["side_approach_pass"]),
            "closure_path_support_pass": bool(base_metrics["closure_path_support_pass"]),
            "object_horizontal_span_mm": base_metrics["object_horizontal_span_mm"],
            "support_height_mm": base_metrics["support_height_mm"],
            "left_root_height_above_table_mm": base_metrics["left_root_height_above_table_mm"],
            "right_root_height_above_table_mm": base_metrics["right_root_height_above_table_mm"],
            "left_closure_path_min_world_z_mm": base_metrics["left_closure_path_min_world_z_mm"],
            "right_closure_path_min_world_z_mm": base_metrics["right_closure_path_min_world_z_mm"],
        },
        "provenance": {
            "candidate_index": source["candidate_index"],
            "source_pair_index": source["source_pair_index"],
            "opposition_roll_degrees": source["opposition_roll_degrees"],
        },
    }
    manifest = {
        "schema": "tro_grasp_xlarge_tabletop_bimanual_v1",
        "generation_method": "region_opposed_tabletop_plus_cooperative_local_quality",
        "num_unique_samples": 1,
        "object_extents_mm": [240.0, 240.0, 180.0],
        "minimum_horizontal_span_mm": 220.0,
        "gravity_m_s2": 9.8,
        "robot_friction": 1.0,
        "object_friction": 1.0,
        "finger_effort_limit_nm": 0.7,
        "object_density_kg_m3": 50.0,
        "object_mass_kg": float(both["object_mass_kg"][representative]),
        "contact_mm": 2.0,
        "max_penetration_mm": 2.0,
        "min_contact_links_per_hand": 3,
        "max_mean_local_wrench_residual": 0.35,
        "tabletop_constraints": {
            "object_bottom_is_support_height": True,
            "root_must_be_above_table": True,
            "opposed_lateral_roots": True,
            "closure_path_samples": 5,
            "entire_hand_above_table_at_all_samples": True,
            "no_bottom_approach": True,
        },
        "repeat_count": 3,
        "single_hand_ablation_required_to_fail": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "samples": [sample]}, args.output_dir / "bimanual_dataset.pt")
    summary = {"manifest": manifest, "representative_metrics": sample["metrics"], "repeat_results": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "repeat_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
