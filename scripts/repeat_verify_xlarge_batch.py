#!/usr/bin/env python3
"""Repeat-verify a batch selected by the cooperative local-quality gate."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_bimanual_pairing_baseline import run_isaac


def merge_results(parts):
    merged = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        if all(torch.is_tensor(value) for value in values):
            merged[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values
    return merged


def active_physical_contacts(rows):
    """Count positive-impulse object contacts by real hand and link."""
    points = {"left": 0, "right": 0}
    links = {"left": set(), "right": set()}
    for row in rows:
        if float(row.get("normal_impulse", 0.0)) <= 0.0:
            continue
        side = row["actor1"] if row["actor0"] == "object" else row["actor0"]
        if side not in points:
            continue
        link = row["body1_name"] if row["actor1"] == side else row["body0_name"]
        points[side] += 1
        links[side].add(link)
    return {
        "left_points": points["left"],
        "right_points": points["right"],
        "left_links": len(links["left"]),
        "right_links": len(links["right"]),
    }


def run_physics_chunks(args, left_q, right_q, active_hands, gravity_only):
    parts = []
    for start in range(0, len(left_q), args.batch_size):
        end = min(start + args.batch_size, len(left_q))
        chunk_dir = args.output_dir / f"isaac_{active_hands}" / f"{start:05d}_{end:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        run_args = SimpleNamespace(
            repo=args.repo,
            isaac_python=args.isaac_python,
            gpu=args.gpu,
            force=True,
            left_robot_name="allegro_left",
            right_robot_name="allegro_right",
            gravity=9.8,
            gravity_settle_step=500,
            staged_gravity=False,
            independent_directions=not gravity_only,
            active_hands=active_hands,
            gravity_only=gravity_only,
            support_during_closure=True,
            fixture_during_closure=False,
            lift_height=args.lift_height,
            lift_step=args.lift_step,
            min_lift_height=args.min_lift_height,
            robot_friction=args.robot_friction,
            object_friction=args.object_friction,
            finger_effort_limit=args.finger_effort_limit,
            contact_offset=args.contact_offset,
            max_gravity_displacement=args.max_gravity_displacement,
            max_direction_displacement=args.max_direction_displacement,
            object_density=args.object_density,
            object_vhacd=args.object_vhacd,
            object_vhacd_resolution=args.object_vhacd_resolution,
            object_vhacd_max_convex_hulls=args.object_vhacd_max_convex_hulls,
            object_vhacd_max_vertices=args.object_vhacd_max_vertices,
            object_multicollision=args.object_multicollision,
            object_vhacd_high_v1=args.object_vhacd_high_v1,
            object_vhacd_visual_high_v2=(
                args.object_vhacd_visual_high_v2
            ),
            capture_contacts=(active_hands == "both"),
        )
        parts.append(
            run_isaac(
                run_args,
                args.object_name,
                left_q[start:end],
                right_q[start:end],
                chunk_dir,
            )
        )
    return merge_results(parts)


def run_geometry_chunks(args, left_q, right_q):
    rows = []
    for start in range(0, len(left_q), args.geometry_batch_size):
        end = min(start + args.geometry_batch_size, len(left_q))
        chunk_dir = args.output_dir / "geometry" / f"{start:05d}_{end:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        left_path = chunk_dir / "left_q.pt"
        right_path = chunk_dir / "right_q.pt"
        result_path = chunk_dir / "geometry_result.pt"
        torch.save(left_q[start:end], left_path)
        torch.save(right_q[start:end], right_path)
        subprocess.run(
            [
                sys.executable,
                str(args.repo / "validation/bimanual_realized_geometry_main.py"),
                "--repo",
                str(args.repo),
                "--object-name",
                args.object_name,
                "--left-q-file",
                str(left_path),
                "--right-q-file",
                str(right_path),
                "--output-file",
                str(result_path),
                "--contact-mm",
                str(args.geometry_contact_mm),
                "--left-robot-name",
                "allegro_left",
                "--right-robot-name",
                "allegro_right",
            ],
            cwd=args.repo,
            check=True,
        )
        rows.extend(torch.load(result_path, map_location="cpu", weights_only=False))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(os.environ.get("ISAAC_PYTHON", "python")),
    )
    parser.add_argument("--additional-repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--geometry-batch-size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--object-name")
    parser.add_argument("--object-density", type=float, default=50.0)
    parser.add_argument("--object-vhacd", action="store_true")
    parser.add_argument("--object-vhacd-resolution", type=int, default=300000)
    parser.add_argument(
        "--object-vhacd-max-convex-hulls", type=int, default=64
    )
    parser.add_argument("--object-vhacd-max-vertices", type=int, default=64)
    parser.add_argument("--object-multicollision", action="store_true")
    parser.add_argument("--object-vhacd-high-v1", action="store_true")
    parser.add_argument("--object-vhacd-visual-high-v2", action="store_true")
    parser.add_argument("--finger-effort-limit", type=float, default=0.7)
    parser.add_argument("--lift-height", type=float, default=0.05)
    parser.add_argument("--lift-step", type=int, default=100)
    parser.add_argument("--min-lift-height", type=float, default=0.03)
    parser.add_argument("--robot-friction", type=float, default=1.0)
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument("--contact-offset", type=float, default=0.002)
    parser.add_argument("--max-gravity-displacement", type=float, default=0.01)
    parser.add_argument("--max-direction-displacement", type=float, default=0.015)
    parser.add_argument("--min-contact-links", type=int, default=3)
    parser.add_argument(
        "--geometry-contact-mm",
        type=float,
        default=2.0,
        help=(
            "Point-cloud proximity band used only for the realized geometry "
            "audit. PhysX positive-impulse contact remains a separate hard gate."
        ),
    )
    parser.add_argument(
        "--require-local-quality",
        action="store_true",
        help="Make the optional static local force-closure audit a hard gate.",
    )
    args = parser.parse_args()
    if args.object_vhacd_high_v1 and not args.object_vhacd:
        parser.error("--object-vhacd-high-v1 requires --object-vhacd")
    if args.object_vhacd_visual_high_v2 and not args.object_vhacd:
        parser.error("--object-vhacd-visual-high-v2 requires --object-vhacd")
    if args.object_vhacd_high_v1 and args.object_vhacd_visual_high_v2:
        parser.error("Choose exactly one versioned V-HACD asset")
    if args.object_vhacd_high_v1 and args.object_multicollision:
        parser.error(
            "--object-vhacd-high-v1 and --object-multicollision are mutually exclusive"
        )
    if args.object_vhacd_visual_high_v2 and args.object_multicollision:
        parser.error(
            "--object-vhacd-visual-high-v2 and --object-multicollision are "
            "mutually exclusive"
        )
    if args.object_vhacd_high_v1 and (
        args.object_vhacd_resolution != 1_000_000
        or args.object_vhacd_max_convex_hulls != 128
        or args.object_vhacd_max_vertices != 64
    ):
        parser.error(
            "high-v1 requires resolution=1000000, max-convex-hulls=128, "
            "max-vertices=64"
        )
    if args.object_vhacd_visual_high_v2 and (
        args.object_vhacd_resolution != 1_000_000
        or args.object_vhacd_max_convex_hulls != 128
        or args.object_vhacd_max_vertices != 64
    ):
        parser.error(
            "visual-high-v2 requires resolution=1000000, "
            "max-convex-hulls=128, max-vertices=64"
        )
    args.repo = args.repo.resolve()
    args.source_dataset = args.source_dataset.resolve()
    if args.quality_json is not None:
        args.quality_json = args.quality_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.source_dataset, map_location="cpu", weights_only=False)
    if args.object_name is None:
        object_names = {sample["object_name"] for sample in payload["samples"]}
        if len(object_names) != 1:
            raise ValueError("--object-name is required for multi-object input")
        args.object_name = next(iter(object_names))
    if args.quality_json is None:
        selected_indices = list(range(len(payload["samples"])))
    else:
        quality = json.loads(args.quality_json.read_text())
        selected_indices = [
            row["sample_index"]
            for row in quality["samples"]
            if row["decoupled_force_closure_pass"]
        ]
    selected = [payload["samples"][index] for index in selected_indices]
    if not selected:
        raise RuntimeError("No locally balanced samples were selected")

    repeat_sources = []
    left_seed = []
    right_seed = []
    for source_index, sample in zip(selected_indices, selected):
        for repeat in range(args.additional_repeats):
            repeat_sources.append((source_index, repeat + 2))
            left_seed.append(sample["left_q_seed"].clone())
            right_seed.append(sample["right_q_seed"].clone())
    left_seed = torch.stack(left_seed)
    right_seed = torch.stack(right_seed)
    torch.save(left_seed, args.output_dir / "left_q_seed.pt")
    torch.save(right_seed, args.output_dir / "right_q_seed.pt")

    both = run_physics_chunks(args, left_seed, right_seed, "both", False)
    left = run_physics_chunks(args, left_seed, right_seed, "left", True)
    right = run_physics_chunks(args, left_seed, right_seed, "right", True)
    torch.save(both, args.output_dir / "both_result.pt")
    torch.save(left, args.output_dir / "left_result.pt")
    torch.save(right, args.output_dir / "right_result.pt")

    geometry = run_geometry_chunks(args, both["left_q_final"], both["right_q_final"])
    torch.save(geometry, args.output_dir / "geometry_result.pt")
    repeat_samples = []
    for index, (source_index, repeat) in enumerate(repeat_sources):
        source = payload["samples"][source_index]
        repeat_sample = dict(source)
        repeat_sample["left_q"] = both["left_q_final"][index].clone()
        repeat_sample["right_q"] = both["right_q_final"][index].clone()
        repeat_sample["repeat_source_index"] = source_index
        repeat_sample["repeat_number"] = repeat
        repeat_samples.append(repeat_sample)
    repeat_dataset_path = args.output_dir / "repeat_final_samples.pt"
    torch.save({"manifest": payload.get("manifest", {}), "samples": repeat_samples}, repeat_dataset_path)
    local_path = args.output_dir / "repeat_local_quality.json"
    subprocess.run(
        [
            sys.executable,
            str(args.repo / "scripts/audit_decoupled_force_closure.py"),
            "--dataset",
            str(repeat_dataset_path),
            "--output",
            str(local_path),
            "--friction",
            "1",
            "--contact-mm",
            str(args.geometry_contact_mm),
            "--points-per-link",
            "8",
            "--min-contact-links",
            str(args.min_contact_links),
            "--max-wrench-residual",
            "0.35",
            "--residual-statistic",
            "mean",
        ],
        cwd=args.repo,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    repeat_quality = json.loads(local_path.read_text())

    rows = []
    pass_by_source = {index: True for index in selected_indices}
    local_pass_by_source = {index: True for index in selected_indices}
    for index, (source_index, repeat) in enumerate(repeat_sources):
        geom = geometry[index]
        contact_by_phase = {
            "closure": active_physical_contacts(
                both["closure_contacts"][index]
            ),
            "lifted": active_physical_contacts(
                both["lifted_contacts"][index]
            ),
            "settled": active_physical_contacts(
                both["settled_contacts"][index]
            ),
        }
        bilateral_physical_contact = all(
            counts["left_points"] >= 1 and counts["right_points"] >= 1
            for counts in contact_by_phase.values()
        )
        local_pass = bool(repeat_quality["samples"][index]["decoupled_force_closure_pass"])
        local_pass_by_source[source_index] &= local_pass
        passed = bool(
            both["success"][index]
            and not left["gravity_success"][index]
            and not right["gravity_success"][index]
            and geom["left"]["penetration_depth_mm"] <= 2.0
            and geom["right"]["penetration_depth_mm"] <= 2.0
            and geom["left"]["contact_link_count"] >= args.min_contact_links
            and geom["right"]["contact_link_count"] >= args.min_contact_links
            and geom["clearance_mm"] > 2.0
            and bilateral_physical_contact
            and (local_pass or not args.require_local_quality)
        )
        pass_by_source[source_index] &= passed
        row = {
                "source_sample_index": source_index,
                "repeat": repeat,
                "passed": passed,
                "lift_mm": float(both["lift_displacement_z"][index] * 1000),
                "gravity_mm": float(both["gravity_displacement"][index] * 1000),
                "max_6dir_mm": float(both["max_direction_displacement"][index] * 1000),
                "left_only_gravity_success": bool(left["gravity_success"][index]),
                "right_only_gravity_success": bool(right["gravity_success"][index]),
                "left_links": geom["left"]["contact_link_count"],
                "right_links": geom["right"]["contact_link_count"],
                "left_penetration_mm": geom["left"]["penetration_depth_mm"],
                "right_penetration_mm": geom["right"]["penetration_depth_mm"],
                "hand_clearance_mm": geom["clearance_mm"],
                "local_quality_pass": local_pass,
                "bilateral_physical_contact_all_phases": (
                    bilateral_physical_contact
                ),
            }
        for phase, counts in contact_by_phase.items():
            for name, value in counts.items():
                row[f"{phase}_active_{name}"] = value
        rows.append(row)

    verified = []
    for source_index, sample in zip(selected_indices, selected):
        if not pass_by_source[source_index]:
            continue
        source_repeat_rows = [
            row for row in rows if row["source_sample_index"] == source_index
        ]
        source_metrics = sample["metrics"]
        all_lift = [float(source_metrics["lift_displacement_mm"])] + [
            row["lift_mm"] for row in source_repeat_rows
        ]
        all_gravity = [float(source_metrics["gravity_displacement_mm"])] + [
            row["gravity_mm"] for row in source_repeat_rows
        ]
        all_six_direction = [
            float(source_metrics["max_direction_displacement_mm"])
        ] + [row["max_6dir_mm"] for row in source_repeat_rows]
        all_left_links = [
            int(source_metrics["left_realized_contact_link_count"])
        ] + [int(row["left_links"]) for row in source_repeat_rows]
        all_right_links = [
            int(source_metrics["right_realized_contact_link_count"])
        ] + [int(row["right_links"]) for row in source_repeat_rows]
        all_left_penetration = [
            float(source_metrics["left_realized_penetration_mm"])
        ] + [float(row["left_penetration_mm"]) for row in source_repeat_rows]
        all_right_penetration = [
            float(source_metrics["right_realized_penetration_mm"])
        ] + [float(row["right_penetration_mm"]) for row in source_repeat_rows]
        all_clearance = [
            float(source_metrics["realized_hand_clearance_mm"])
        ] + [float(row["hand_clearance_mm"]) for row in source_repeat_rows]
        any_left_only_success = bool(
            source_metrics["left_only_gravity_success"]
        ) or any(row["left_only_gravity_success"] for row in source_repeat_rows)
        any_right_only_success = bool(
            source_metrics["right_only_gravity_success"]
        ) or any(row["right_only_gravity_success"] for row in source_repeat_rows)
        minimum_physical_contacts = {}
        for phase in ("closure", "lifted", "settled"):
            for side in ("left", "right"):
                for unit in ("points", "links"):
                    key = f"{phase}_active_{side}_{unit}"
                    minimum_physical_contacts[
                        f"repeat_min_{key}"
                    ] = min(int(row[key]) for row in source_repeat_rows)
        verified_sample = dict(sample)
        verified_sample["metrics"] = dict(sample["metrics"])
        verified_sample["metrics"].update(
            {
                # This audit is diagnostic unless --require-local-quality is
                # supplied.  Preserve the measured value instead of claiming
                # that an optional gate passed.
                "local_cooperative_quality_pass": local_pass_by_source[
                    source_index
                ],
                "repeat_count": 1 + args.additional_repeats,
                "repeat_success_rate": 1.0,
                "repeat_min_lift_mm": min(all_lift),
                "repeat_max_gravity_mm": max(all_gravity),
                "repeat_max_6dir_mm": max(all_six_direction),
                "repeat_min_left_contact_links": min(all_left_links),
                "repeat_min_right_contact_links": min(all_right_links),
                "repeat_max_left_penetration_mm": max(all_left_penetration),
                "repeat_max_right_penetration_mm": max(all_right_penetration),
                "repeat_min_hand_clearance_mm": min(all_clearance),
                "repeat_any_left_only_gravity_success": any_left_only_success,
                "repeat_any_right_only_gravity_success": any_right_only_success,
                "physical_contact_repeat_count": args.additional_repeats,
                "repeat_bilateral_physical_contact_all_phases": all(
                    row["bilateral_physical_contact_all_phases"]
                    for row in source_repeat_rows
                ),
                **minimum_physical_contacts,
            }
        )
        verified_sample["provenance"] = {
            **sample.get("provenance", {}),
            "source_dataset": str(args.source_dataset),
            "source_sample_index": source_index,
            "repeat_verification_dir": str(args.output_dir),
            "repeat_result_indices": [
                index
                for index, (repeat_source, _) in enumerate(repeat_sources)
                if repeat_source == source_index
            ],
        }
        verified.append(verified_sample)

    manifest = dict(payload.get("manifest", {}))
    manifest.update(
        {
            "schema": "tro_grasp_tabletop_bimanual_repeat_verified_v2",
            "repeat_count": 1 + args.additional_repeats,
            "num_local_quality_candidates": len(selected),
            "num_repeat_verified_samples": len(verified),
            "min_contact_links_per_hand": args.min_contact_links,
            "object_name": args.object_name,
            "object_density_kg_m3": args.object_density,
            "object_vhacd": args.object_vhacd,
            "object_vhacd_resolution": args.object_vhacd_resolution,
            "object_vhacd_max_convex_hulls": (
                args.object_vhacd_max_convex_hulls
            ),
            "object_vhacd_max_vertices": args.object_vhacd_max_vertices,
            "object_multicollision": args.object_multicollision,
            "object_vhacd_high_v1": args.object_vhacd_high_v1,
            "object_vhacd_visual_high_v2": (
                args.object_vhacd_visual_high_v2
            ),
            "object_collision_mode": (
                "vhacd_visual_high_v2"
                if args.object_vhacd_visual_high_v2
                else (
                    "vhacd_high_v1"
                    if args.object_vhacd_high_v1
                    else (
                        "coacd_multicollision_v1"
                        if args.object_multicollision
                        else (
                            "vhacd"
                            if args.object_vhacd
                            else "legacy_single_hull"
                        )
                    )
                )
            ),
            "lift_command_height_m": args.lift_height,
            "lift_step": args.lift_step,
            "minimum_lift_height_m": args.min_lift_height,
            "robot_friction": args.robot_friction,
            "object_friction": args.object_friction,
            "contact_offset_m": args.contact_offset,
            "max_gravity_displacement_m": args.max_gravity_displacement,
            "max_direction_displacement_m": args.max_direction_displacement,
            "max_mean_local_wrench_residual": 0.35,
            "local_quality_hard_gate": args.require_local_quality,
            "physical_contact_capture": True,
            "geometry_contact_band_mm": args.geometry_contact_mm,
            "physical_contact_hard_gate": (
                "positive normal impulse from both hands at closure, after lift, "
                "and after gravity settling in every additional rollout"
            ),
        }
    )
    torch.save(
        {"manifest": manifest, "samples": verified},
        args.output_dir / "verified_dataset.pt",
    )
    with (args.output_dir / "repeat_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_selected": len(selected),
        "additional_repeats": args.additional_repeats,
        "num_repeat_verified": len(verified),
        "verified_source_indices": [
            index for index in selected_indices if pass_by_source[index]
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
