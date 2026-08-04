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
        else:
            merged[key] = values
    return merged


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
            lift_height=0.05,
            lift_step=100,
            min_lift_height=0.03,
            robot_friction=1.0,
            object_friction=1.0,
            finger_effort_limit=0.7,
            contact_offset=0.002,
            max_gravity_displacement=0.01,
            max_direction_displacement=0.015,
            object_density=50.0,
        )
        parts.append(
            run_isaac(
                run_args,
                "contactdb+cylinder_xlarge",
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
                "contactdb+cylinder_xlarge",
                "--left-q-file",
                str(left_path),
                "--right-q-file",
                str(right_path),
                "--output-file",
                str(result_path),
                "--contact-mm",
                "2",
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
    parser.add_argument("--quality-json", type=Path, required=True)
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
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.source_dataset = args.source_dataset.resolve()
    args.quality_json = args.quality_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.source_dataset, map_location="cpu", weights_only=False)
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
            "2",
            "--points-per-link",
            "8",
            "--min-contact-links",
            "3",
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
    for index, (source_index, repeat) in enumerate(repeat_sources):
        geom = geometry[index]
        local_pass = bool(repeat_quality["samples"][index]["decoupled_force_closure_pass"])
        passed = bool(
            both["success"][index]
            and not left["gravity_success"][index]
            and not right["gravity_success"][index]
            and geom["left"]["penetration_depth_mm"] <= 2.0
            and geom["right"]["penetration_depth_mm"] <= 2.0
            and geom["left"]["contact_link_count"] >= 3
            and geom["right"]["contact_link_count"] >= 3
            and geom["clearance_mm"] > 2.0
            and local_pass
        )
        pass_by_source[source_index] &= passed
        rows.append(
            {
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
                "local_quality_pass": local_pass,
            }
        )

    verified = []
    for source_index, sample in zip(selected_indices, selected):
        if not pass_by_source[source_index]:
            continue
        verified_sample = dict(sample)
        verified_sample["metrics"] = dict(sample["metrics"])
        verified_sample["metrics"].update(
            {
                "local_cooperative_quality_pass": True,
                "repeat_count": 1 + args.additional_repeats,
                "repeat_success_rate": 1.0,
            }
        )
        verified_sample["provenance"] = {
            "source_dataset": str(args.source_dataset),
            "source_sample_index": source_index,
        }
        verified.append(verified_sample)

    manifest = dict(payload.get("manifest", {}))
    manifest.update(
        {
            "schema": "tro_grasp_xlarge_tabletop_bimanual_batch_v1",
            "repeat_count": 1 + args.additional_repeats,
            "num_local_quality_candidates": len(selected),
            "num_repeat_verified_samples": len(verified),
            "min_contact_links_per_hand": 3,
            "max_mean_local_wrench_residual": 0.35,
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
