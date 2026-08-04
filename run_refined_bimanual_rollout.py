#!/usr/bin/env python3
"""Run Isaac rollout for cached geometry-refined bimanual candidates."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import trimesh

from generate_bimanual_pilot import (
    DIRECTION_NAMES,
    joint_limit_pass,
)
from run_bimanual_pairing_baseline import run_isaac
from utils.hand_model import create_hand_model
from validation.bimanual_realized_geometry_main import (
    exact_metrics,
    transformed_points,
)


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_realized_worker(args, repo, object_dir, rollout_dir):
    """Measure one final pose, then exit to release mesh-query memory."""
    isaac = torch.load(
        rollout_dir / "isaac_result.pt",
        map_location="cpu",
        weights_only=False,
    )
    index = args.realized_worker_index
    dataset, name = args.object_name.split("+")
    mesh = trimesh.load_mesh(
        repo
        / "data/data_urdf/object"
        / dataset
        / name
        / f"{name}.stl"
    )
    hand = create_hand_model("allegro", torch.device("cpu"))
    query = trimesh.proximity.ProximityQuery(mesh)
    left_points = transformed_points(hand, isaac["left_q_final"][index])
    right_points = transformed_points(hand, isaac["right_q_final"][index])
    metrics = {
        "left": exact_metrics(
            mesh,
            query,
            left_points,
            args.contact_mm / 1000.0,
        ),
        "right": exact_metrics(
            mesh,
            query,
            right_points,
            args.contact_mm / 1000.0,
        ),
        "clearance_mm": float(
            torch.cdist(left_points, right_points).min() * 1000
        ),
    }
    realized_pass = (
        joint_limit_pass(hand, isaac["left_q_final"][index])
        and joint_limit_pass(hand, isaac["right_q_final"][index])
        and metrics["left"]["penetration_depth_mm"]
        <= args.penetration_mm
        and metrics["right"]["penetration_depth_mm"]
        <= args.penetration_mm
        and metrics["clearance_mm"] > args.clearance_mm
    )
    result = {
        "left_realized_penetration_mm": metrics["left"][
            "penetration_depth_mm"
        ],
        "right_realized_penetration_mm": metrics["right"][
            "penetration_depth_mm"
        ],
        "realized_hand_clearance_mm": metrics["clearance_mm"],
        "realized_pose_pass": bool(realized_pass),
        "audit_error": "",
    }
    output = rollout_dir / "realized_workers" / f"{index:03d}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"realized worker {index} complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(os.environ.get("ISAAC_PYTHON", "python")),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--penetration-mm", type=float, default=5.0)
    parser.add_argument("--contact-mm", type=float, default=5.0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument(
        "--geometry-batch-size",
        type=int,
        default=1,
        help="Small batches bound exact-mesh realized-pose memory use.",
    )
    parser.add_argument("--realized-worker-index", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    args.evaluation_dir = args.evaluation_dir.resolve()
    object_dir = (
        args.evaluation_dir
        / args.object_name.replace("+", "__")
    )
    rollout_dir = object_dir / "isaac_refined"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    if args.realized_worker_index is not None:
        run_realized_worker(args, repo, object_dir, rollout_dir)
        return
    generated_path = object_dir / "generated_q.pt"
    generated = torch.load(
        generated_path,
        map_location="cpu",
        weights_only=False,
    )
    indices = list(map(int, generated["geometry_indices"]))
    if not indices:
        raise RuntimeError("No geometry-valid candidates to roll out")
    selected = torch.as_tensor(indices, dtype=torch.long)
    left_q = generated["left_q_seed"][selected]
    right_q = generated["right_q_seed"][selected]
    isaac_result_path = rollout_dir / "isaac_result.pt"
    cache_is_current = (
        isaac_result_path.is_file()
        and isaac_result_path.stat().st_mtime
        >= generated_path.stat().st_mtime
    )
    isaac = run_isaac(
        SimpleNamespace(
            repo=repo,
            isaac_python=args.isaac_python,
            gpu=args.gpu,
            force=args.force or not cache_is_current,
        ),
        args.object_name,
        left_q,
        right_q,
        rollout_dir,
    )

    worker_dir = rollout_dir / "realized_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    realized = []
    for local_index in range(len(indices)):
        worker_output = worker_dir / f"{local_index:03d}.json"
        cache_is_current = (
            worker_output.is_file()
            and worker_output.stat().st_mtime
            >= isaac_result_path.stat().st_mtime
        )
        if not cache_is_current:
            command = [
                sys.executable,
                Path(__file__).resolve(),
                "--evaluation-dir",
                args.evaluation_dir,
                "--object-name",
                args.object_name,
                "--penetration-mm",
                str(args.penetration_mm),
                "--contact-mm",
                str(args.contact_mm),
                "--clearance-mm",
                str(args.clearance_mm),
                "--realized-worker-index",
                str(local_index),
            ]
            try:
                subprocess.run(command, cwd=repo, check=True)
            except subprocess.CalledProcessError as error:
                fallback = {
                    "left_realized_penetration_mm": None,
                    "right_realized_penetration_mm": None,
                    "realized_hand_clearance_mm": None,
                    "realized_pose_pass": False,
                    "audit_error": (
                        "realized_geometry_worker_failed_"
                        f"returncode_{error.returncode}"
                    ),
                }
                worker_output.write_text(
                    json.dumps(fallback, indent=2) + "\n"
                )
                print(
                    f"realized worker {local_index} failed; "
                    "conservatively rejecting sample",
                    flush=True,
                )
        realized.append(json.loads(worker_output.read_text()))
    rows = read_csv(args.evaluation_dir / "sample_results.csv")
    strict_count = 0
    for local_index, sample_index in enumerate(indices):
        metrics = realized[local_index]
        realized_pass = metrics["realized_pose_pass"]
        isaac_success = bool(isaac["success"][local_index])
        strict_success = isaac_success and realized_pass
        strict_count += int(strict_success)
        rows[sample_index].update(
            {
                "isaac_evaluated": True,
                "isaac_success": isaac_success,
                "settle_displacement_mm": float(
                    isaac["settle_displacement"][local_index] * 1000
                ),
                "max_direction_displacement_mm": float(
                    isaac["max_direction_displacement"][local_index] * 1000
                ),
                "left_realized_penetration_mm": metrics[
                    "left_realized_penetration_mm"
                ],
                "right_realized_penetration_mm": metrics[
                    "right_realized_penetration_mm"
                ],
                "realized_hand_clearance_mm": metrics[
                    "realized_hand_clearance_mm"
                ],
                "realized_pose_pass": realized_pass,
                "realized_audit_error": metrics.get("audit_error", ""),
                "strict_success": strict_success,
            }
        )
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            rows[sample_index][f"displacement_{direction}_mm"] = float(
                isaac["direction_displacements"][
                    local_index,
                    direction_index,
                ]
                * 1000
            )
    output_csv = args.evaluation_dir / "refined_rollout_results.csv"
    write_csv(output_csv, rows)
    summary = {
        "object_name": args.object_name,
        "total_generated": len(rows),
        "geometry_valid": len(indices),
        "isaac_success": int(
            sum(
                row.get("isaac_success") in (True, "True")
                for row in rows
            )
        ),
        "strict_success": strict_count,
        "strict_rate_all": strict_count / len(rows),
        "strict_rate_geometry_valid": strict_count / len(indices),
    }
    (args.evaluation_dir / "refined_rollout_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
