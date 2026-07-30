"""Recompute mesh-based penetration metrics from saved evaluation vis.pt files."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_depth


METHOD = "mesh_point_containment"


def update_result_file(path, pass_rate, depths, joint_success_rate, threshold_mm):
    lines = path.read_text().splitlines()
    prefixes = (
        "Penetration method:",
        "Penetration pass rate:",
        "Penetration depth (mm):",
        "Joint valid-and-stable success rate:",
    )
    lines = [line for line in lines if not line.startswith(prefixes)]
    lines.extend(
        [
            f"Penetration method: {METHOD}.",
            (
                f"Penetration pass rate: {pass_rate:.6f} "
                f"(threshold={threshold_mm:.2f} mm)."
            ),
            (
                "Penetration depth (mm): "
                f"mean={depths.mean():.4f}, "
                f"p95={np.percentile(depths, 95):.4f}, "
                f"max={depths.max():.4f}."
            ),
            (
                "Joint valid-and-stable success rate: "
                f"{joint_success_rate:.6f}."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def evaluate_vis(vis_path, threshold_mm):
    entries = torch.load(vis_path, map_location="cpu")
    hands = {}
    sample_rows = []
    global_index = 0

    for entry in entries:
        robot_name = entry["robot_name"]
        if robot_name not in hands:
            hands[robot_name] = create_hand_model(
                robot_name,
                device=torch.device("cpu"),
            )
        passes, depths = validate_depth(
            hands[robot_name],
            entry["object_name"],
            entry["predict_q"],
            threshold=threshold_mm / 1000,
            exact=True,
        )
        for local_index, (passed, depth) in enumerate(zip(passes, depths)):
            rollout_success = bool(entry["success"][local_index])
            sample_rows.append(
                {
                    "global_index": global_index,
                    "object_name": entry["object_name"],
                    "object_local_index": local_index,
                    "rollout_success": rollout_success,
                    "penetration_pass": bool(passed),
                    "penetration_depth_mm": depth,
                    "joint_valid_and_stable": bool(passed)
                    and rollout_success,
                }
            )
            global_index += 1

    passes = np.asarray(
        [row["penetration_pass"] for row in sample_rows],
        dtype=bool,
    )
    depths = np.asarray(
        [row["penetration_depth_mm"] for row in sample_rows],
        dtype=float,
    )
    joint = np.asarray(
        [row["joint_valid_and_stable"] for row in sample_rows],
        dtype=bool,
    )
    return sample_rows, passes.mean(), depths, joint.mean()


def evaluate_summary_row(row, threshold_mm):
    result_path = Path(row["result_path"])
    vis_path = result_path.with_name("vis.pt")
    sample_rows, pass_rate, depths, joint_success_rate = evaluate_vis(
        vis_path,
        threshold_mm,
    )
    return (
        row,
        sample_rows,
        pass_rate,
        depths,
        joint_success_rate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "graph_exp/reproduction/train-multi-hand-rtx3070"
        ),
    )
    parser.add_argument("--threshold-mm", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "evaluation_summary.csv"
    with summary_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
        original_fields = list(rows[0].keys())

    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_summary_row,
                row,
                args.threshold_mm,
            ): (row["epoch"], row["evaluation"])
            for row in rows
        }
        for future in as_completed(futures):
            epoch, evaluation = futures[future]
            result = future.result()
            completed.append(result)
            print(
                f"completed epoch={epoch} mode={evaluation}",
                flush=True,
            )

    for row, sample_rows, pass_rate, depths, joint_success_rate in completed:
        result_path = Path(row["result_path"])
        sample_path = result_path.with_name("penetration_mesh.csv")
        with sample_path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(sample_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(sample_rows)

        row["penetration_pass_rate"] = f"{pass_rate:.6f}"
        row["penetration_mean_mm"] = f"{depths.mean():.4f}"
        row["penetration_p95_mm"] = (
            f"{np.percentile(depths, 95):.4f}"
        )
        row["penetration_max_mm"] = f"{depths.max():.4f}"
        row["penetration_method"] = METHOD
        row["joint_success_rate"] = f"{joint_success_rate:.6f}"
        update_result_file(
            result_path,
            pass_rate,
            depths,
            joint_success_rate,
            args.threshold_mm,
        )

    rows = [result[0] for result in completed]
    rows.sort(key=lambda row: (int(row["epoch"]), row["evaluation"]))
    fieldnames = [
        field
        for field in original_fields
        if field not in {"penetration_method", "joint_success_rate"}
    ]
    result_index = fieldnames.index("result_path")
    fieldnames[result_index:result_index] = [
        "penetration_method",
        "joint_success_rate",
    ]
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field, "") for field in fieldnames} for row in rows]
        )


if __name__ == "__main__":
    main()
