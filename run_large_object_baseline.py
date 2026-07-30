"""Run and summarize official-checkpoint single-hand baselines on large objects."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def run_hand(args, hand, objects):
    save_dir = args.output_dir / f"{hand}_unconditioned"
    result_path = save_dir / "res.txt"
    if result_path.is_file() and (save_dir / "vis.pt").is_file():
        print(f"Reusing {result_path}", flush=True)
        return save_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "evaluate_checkpoint.py",
        "--base-config",
        str(args.base_config),
        "--embodiment",
        hand,
        "--checkpoint",
        str(args.checkpoint),
        "--save-dir",
        str(save_dir),
        "--batch-size",
        str(args.batch_size),
        "--split-batch-size",
        str(args.split_batch_size),
        "--seed",
        str(args.seed),
        "--objects",
        *objects,
        "--penetration-exact",
    ]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    with (save_dir / "evaluation.log").open("w") as log:
        subprocess.run(
            command,
            cwd=args.repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return save_dir


def summarize_hand(hand, save_dir, size_lookup):
    entries = torch.load(save_dir / "vis.pt", map_location="cpu")
    rows = []
    global_index = 0
    for entry in entries:
        penetration_passes = entry.get("penetration_success")
        penetration_depths = entry.get("penetration_depth_mm")
        if penetration_passes is None or penetration_depths is None:
            raise RuntimeError(
                f"{save_dir / 'vis.pt'} lacks penetration data"
            )
        for local_index in range(entry["predict_q"].shape[0]):
            rollout_success = bool(entry["success"][local_index])
            penetration_pass = bool(penetration_passes[local_index])
            rows.append(
                {
                    "hand": hand,
                    "object_name": entry["object_name"],
                    "global_index": global_index,
                    "object_local_index": local_index,
                    "max_dimension_mm": size_lookup[
                        entry["object_name"]
                    ]["max_dimension_mm"],
                    "rollout_success": rollout_success,
                    "penetration_pass": penetration_pass,
                    "penetration_depth_mm": penetration_depths[local_index],
                    "joint_valid_and_stable": rollout_success
                    and penetration_pass,
                }
            )
            global_index += 1
    return rows


def aggregate(sample_rows):
    groups = {}
    for row in sample_rows:
        key = (row["hand"], row["object_name"])
        groups.setdefault(key, []).append(row)
    rows = []
    for (hand, object_name), samples in sorted(groups.items()):
        rows.append(
            {
                "hand": hand,
                "object_name": object_name,
                "num_grasps": len(samples),
                "max_dimension_mm": samples[0]["max_dimension_mm"],
                "rollout_success_rate": np.mean(
                    [row["rollout_success"] for row in samples]
                ),
                "penetration_pass_rate": np.mean(
                    [row["penetration_pass"] for row in samples]
                ),
                "joint_success_rate": np.mean(
                    [row["joint_valid_and_stable"] for row in samples]
                ),
                "penetration_mean_mm": np.mean(
                    [row["penetration_depth_mm"] for row in samples]
                ),
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, aggregate_rows, object_split):
    lines = [
        "# Official-checkpoint single-hand baseline on large objects",
        "",
        "Both AllegroHand and ShadowHand use unconditioned inference, 20 grasps",
        "per object, exact object-mesh penetration screening at 5 mm, and the",
        "Isaac six-direction disturbance rollout.",
        "",
        "These objects belong to the repository's original training split. This",
        "is a development baseline for the bimanual extension, not an unseen-",
        "object generalization benchmark. The three held-out labels below mean",
        "held out from future *bimanual* data construction.",
        "",
        "| Hand | Object | Role | Rollout | Mesh pass | Joint |",
        "|---|---|---|---:|---:|---:|",
    ]
    role_lookup = {
        object_name: role
        for role, object_names in object_split.items()
        for object_name in object_names
    }
    for row in aggregate_rows:
        lines.append(
            f"| {row['hand']} | `{row['object_name']}` | "
            f"{role_lookup[row['object_name']]} | "
            f"{row['rollout_success_rate']:.1%} | "
            f"{row['penetration_pass_rate']:.1%} | "
            f"{row['joint_success_rate']:.1%} |"
        )
    lines.extend(["", "## Overall", ""])
    for hand in ("allegro", "shadowhand"):
        hand_rows = [row for row in aggregate_rows if row["hand"] == hand]
        lines.append(
            f"- {hand}: rollout "
            f"{np.mean([row['rollout_success_rate'] for row in hand_rows]):.1%}, "
            f"mesh pass "
            f"{np.mean([row['penetration_pass_rate'] for row in hand_rows]):.1%}, "
            f"joint "
            f"{np.mean([row['joint_success_rate'] for row in hand_rows]):.1%}."
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--object-split",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/object_selection/"
            "bimanual_object_split.json"
        ),
    )
    parser.add_argument(
        "--object-sizes",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/object_selection/"
            "object_sizes.csv"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/test_palm_unconditioned_rtx3070.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ckpt/multi_hand.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/single_hand_baseline"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--split-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for name in (
        "object_split",
        "object_sizes",
        "base_config",
        "checkpoint",
        "output_dir",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (args.repo / value).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    object_split = json.loads(args.object_split.read_text())
    objects = object_split["development"] + object_split["held_out_test"]
    with args.object_sizes.open(newline="") as file:
        size_lookup = {
            row["object_name"]: row for row in csv.DictReader(file)
        }

    sample_rows = []
    for hand in ("allegro", "shadowhand"):
        save_dir = run_hand(args, hand, objects)
        sample_rows.extend(summarize_hand(hand, save_dir, size_lookup))

    aggregate_rows = aggregate(sample_rows)
    write_csv(args.output_dir / "sample_results.csv", sample_rows)
    write_csv(args.output_dir / "object_summary.csv", aggregate_rows)
    write_report(
        args.output_dir / "REPORT.md",
        aggregate_rows,
        object_split,
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
