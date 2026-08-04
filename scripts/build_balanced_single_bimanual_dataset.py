#!/usr/bin/env python3
"""Build object-matched single/bimanual corpora at an exact count ratio.

The script never duplicates a grasp to satisfy the ratio.  Surplus bimanual
samples are kept as held-out samples, and the smaller single-hand subset is
drawn only from strict Allegro rollout/penetration/joint-valid results for the
same object.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def semantic_key(sample):
    return (
        sample["object_name"],
        int(sample.get("left_source_index", -1)),
        int(sample.get("right_source_index", -1)),
        round(float(sample.get("opposition_roll_degrees", 0.0)), 6),
    )


def quality_key(sample):
    metrics = sample.get("metrics", {})
    return (
        float(metrics.get("max_direction_displacement_mm", float("inf"))),
        float(metrics.get("gravity_displacement_mm", float("inf"))),
        -int(metrics.get("left_realized_contact_link_count", 0)),
        -int(metrics.get("right_realized_contact_link_count", 0)),
    )


def load_bimanual(paths):
    by_key = {}
    provenance = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"] if isinstance(payload, dict) else payload
        for sample in samples:
            if not bool(sample.get("metrics", {}).get("strict_success", True)):
                continue
            key = semantic_key(sample)
            if key not in by_key or quality_key(sample) < quality_key(by_key[key]):
                by_key[key] = sample
                provenance[key] = str(path)
    result = []
    for key, sample in by_key.items():
        copied = dict(sample)
        copied["source_dataset"] = provenance[key]
        result.append(copied)
    return result


def load_valid_single(vis_path, results_path):
    entries = torch.load(vis_path, map_location="cpu", weights_only=False)
    entry_by_object = {entry["object_name"]: entry for entry in entries}
    valid_rows = defaultdict(list)
    with results_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["hand"] != "allegro":
                continue
            valid = all(
                row[name].strip().lower() == "true"
                for name in (
                    "rollout_success",
                    "penetration_pass",
                    "joint_valid_and_stable",
                )
            )
            if valid:
                valid_rows[row["object_name"]].append(row)

    result = defaultdict(list)
    for object_name, rows in valid_rows.items():
        entry = entry_by_object[object_name]
        for row in rows:
            index = int(row["object_local_index"])
            result[object_name].append(
                {
                    "object_name": object_name,
                    "robot_name": "allegro",
                    "object_local_index": index,
                    "global_index": int(row["global_index"]),
                    "q": entry["isaac_q"][index].clone(),
                    "predict_q": entry["predict_q"][index].clone(),
                    "initial_q": entry["initial_q"][index].clone(),
                    "object_pc": entry["object_pc"][index].clone(),
                    "metrics": {
                        "rollout_success": True,
                        "penetration_pass": True,
                        "joint_valid_and_stable": True,
                        "penetration_depth_mm": float(
                            row["penetration_depth_mm"]
                        ),
                    },
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bimanual-datasets",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--single-vis",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/single_hand_baseline/"
            "allegro_unconditioned/vis.pt"
        ),
    )
    parser.add_argument(
        "--single-results",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/single_hand_baseline/"
            "sample_results.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bimanual-per-single", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--preserve-all-single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep every strict single-hand sample and report the bimanual "
            "deficit until the requested ratio is reached."
        ),
    )
    parser.add_argument(
        "--verification-level",
        default="strict_physics_and_geometry",
    )
    args = parser.parse_args()
    if args.bimanual_per_single < 1:
        parser.error("--bimanual-per-single must be positive")

    rng = random.Random(args.seed)
    bimanual_by_object = defaultdict(list)
    for sample in load_bimanual(args.bimanual_datasets):
        bimanual_by_object[sample["object_name"]].append(sample)
    single_by_object = load_valid_single(
        args.single_vis,
        args.single_results,
    )

    selected_bimanual = []
    held_out_bimanual = []
    selected_single = []
    rows = []
    all_objects = set(bimanual_by_object) | set(single_by_object)
    for object_name in sorted(all_objects):
        bimanual = sorted(bimanual_by_object[object_name], key=quality_key)
        single = list(single_by_object.get(object_name, []))
        rng.shuffle(single)
        target_bimanual = len(single) * args.bimanual_per_single
        if args.preserve_all_single:
            single_count = len(single)
            usable_bimanual = min(len(bimanual), target_bimanual)
        else:
            usable_bimanual = min(len(bimanual), target_bimanual)
            usable_bimanual -= usable_bimanual % args.bimanual_per_single
            single_count = usable_bimanual // args.bimanual_per_single
        selected_bimanual.extend(bimanual[:usable_bimanual])
        held_out_bimanual.extend(bimanual[usable_bimanual:])
        selected_single.extend(single[:single_count])
        rows.append(
            {
                "object_name": object_name,
                "bimanual_available": len(bimanual),
                "bimanual_target": target_bimanual,
                "bimanual_train": usable_bimanual,
                "bimanual_held_out": len(bimanual) - usable_bimanual,
                "bimanual_deficit": max(
                    target_bimanual - usable_bimanual,
                    0,
                ),
                "single_available": len(single),
                "single_train": single_count,
            }
        )

    expected = len(selected_single) * args.bimanual_per_single
    complete = len(selected_bimanual) == expected
    if not args.preserve_all_single and not complete:
        raise RuntimeError("Internal ratio error")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selected_bimanual, args.output_dir / "bimanual_train.pt")
    torch.save(selected_single, args.output_dir / "single_train.pt")
    torch.save(held_out_bimanual, args.output_dir / "bimanual_held_out.pt")
    manifest = {
        "dataset_version": "single_bimanual_object_matched_v1",
        "verification_level": args.verification_level,
        "bimanual_per_single": args.bimanual_per_single,
        "preserve_all_strict_single_samples": args.preserve_all_single,
        "target_bimanual_count": expected,
        "bimanual_train_count": len(selected_bimanual),
        "single_train_count": len(selected_single),
        "bimanual_deficit": expected - len(selected_bimanual),
        "ratio_complete": complete,
        "bimanual_held_out_count": len(held_out_bimanual),
        "deduplication_key": [
            "object_name",
            "left_source_index",
            "right_source_index",
            "opposition_roll_degrees",
        ],
        "duplicates_added": 0,
        "object_summary": rows,
        "source_bimanual_datasets": [
            str(path) for path in args.bimanual_datasets
        ],
        "source_single_vis": str(args.single_vis),
        "source_single_results": str(args.single_results),
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    with (args.output_dir / "object_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
