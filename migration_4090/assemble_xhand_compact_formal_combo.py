#!/usr/bin/env python3
"""Assemble the 100 accepted records for one object/method combination."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = []
    samples = []
    for path in sorted(args.accepted_root.glob("sample_*.json")):
        row = json.loads(path.read_text())
        if row.get("object") != args.object or row.get("method") != args.method:
            raise RuntimeError(f"mixed accepted directory: {path}")
        if row.get("strict_success_rate") != 1.0:
            raise RuntimeError(f"non-strict accepted record: {path}")
        sample_path = Path(row["sample_path"])
        if not sample_path.is_file():
            raise FileNotFoundError(sample_path)
        samples.append(torch.load(sample_path, map_location="cpu", weights_only=False))
        manifests.append(row)
    if len(samples) != args.target:
        raise RuntimeError(f"expected {args.target} accepted samples, found {len(samples)}")

    counts = Counter(int(row["size_index"]) for row in manifests)
    payload = {
        "schema": "xhand_compact_formal_dataset_v1",
        "object": args.object,
        "method": args.method,
        "strict_isaac_validated": True,
        "strict_repeat_count": 3,
        "sample_count": len(samples),
        "size_counts": dict(sorted(counts.items())),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "dataset": str(args.output.resolve()),
        "object": args.object,
        "method": args.method,
        "sample_count": len(samples),
        "size_counts": dict(sorted(counts.items())),
        "mean_lift_height_mm": sum(r["lift_height_mm"] for r in manifests) / len(manifests),
        "max_gravity_displacement_mm": max(r["gravity_displacement_mm"] for r in manifests),
        "max_six_direction_displacement_mm": max(r["six_direction_max_mm"] for r in manifests),
        "all_ablation_pass": all(
            not r["left_only_physical_pass"] and not r["right_only_physical_pass"]
            for r in manifests
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
