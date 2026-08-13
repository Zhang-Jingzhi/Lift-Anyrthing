#!/usr/bin/env python3
"""Combine six already-capped object datasets into one 600-record corpus."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-object", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.method_root.glob("*/bimanual_dataset.pt"))
    samples = []
    sources = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        part = payload.get("samples", [])
        if len(part) != args.per_object:
            raise RuntimeError(f"{path} has {len(part)}, expected {args.per_object}")
        samples.extend(part)
        sources.append(str(path.resolve()))
    counts = Counter(sample["object_name"] for sample in samples)
    if len(counts) != 6 or set(counts.values()) != {args.per_object}:
        raise RuntimeError(f"Expected 6 x {args.per_object}, got {dict(counts)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "tro_grasp_bulk_irregular_600_repeat_verified_v1",
        "method": args.method,
        "total_samples": len(samples),
        "counts_by_object": dict(sorted(counts.items())),
        "repeat_rollouts_per_sample": 3,
        "physical_contact_capture_rollouts_per_sample": 2,
        "bilateral_positive_impulse_contact_required_at": [
            "closure",
            "after_lift",
            "after_gravity_settle",
        ],
        "source_datasets": sources,
    }
    torch.save(
        {"version": "bulk_irregular_v1", "manifest": manifest, "samples": samples},
        args.output,
    )
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
