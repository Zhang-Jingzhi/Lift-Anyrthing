#!/usr/bin/env python3
"""Assemble exactly N repeat-verified samples for one object and method."""

import argparse
import csv
import json
from pathlib import Path

import torch

from select_bulk_repeat_candidates import pose_key, rank_key
from formal_large_random_protocol import (
    PROTOCOL_ID,
    is_formal_protocol_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dataset = args.output_dir / "bimanual_dataset.pt"
    if args.output_dir.exists() or output_dataset.exists():
        raise FileExistsError(args.output_dir)

    paths = sorted(args.input_root.glob("seed_*/repeat_verified/verified_dataset.pt"))
    if not paths:
        raise RuntimeError(f"No repeat-verified inputs under {args.input_root}")
    unique = {}
    source_rows = []
    skipped_protocol = []
    collision_modes = set()
    simulation_densities = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "large_random_6x100_v1" in str(args.input_root):
            if not is_formal_protocol_manifest(payload.get("manifest", {})):
                skipped_protocol.append(str(path.resolve()))
                continue
        source_manifest = payload.get("manifest", {})
        collision_modes.add(source_manifest.get("object_collision_mode"))
        simulation_densities.add(source_manifest.get("object_density_kg_m3"))
        count = len(payload.get("samples", []))
        source_rows.append({"path": str(path.resolve()), "samples": count})
        for sample in payload.get("samples", []):
            if sample.get("object_name") != args.object_name:
                raise ValueError(f"Unexpected object in {path}: {sample.get('object_name')}")
            key = pose_key(sample)
            previous = unique.get(key)
            if previous is None or rank_key(sample) < rank_key(previous):
                unique[key] = sample
    ranked = sorted(unique.values(), key=rank_key)
    if len(ranked) < args.target:
        raise RuntimeError(
            f"Only {len(ranked)} unique repeat-verified samples; target={args.target}"
        )
    selected = ranked[: args.target]
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema": "tro_grasp_bulk_bimanual_repeat_verified_v1",
        "formal_protocol_id": PROTOCOL_ID,
        "object_collision_modes": sorted(
            mode for mode in collision_modes if mode is not None
        ),
        "simulation_densities_kg_m3": sorted(
            float(value) for value in simulation_densities if value is not None
        ),
        "method": args.method,
        "object_name": args.object_name,
        "target_samples": args.target,
        "repeat_rollouts_per_sample": 3,
        "physical_contact_capture_rollouts_per_sample": 2,
        "gravity_m_s2": 9.8,
        "table_support": True,
        "simultaneous_closure": True,
        "six_independent_disturbances": True,
        "single_hand_ablations_required_to_fail": True,
        "bilateral_positive_impulse_contact_required_at": [
            "closure",
            "after_lift",
            "after_gravity_settle",
        ],
        "real_allegro_left_and_right": True,
        "source_datasets": source_rows,
        "skipped_legacy_protocol_datasets": skipped_protocol,
        "unique_verified_available": len(ranked),
        "selection_rank": (
            "balanced_contact,total_contact,penetration,disturbance,"
            "gravity,lift,clearance"
        ),
    }
    torch.save(
        {"version": "bulk_irregular_v1", "manifest": manifest, "samples": selected},
        output_dataset,
    )
    with (args.output_dir / "sources.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "samples"])
        writer.writeheader()
        writer.writerows(source_rows)
    summary = {
        "method": args.method,
        "object_name": args.object_name,
        "num_samples": len(selected),
        "unique_verified_available": len(ranked),
        "dataset": str(output_dataset.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
