#!/usr/bin/env python3
"""Assemble one method/base-object result according to the shared 100-row schedule."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from select_bulk_repeat_candidates import pose_key, rank_key
from formal_large_random_protocol import PROTOCOL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--base-object-name", required=True)
    parser.add_argument("--method", choices=("baseline", "bidex_v3"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    schedule = json.loads(args.schedule.read_text())
    rows = [
        row
        for row in schedule["entries"]
        if row["base_object_name"] == args.base_object_name
    ]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 schedule rows, found {len(rows)}")
    rows_by_size = defaultdict(list)
    for row in rows:
        rows_by_size[int(row["size_index"])].append(row)

    selected = []
    sources = []
    seen = set()
    collision_modes = set()
    simulation_densities = set()
    for size_index in sorted(rows_by_size):
        expected = sorted(rows_by_size[size_index], key=lambda row: row["sample_index"])
        path = (
            args.input_root
            / f"size_{size_index:03d}"
            / f"assembled_target_{len(expected):02d}"
            / "bimanual_dataset.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("manifest", {}).get("formal_protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"size {size_index}: protocol mismatch in {path}")
        collision_modes.update(
            payload.get("manifest", {}).get("object_collision_modes", [])
        )
        simulation_densities.update(
            payload.get("manifest", {}).get(
                "simulation_densities_kg_m3", []
            )
        )
        candidates = sorted(payload.get("samples", []), key=rank_key)
        unique = []
        for sample in candidates:
            key = (sample["object_name"], pose_key(sample))
            if key not in seen:
                seen.add(key)
                unique.append(sample)
        if len(unique) < len(expected):
            raise RuntimeError(
                f"size {size_index}: {len(unique)} unique, need {len(expected)}"
            )
        for sample, schedule_row in zip(unique[: len(expected)], expected):
            item = dict(sample)
            provenance = dict(item.get("provenance", {}))
            provenance.update(
                {
                    "formal_schedule_sample_index": int(schedule_row["sample_index"]),
                    "formal_schedule_sample_seed": int(schedule_row["sample_seed"]),
                    "formal_size_index": size_index,
                    "formal_scale_from_source": schedule_row["scale_from_source"],
                    "formal_extents_mm": schedule_row["extents_mm"],
                    "formal_density_kg_m3": schedule_row["density_kg_m3"],
                    "formal_target_mass_kg": schedule_row["target_mass_kg"],
                }
            )
            item["provenance"] = provenance
            selected.append(item)
        sources.append({"size_index": size_index, "dataset": str(path.resolve())})

    selected.sort(key=lambda sample: sample["provenance"]["formal_schedule_sample_index"])
    counts = Counter(sample["provenance"]["formal_size_index"] for sample in selected)
    if len(selected) != 100 or sorted(counts) != list(range(8)):
        raise RuntimeError(f"invalid final selection count={len(selected)} sizes={counts}")
    manifest = {
        "schema": "tro_grasp_large_random_6x100_method_dataset_v1",
        "formal_protocol_id": PROTOCOL_ID,
        "object_collision_mode": (
            next(iter(collision_modes))
            if len(collision_modes) == 1
            else "mixed_versioned_vhacd"
        ),
        "object_collision_modes": sorted(collision_modes),
        "object_vhacd": True,
        "object_vhacd_high_v1": "vhacd_high_v1" in collision_modes,
        "object_vhacd_visual_high_v2": (
            "vhacd_visual_high_v2" in collision_modes
        ),
        "simulation_densities_kg_m3": sorted(
            float(value) for value in simulation_densities
        ),
        "object_vhacd_resolution": 1_000_000,
        "object_vhacd_max_convex_hulls": 128,
        "object_vhacd_max_vertices": 64,
        "lift_command_height_m": 0.10,
        "contact_offset_m": 0.001,
        "method": args.method,
        "base_object_name": args.base_object_name,
        "num_samples": len(selected),
        "size_counts": dict(sorted(counts.items())),
        "shared_schedule": str(args.schedule.resolve()),
        "sources": sources,
        "gravity_m_s2": 9.8,
        "table_support": True,
        "simultaneous_closure": True,
        "six_independent_disturbances": True,
        "single_hand_ablations_required_to_fail": True,
        "repeat_rollouts_per_sample": 3,
        "bilateral_positive_impulse_contact_all_phases": True,
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "bimanual_dataset.pt"
    torch.save({"manifest": manifest, "samples": selected}, output)
    summary = {
        "method": args.method,
        "base_object_name": args.base_object_name,
        "num_samples": len(selected),
        "size_counts": dict(sorted(counts.items())),
        "dataset": str(output.resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
