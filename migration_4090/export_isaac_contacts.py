#!/usr/bin/env python3
"""Export Isaac Gym object/hand contact records without dropping points."""

import argparse
import csv
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = torch.load(args.input, map_location="cpu", weights_only=False)
    rows = []
    for result_key, stage in (
        ("closure_contacts", "closure"),
        ("lifted_contacts", "lifted"),
        ("settled_contacts", "settled"),
    ):
        for sample_index, contacts in enumerate(result.get(result_key, [])):
            for contact in contacts:
                row = {"stage": stage, "sample_index": sample_index, **contact}
                row["active_impulse"] = float(row.get("normal_impulse", 0.0)) > 0.0
                rows.append(row)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    scalar_fields = [
        "stage", "sample_index", "contact_index", "actor0", "actor1",
        "body0_name", "body1_name", "normal_impulse", "friction",
        "initial_overlap_m", "min_dist_m", "active_impulse",
        "world_x_m", "world_y_m", "world_z_m",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            midpoint = row["world_midpoint_m"]
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in scalar_fields
                    if not key.startswith("world_")
                }
                | {
                    "world_x_m": midpoint[0],
                    "world_y_m": midpoint[1],
                    "world_z_m": midpoint[2],
                }
            )
    summary = {
        "input": str(args.input.resolve()),
        "total_records": len(rows),
        "active_impulse_records": sum(row["active_impulse"] for row in rows),
        "by_stage": {
            stage: {
                "records": sum(row["stage"] == stage for row in rows),
                "active_impulse_records": sum(
                    row["stage"] == stage and row["active_impulse"] for row in rows
                ),
            }
            for stage in ("closure", "lifted", "settled")
        },
        "records": rows,
    }
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
