#!/usr/bin/env python3
"""Extract full-physics-successful poses from a completed pilot run.

The Isaac chunk inputs are the exact candidate seeds evaluated by physics.
This utility maps them back to ``sample_results.csv`` and writes a normal
``source_vis`` entry with precomputed bimanual candidates for local retry.
"""

import argparse
import csv
from pathlib import Path

import torch


def load_chunk_inputs(chunk_root: Path):
    left = []
    right = []
    for chunk in sorted(chunk_root.glob("*/left_q.pt")):
        right_path = chunk.with_name("right_q.pt")
        if not right_path.is_file():
            raise FileNotFoundError(right_path)
        left.append(torch.load(chunk, map_location="cpu", weights_only=False))
        right.append(
            torch.load(right_path, map_location="cpu", weights_only=False)
        )
    if not left:
        raise RuntimeError(f"no Isaac chunk inputs under {chunk_root}")
    return torch.cat(left), torch.cat(right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = list(csv.DictReader((args.run_dir / "sample_results.csv").open()))
    geometry_rows = [row for row in rows if row["geometry_pass"] == "True"]
    selected_positions = [
        index
        for index, row in enumerate(geometry_rows)
        if row["isaac_success"] == "True"
        and row["gravity_success"] == "True"
    ]
    if not selected_positions:
        raise RuntimeError("run has no full-physics-successful candidates")

    object_dir = args.run_dir / args.object.replace("+", "__")
    left_q, right_q = load_chunk_inputs(object_dir / "isaac_both_chunks")
    if len(left_q) != len(geometry_rows):
        raise RuntimeError(
            f"chunk rows={len(left_q)} != geometry rows={len(geometry_rows)}"
        )

    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    matches = [entry for entry in entries if entry["object_name"] == args.object]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source entry for {args.object}")
    entry = dict(matches[0])
    metadata = []
    for output_index, position in enumerate(selected_positions):
        row = geometry_rows[position]
        metadata.append(
            {
                "source_pair_index": output_index,
                "candidate_method": "physics_success_local_contact_retry",
                "extracted_parent_candidate_index": int(row["candidate_index"]),
                "extracted_parent_source_pair_index": int(
                    row["source_pair_index"]
                ),
                "extracted_parent_opposition_roll_degrees": float(
                    row["opposition_roll_degrees"]
                ),
                "extracted_parent_gravity_displacement_mm": float(
                    row["gravity_displacement_mm"]
                ),
                "extracted_parent_lift_displacement_mm": float(
                    row["lift_displacement_mm"]
                ),
                "extracted_parent_max_direction_displacement_mm": float(
                    row["max_direction_displacement_mm"]
                ),
            }
        )
    index_tensor = torch.as_tensor(selected_positions, dtype=torch.long)
    entry["precomputed_bimanual_candidates"] = {
        "left_q": left_q[index_tensor].clone(),
        "right_q": right_q[index_tensor].clone(),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(
        f"extracted {len(selected_positions)} physics-success parents -> "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
