#!/usr/bin/env python3
"""Build a traceable precomputed-candidate pool from completed Isaac chunks."""

import argparse
import csv
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == args.object)
    chunks = sorted((args.run_root / args.object.replace("+", "__") / "isaac_both_chunks").glob("*"))
    if not chunks:
        raise RuntimeError("no completed isaac_both_chunks found")
    left_q = torch.cat(
        [torch.load(path / "left_q.pt", map_location="cpu", weights_only=False) for path in chunks]
    )
    right_q = torch.cat(
        [torch.load(path / "right_q.pt", map_location="cpu", weights_only=False) for path in chunks]
    )
    with (args.run_root / "sample_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    geometry_indices = [
        int(row["candidate_index"]) for row in rows if row["geometry_pass"] == "True"
    ]
    if len(left_q) != len(right_q) or len(left_q) != len(geometry_indices):
        raise RuntimeError(
            f"pool mismatch left={len(left_q)} right={len(right_q)} "
            f"geometry={len(geometry_indices)}"
        )
    metadata = [
        {
            "candidate_method": "baseline_geometry_pass_replay_v1",
            "source_run": str(args.run_root.resolve()),
            "source_candidate_index": index,
        }
        for index in geometry_indices
    ]
    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": left_q,
        "right_q": right_q,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([derived], args.output)
    print(f"saved {len(metadata)} candidates -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
