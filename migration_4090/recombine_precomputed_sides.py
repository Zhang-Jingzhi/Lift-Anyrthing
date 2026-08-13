#!/usr/bin/env python3
"""Recombine audited decoupled left/right candidates with full provenance."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--left-indices", type=int, nargs="+", required=True)
    parser.add_argument("--right-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == args.object)
    pool = entry["precomputed_bimanual_candidates"]
    count = len(pool["metadata"])
    requested = args.left_indices + args.right_indices
    if min(requested) < 0 or max(requested) >= count:
        raise IndexError(f"indices outside candidate pool of length {count}")

    left_q = []
    right_q = []
    metadata = []
    for left_index in args.left_indices:
        for right_index in args.right_indices:
            left_meta = pool["metadata"][left_index]
            right_meta = pool["metadata"][right_index]
            left_q.append(pool["left_q"][left_index].clone())
            right_q.append(pool["right_q"][right_index].clone())
            metadata.append(
                {
                    "candidate_method": "bidex_v3_same_parent_decoupled_recombination_v1",
                    "left_source_candidate_index": left_index,
                    "right_source_candidate_index": right_index,
                    "left_region_pair_order": left_meta.get("region_pair_order"),
                    "right_region_pair_order": right_meta.get("region_pair_order"),
                    "same_region_pair_parent": left_meta.get("region_pair_order")
                    == right_meta.get("region_pair_order"),
                    "left_source_metadata": left_meta,
                    "right_source_metadata": right_meta,
                }
            )
    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_q),
        "right_q": torch.stack(right_q),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([derived], args.output)
    same_parent = sum(row["same_region_pair_parent"] for row in metadata)
    print(
        f"saved {len(metadata)} recombinations ({same_parent} same-parent) "
        f"-> {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
