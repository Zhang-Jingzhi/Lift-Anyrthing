#!/usr/bin/env python3
"""Concatenate compatible single-entry precomputed candidate sources."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    entry = None
    left_rows = []
    right_rows = []
    metadata = []
    object_name = None
    for source_index, path in enumerate(args.inputs):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, list) or len(payload) != 1:
            raise RuntimeError(f"expected one source entry in {path}")
        current = payload[0]
        if object_name is None:
            object_name = current["object_name"]
            entry = dict(current)
        elif current["object_name"] != object_name:
            raise RuntimeError("input object names differ")
        candidates = current["precomputed_bimanual_candidates"]
        left = candidates["left_q"]
        right = candidates["right_q"]
        if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 22:
            raise RuntimeError(f"invalid candidate tensors in {path}")
        rows = candidates.get("metadata", [])
        left_rows.append(left)
        right_rows.append(right)
        for row_index in range(len(left)):
            row = dict(rows[row_index]) if row_index < len(rows) else {}
            row.update(
                {
                    "source_pair_index": len(metadata),
                    "combined_source_index": source_index,
                    "combined_source_row_index": row_index,
                    "combined_source_path": str(path.resolve()),
                }
            )
            metadata.append(row)

    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.cat(left_rows),
        "right_q": torch.cat(right_rows),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(f"combined {len(metadata)} candidates -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
