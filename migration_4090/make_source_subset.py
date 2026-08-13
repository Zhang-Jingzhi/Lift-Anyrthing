#!/usr/bin/env python3
"""Create an auditable source_vis subset from real stored pose indices."""

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--object", required=True)
parser.add_argument("--indices", type=int, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

entries = torch.load("data/bimanual/source_vis.pt", map_location="cpu")
entry = next(item for item in entries if item["object_name"] == args.object)
indices = torch.as_tensor(args.indices, dtype=torch.long)
if int(indices.min()) < 0 or int(indices.max()) >= len(entry["predict_q"]):
    raise IndexError(args.indices)
derived = dict(entry)
derived["predict_q"] = entry["predict_q"][indices].clone()
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise RuntimeError(f"refusing to overwrite {args.output}")
torch.save([derived], args.output)
print("object", args.object)
print("parent_indices", args.indices)
print("shape", tuple(derived["predict_q"].shape))
print("all_finite", bool(torch.isfinite(derived["predict_q"]).all()))
print("output", args.output.resolve())
