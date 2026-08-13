#!/usr/bin/env python3
"""Derive a minimal smoke input from source_vis without fabricating poses."""

from pathlib import Path

import torch


source_path = Path("data/bimanual/source_vis.pt")
output_path = Path(
    "migration_4090/derived/source_vis_cylinder_xlarge_source14.pt"
)
entries = torch.load(source_path, map_location="cpu")
source = next(
    entry
    for entry in entries
    if entry["object_name"] == "contactdb+cylinder_xlarge"
)
if source["predict_q"].shape[0] <= 14:
    raise RuntimeError("cylinder_xlarge does not contain source pose 14")
derived = dict(source)
derived["predict_q"] = source["predict_q"][14:15].clone()
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    existing = torch.load(output_path, map_location="cpu")
    if not torch.equal(existing[0]["predict_q"], derived["predict_q"]):
        raise RuntimeError(f"refusing to overwrite differing {output_path}")
else:
    torch.save([derived], output_path)
print(output_path.resolve())
print("object", derived["object_name"])
print("predict_q_shape", tuple(derived["predict_q"].shape))
print("all_finite", bool(torch.isfinite(derived["predict_q"]).all()))
