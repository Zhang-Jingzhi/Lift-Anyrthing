#!/usr/bin/env python3
"""Select one strict generated sample for two additional rollouts."""

import json
from pathlib import Path

import torch


base = Path(
    "graph_exp/bimanual_data/"
    "4090_smoke_cylinder_xlarge_source14_targeted_retry5"
)
payload = torch.load(base / "bimanual_dataset.pt", map_location="cpu")
quality = json.loads((base / "local_quality.json").read_text())
eligible = [
    index
    for index, sample in enumerate(payload["samples"])
    if sample["metrics"]["strict_success"]
    and quality["samples"][index]["decoupled_force_closure_pass"]
]
if not eligible:
    raise RuntimeError("No strict locally balanced candidate is available")
selected_index = max(
    eligible,
    key=lambda index: payload["samples"][index]["metrics"][
        "lift_displacement_mm"
    ],
)
sample = payload["samples"][selected_index]
subset = {
    "manifest": {
        **payload.get("manifest", {}),
        "subset_purpose": "single-candidate repeat smoke test",
        "parent_sample_index": selected_index,
    },
    "samples": [sample],
}
quality_row = dict(quality["samples"][selected_index])
quality_row["sample_index"] = 0
quality_subset = {
    **{key: value for key, value in quality.items() if key != "samples"},
    "num_samples": 1,
    "num_pass": 1,
    "samples": [quality_row],
}
source_path = base / "repeat_source_sample.pt"
quality_path = base / "repeat_source_quality.json"
if source_path.exists() or quality_path.exists():
    raise RuntimeError("refusing to overwrite existing repeat source files")
torch.save(subset, source_path)
quality_path.write_text(json.dumps(quality_subset, indent=2) + "\n")
print("selected_parent_index", selected_index)
print("roll_degrees", sample["opposition_roll_degrees"])
print("first_rollout_lift_mm", sample["metrics"]["lift_displacement_mm"])
print("source", source_path.resolve())
print("quality", quality_path.resolve())
