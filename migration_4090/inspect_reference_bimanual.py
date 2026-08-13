#!/usr/bin/env python3
"""Read only provenance/metrics from the optional packaged reference data."""

from pathlib import Path

import torch


path = Path("data/bimanual/bimanual_dataset.pt")
payload = torch.load(path, map_location="cpu")
print("TYPE", type(payload).__name__)
if isinstance(payload, dict):
    print("MANIFEST", payload.get("manifest"))
    samples = payload.get("samples", [])
else:
    samples = payload
print("SAMPLES", len(samples))
for index, sample in enumerate(samples):
    print(
        index,
        "object=", sample.get("object_name"),
        "left_source=", sample.get("left_source_index"),
        "right_source=", sample.get("right_source_index"),
        "roll=", sample.get("opposition_roll_degrees"),
        "provenance=", sample.get("provenance"),
        "metrics=", sample.get("metrics"),
    )
