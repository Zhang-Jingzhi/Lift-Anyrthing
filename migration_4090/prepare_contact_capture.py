#!/usr/bin/env python3
"""Prepare one generated strict seed for the Isaac contact-report run."""

from pathlib import Path

import torch


source = Path(
    "graph_exp/bimanual_data/"
    "4090_smoke_cylinder_xlarge_source14_targeted_retry5/"
    "repeat_source_sample.pt"
)
output = source.parent / "contact_capture_input"
payload = torch.load(source, map_location="cpu")
sample = payload["samples"][0]
output.mkdir(parents=True, exist_ok=False)
torch.save(sample["left_q_seed"].unsqueeze(0), output / "left_q.pt")
torch.save(sample["right_q_seed"].unsqueeze(0), output / "right_q.pt")
print(output.resolve())
