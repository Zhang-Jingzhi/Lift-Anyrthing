#!/usr/bin/env python3
"""Extract the contact-capture final pose for matching mesh audit."""

from pathlib import Path

import torch


base = Path(
    "graph_exp/bimanual_data/"
    "4090_smoke_cylinder_xlarge_source14_targeted_retry5"
)
result = torch.load(base / "contact_capture_result.pt", map_location="cpu")
output = base / "contact_capture_geometry"
output.mkdir(parents=True, exist_ok=False)
torch.save(result["left_q_final"], output / "left_q.pt")
torch.save(result["right_q_final"], output / "right_q.pt")
print(output.resolve())
