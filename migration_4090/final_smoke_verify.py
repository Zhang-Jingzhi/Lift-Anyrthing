#!/usr/bin/env python3
"""Independently verify all final smoke artifacts and numeric invariants."""

import csv
import json
import math
from pathlib import Path

from PIL import Image
import torch


repo = Path(__file__).resolve().parents[1]
base = (
    repo
    / "graph_exp/bimanual_data/"
    "4090_smoke_cylinder_xlarge_source14_targeted_retry5"
)
repeat = base / "repeat_verified_gpu0"

dataset = torch.load(repeat / "verified_dataset.pt", map_location="cpu")
assert isinstance(dataset, dict) and len(dataset["samples"]) == 1
sample = dataset["samples"][0]
assert sample["metrics"]["strict_success"] is True
assert sample["metrics"]["repeat_count"] == 3
assert sample["metrics"]["repeat_success_rate"] == 1.0
for key in ("left_q", "right_q", "left_q_seed", "right_q_seed"):
    assert tuple(sample[key].shape) == (22,)
    assert bool(torch.isfinite(sample[key]).all())

metrics = json.loads((repeat / "strict_sample_metrics.json").read_text())
assert metrics["repeat_count"] == 3
assert metrics["repeat_success_rate"] == 1.0
assert metrics["left_only"]["success"] is False
assert metrics["right_only"]["success"] is False
assert metrics["lift_height_mm"] >= 30.0
assert metrics["gravity_stage_object_displacement_mm"] <= 10.0
assert metrics["six_direction_max_displacement_mm"] <= 15.0
assert metrics["left_max_penetration_mm"] <= 2.0
assert metrics["right_max_penetration_mm"] <= 2.0
assert metrics["inter_hand_min_clearance_mm"] > 2.0

contacts = json.loads((repeat / "physical_contact_points.json").read_text())
for phase_name, phase in contacts["phases"].items():
    assert phase["num_points"] == len(phase["points"])
    assert phase["point_count_by_side"]["left"] > 0
    assert phase["point_count_by_side"]["right"] > 0
    for row in phase["points"]:
        values = (
            row["local_pos0_m"]
            + row["local_pos1_m"]
            + row["world_pos0_m"]
            + row["world_pos1_m"]
            + row["world_midpoint_m"]
            + row["normal"]
            + [
                row["initial_overlap_m"],
                row["min_dist_m"],
                row["normal_impulse"],
                row["friction"],
            ]
        )
        assert all(math.isfinite(value) for value in values), phase_name

with (repeat / "physical_settled_contact_points.csv").open(newline="") as f:
    physical_rows = list(csv.DictReader(f))
with (repeat / "geometric_contact_points.csv").open(newline="") as f:
    geometric_rows = list(csv.DictReader(f))
assert len(physical_rows) == 48
assert len(geometric_rows) == 229

with Image.open(repeat / "verified_lateral_grasp_three_views.png") as image:
    assert image.size == (3600, 1280)
    assert image.mode == "RGBA"

print("FINAL_SMOKE_ARTIFACTS_OK")
print("verified_samples", len(dataset["samples"]))
print("repeat_success_rate", metrics["repeat_success_rate"])
print("physical_contacts", len(physical_rows))
print("geometric_contacts", len(geometric_rows))
print("image", "3600x1280 RGBA")
