#!/usr/bin/env python3
"""Rebuild selected baseline candidates without repeating radial search."""

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_bimanual_pilot import (
    clamp_to_joint_limits,
    place_opposite_tabletop,
)
from utils.hand_model import create_hand_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-indices", type=int, nargs="*")
    parser.add_argument("--tabletop-left-roll-degrees", type=float, default=0.0)
    parser.add_argument("--tabletop-root-height-mm", type=float, default=60.0)
    args = parser.parse_args()

    entries = torch.load(args.source_vis, map_location="cpu")
    if len(entries) != 1:
        raise ValueError("Replay source must contain exactly one object entry")
    entry = dict(entries[0])
    left_hand = create_hand_model("allegro_left", torch.device("cpu"))
    q_batch = clamp_to_joint_limits(
        left_hand, entry["predict_q"].detach().cpu()
    )
    with args.sample_csv.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["geometry_pass"] == "True"]
    if args.candidate_indices:
        selected = set(args.candidate_indices)
        rows = [row for row in rows if int(row["candidate_index"]) in selected]
    if not rows:
        raise RuntimeError("No geometry-pass candidates found")

    left_candidates = []
    right_candidates = []
    metadata = []
    for replay_index, row in enumerate(rows):
        source_index = int(row["left_index"])
        roll = float(row["opposition_roll_degrees"])
        left_q, right_q = place_opposite_tabletop(
            q_batch[source_index],
            q_batch[int(row["right_index"])],
            roll,
            args.tabletop_left_roll_degrees,
            args.tabletop_root_height_mm / 1000.0,
        )
        for candidate, offset_key in (
            (left_q, "left_outward_offset_mm"),
            (right_q, "right_outward_offset_mm"),
        ):
            direction = candidate[:3].clone()
            direction[2] = 0.0
            direction /= direction.norm().clamp_min(1e-8)
            candidate[:3] += direction * (float(row[offset_key]) / 1000.0)
        left_candidates.append(left_q)
        right_candidates.append(right_q)
        metadata.append(
            {
                "source_pair_index": replay_index,
                "left_index": replay_index,
                "right_index": replay_index,
                "opposition_roll_degrees": roll,
                "left_outward_offset_mm": float(row["left_outward_offset_mm"]),
                "right_outward_offset_mm": float(row["right_outward_offset_mm"]),
                "synthesis_method": "baseline_opposed_radial_replay",
                "original_candidate_index": int(row["candidate_index"]),
            }
        )
    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_candidates),
        "right_q": torch.stack(right_candidates),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(f"saved {len(rows)} candidates to {args.output}")


if __name__ == "__main__":
    main()
