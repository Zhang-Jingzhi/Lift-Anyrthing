#!/usr/bin/env python3
"""Recreate and export one auditable baseline candidate before physics."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from generate_bimanual_pilot import (
    clamp_to_joint_limits,
    place_opposite_tabletop,
    refine_radial_pose,
    select_symmetric_sources,
)
from utils.hand_model import create_hand_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--roll-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--left-index", type=int)
    parser.add_argument("--right-index", type=int)
    parser.add_argument("--roll-degrees", type=float)
    parser.add_argument("--contact-mm", type=float, required=True)
    parser.add_argument("--penetration-mm", type=float, default=2.0)
    parser.add_argument("--radial-min-offset-mm", type=float, default=-20.0)
    parser.add_argument("--radial-max-offset-mm", type=float, default=60.0)
    parser.add_argument("--radial-coarse-step-mm", type=float, default=10.0)
    parser.add_argument("--radial-fine-step-mm", type=float, default=2.0)
    parser.add_argument("--left-roll-degrees", type=float, default=0.0)
    parser.add_argument("--root-height-mm", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.audit_json.exists():
        raise RuntimeError("Refusing to overwrite output or audit JSON")
    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == args.object)
    left_hand = create_hand_model("allegro_left", torch.device("cpu"))
    right_hand = create_hand_model("allegro_right", torch.device("cpu"))
    q_batch = clamp_to_joint_limits(left_hand, entry["predict_q"].detach().cpu())
    direct = (
        args.left_index is not None
        and args.right_index is not None
        and args.roll_degrees is not None
    )
    if any(
        value is not None
        for value in (args.left_index, args.right_index, args.roll_degrees)
    ) and not direct:
        raise ValueError(
            "--left-index, --right-index and --roll-degrees must be supplied together"
        )
    pairs = (
        [
            {
                "left_index": args.left_index,
                "right_index": args.right_index,
                "internal_distance": 0.0,
                "source_root_cosine": 1.0,
            }
        ]
        if direct
        else select_symmetric_sources(q_batch, args.pairs, args.seed)
    )
    rolls = np.linspace(0.0, 180.0, args.roll_count, endpoint=False)
    if direct:
        rolls = np.asarray([args.roll_degrees], dtype=float)

    left_cache = {}
    right_cache = {}
    candidates = []
    dataset, name = args.object.split("+")
    mesh = trimesh.load_mesh(
        REPO / "data/data_urdf/object" / dataset / name / f"{name}.stl"
    )
    query = trimesh.proximity.ProximityQuery(mesh)
    for pair_index, pair in enumerate(pairs):
        for roll_degrees in rolls:
            left, right = place_opposite_tabletop(
                q_batch[pair["left_index"]],
                q_batch[pair["right_index"]],
                float(roll_degrees),
                args.left_roll_degrees,
                args.root_height_mm / 1000.0,
            )
            left_key = pair["left_index"]
            right_key = (
                pair["left_index"],
                pair["right_index"],
                float(roll_degrees),
            )
            if left_key not in left_cache:
                left_cache[left_key] = refine_radial_pose(
                    left_hand,
                    mesh,
                    query,
                    left,
                    args.contact_mm / 1000.0,
                    args.penetration_mm,
                    args.radial_min_offset_mm,
                    args.radial_max_offset_mm,
                    args.radial_coarse_step_mm,
                    args.radial_fine_step_mm,
                    horizontal_only=True,
                )
            if right_key not in right_cache:
                right_cache[right_key] = refine_radial_pose(
                    right_hand,
                    mesh,
                    query,
                    right,
                    args.contact_mm / 1000.0,
                    args.penetration_mm,
                    args.radial_min_offset_mm,
                    args.radial_max_offset_mm,
                    args.radial_coarse_step_mm,
                    args.radial_fine_step_mm,
                    horizontal_only=True,
                )
            left_refined, _, left_offset = left_cache[left_key]
            right_refined, _, right_offset = right_cache[right_key]
            candidates.append(
                (
                    left_refined.clone(),
                    right_refined.clone(),
                    {
                        **pair,
                        "source_pair_index": pair_index,
                        "opposition_roll_degrees": float(roll_degrees),
                        "left_outward_offset_mm": float(left_offset),
                        "right_outward_offset_mm": float(right_offset),
                        "candidate_method": "baseline_recreated_candidate_v1",
                        "original_candidate_index": args.candidate_index,
                    },
                )
            )

    selected_index = 0 if direct else args.candidate_index
    left, right, metadata = candidates[selected_index]
    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": left.unsqueeze(0),
        "right_q": right.unsqueeze(0),
        "metadata": [metadata],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save([derived], args.output)
    audit = {
        "object_name": args.object,
        "source_vis": str(args.source_vis.resolve()),
        "pairs": args.pairs,
        "roll_count": args.roll_count,
        "seed": args.seed,
        "candidate_index": args.candidate_index,
        "direct_source_indices": direct,
        "metadata": metadata,
        "left_q": left.tolist(),
        "right_q": right.tolist(),
        "output": str(args.output.resolve()),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"saved={args.output.resolve()} index={args.candidate_index}")


if __name__ == "__main__":
    main()
