#!/usr/bin/env python3
"""Expand promising piggy poses with constrained asymmetric height repair."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--opposed-z-mm", type=float, nargs="+", default=(15, 20, 25, 27)
    )
    parser.add_argument(
        "--joint-delta-rad", type=float, nargs="+", default=(0.0, 0.02)
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    entries = torch.load(args.input, map_location="cpu", weights_only=False)
    if len(entries) != 1:
        raise ValueError("Expected exactly one object entry")
    entry = dict(entries[0])
    source = entry["precomputed_bimanual_candidates"]
    left_source = source["left_q"]
    right_source = source["right_q"]
    source_metadata = source.get("metadata", [{} for _ in left_source])
    left_rows = []
    right_rows = []
    metadata = []
    for parent_index in args.parent_indices:
        if parent_index < 0 or parent_index >= len(left_source):
            raise IndexError(parent_index)
        for opposed_z_mm in args.opposed_z_mm:
            for left_joint_delta in args.joint_delta_rad:
                for right_joint_delta in args.joint_delta_rad:
                    left = left_source[parent_index].clone()
                    right = right_source[parent_index].clone()
                    # Parents place the right wrist above the left by 26.4 mm.
                    # Move both toward a height-balanced final rollout while
                    # retaining the strict <=55 mm initial height difference.
                    left[2] += opposed_z_mm / 1000.0
                    right[2] -= opposed_z_mm / 1000.0
                    left[6:] += left_joint_delta
                    right[6:] += right_joint_delta
                    left_rows.append(left)
                    right_rows.append(right)
                    metadata.append(
                        {
                            **source_metadata[parent_index],
                            "candidate_method": (
                                "piggy_visual_high_v2_asymmetric_height_repair"
                            ),
                            "repair_parent_index": parent_index,
                            "repair_left_z_mm": opposed_z_mm,
                            "repair_right_z_mm": -opposed_z_mm,
                            "repair_left_joint_delta_rad": left_joint_delta,
                            "repair_right_joint_delta_rad": right_joint_delta,
                        }
                    )
    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_rows),
        "right_q": torch.stack(right_rows),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(f"expanded {len(left_rows)} candidates -> {args.output}")


if __name__ == "__main__":
    main()
