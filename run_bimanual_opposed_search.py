"""Construct and filter opposed two-hand poses from single-hand predictions.

This is a deliberately small data-construction baseline.  One generated
Allegro grasp is kept fixed.  A second generated grasp is moved to the
opposite side of the object with four possible roll choices.  All candidates
are screened for hand-object penetration, hand-hand clearance, and bimanual
Isaac disturbance success.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from run_bimanual_pairing_baseline import (
    hand_clearances,
    opposite_root_matching,
    run_isaac,
)
from utils.hand_model import create_hand_model
from validation.validate_utils import validate_depth


ROLL_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


def oppose_pose(q, roll_degrees):
    """Rotate a world-space hand pose to the opposite object side."""
    result = q.clone()
    direction = q[:3].detach().cpu().numpy().astype(np.float64)
    direction /= max(np.linalg.norm(direction), 1e-12)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    tangent_1 = np.cross(direction, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(direction, tangent_1)
    angle = np.deg2rad(roll_degrees)
    axis = np.cos(angle) * tangent_1 + np.sin(angle) * tangent_2
    world_rotation = Rotation.from_rotvec(np.pi * axis).as_matrix()

    source_rotation = Rotation.from_euler(
        "XYZ",
        q[3:6].detach().cpu().numpy(),
    ).as_matrix()
    result[:3] = torch.as_tensor(
        world_rotation @ q[:3].detach().cpu().numpy(),
        dtype=q.dtype,
    )
    result[3:6] = torch.as_tensor(
        Rotation.from_matrix(world_rotation @ source_rotation).as_euler(
            "XYZ"
        ),
        dtype=q.dtype,
    )
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, object_rows, overall):
    lines = [
        "# Opposed-pose bimanual search and Isaac filtering",
        "",
        "For each pair of independent Allegro predictions, the left pose is kept",
        "fixed and the right pose is moved to the opposite object side. Four",
        "roll choices are searched. This is a joint geometric search baseline,",
        "not a learned bimanual model.",
        "",
        "A strict success requires bimanual Isaac success, both 5 mm exact",
        "hand-object penetration checks, and more than 2 mm sampled hand-hand",
        "clearance at both controller poses.",
        "",
        "| Object | Candidates | Isaac | Right mesh | Clearance | Strict | "
        "Successful source pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in object_rows:
        lines.append(
            f"| `{row['object_name']}` | {row['num_candidates']} | "
            f"{row['isaac_success_rate']:.1%} | "
            f"{row['right_mesh_pass_rate']:.1%} | "
            f"{row['clearance_pass_rate']:.1%} | "
            f"{row['strict_success_rate']:.1%} | "
            f"{row['successful_source_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Candidates: {overall['num_candidates']}.",
            f"- Isaac success: {overall['isaac_success_rate']:.1%}.",
            f"- Strict success: {overall['strict_success_rate']:.1%}.",
            f"- Retained bimanual demonstrations: "
            f"{overall['num_strict_success']}.",
            "",
            "The retained tensor file is intended as a seed dataset for a",
            "bimanual model smoke test. It must not be presented as real",
            "left/right Allegro hardware data because both actors use the same",
            "right-hand URDF.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--source-vis",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/single_hand_baseline/"
            "allegro_unconditioned/vis.pt"
        ),
    )
    parser.add_argument(
        "--object-split",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/object_selection/"
            "bimanual_object_split.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/opposed_pose_search"
        ),
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        required=True,
        help="Python executable from the Isaac Gym environment.",
    )
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--include-held-out", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--penetration-mm", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for name in ("source_vis", "object_split", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (args.repo / value).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.object_split.read_text())
    if args.objects:
        object_names = args.objects
    else:
        object_names = list(split["development"])
        if args.include_held_out:
            object_names += split["held_out_test"]

    entries = torch.load(args.source_vis, map_location="cpu")
    entry_lookup = {entry["object_name"]: entry for entry in entries}
    hand = create_hand_model("allegro", torch.device("cpu"))
    sample_rows = []
    retained = []

    for object_name in object_names:
        print(f"Searching {object_name}", flush=True)
        entry = entry_lookup[object_name]
        pairs = opposite_root_matching(entry["predict_q"])
        left_candidates = []
        right_candidates = []
        metadata = []
        for pair_index, (left_index, right_index, cosine) in enumerate(pairs):
            for roll_degrees in ROLL_ANGLES_DEG:
                left_candidates.append(entry["predict_q"][left_index].clone())
                right_candidates.append(
                    oppose_pose(
                        entry["predict_q"][right_index],
                        roll_degrees,
                    )
                )
                metadata.append(
                    (pair_index, left_index, right_index, cosine, roll_degrees)
                )
        left_q = torch.stack(left_candidates)
        right_q = torch.stack(right_candidates)

        clearances = hand_clearances(hand, left_q, right_q)
        right_mesh_pass, right_depth_mm = validate_depth(
            hand,
            object_name,
            right_q,
            threshold=args.penetration_mm / 1000.0,
            exact=True,
        )

        object_dir = args.output_dir / object_name.replace("+", "__")
        object_dir.mkdir(parents=True, exist_ok=True)
        isaac = run_isaac(
            args,
            object_name,
            left_q,
            right_q,
            object_dir,
        )

        for index, item in enumerate(metadata):
            (
                pair_index,
                left_index,
                right_index,
                cosine,
                roll_degrees,
            ) = item
            outer_clearance, inner_clearance = clearances[index]
            clearance_pass = (
                outer_clearance > args.clearance_mm
                and inner_clearance > args.clearance_mm
            )
            left_mesh_pass = bool(
                entry["penetration_success"][left_index]
            )
            mesh_pass = left_mesh_pass and bool(right_mesh_pass[index])
            isaac_success = bool(isaac["success"][index])
            strict_success = (
                isaac_success and mesh_pass and clearance_pass
            )
            row = {
                "object_name": object_name,
                "source_pair_index": pair_index,
                "left_source_index": left_index,
                "right_source_index": right_index,
                "source_root_cosine": cosine,
                "opposition_roll_degrees": roll_degrees,
                "left_mesh_pass": left_mesh_pass,
                "right_mesh_pass": bool(right_mesh_pass[index]),
                "right_penetration_depth_mm": right_depth_mm[index],
                "outer_hand_clearance_mm": outer_clearance,
                "inner_hand_clearance_mm": inner_clearance,
                "hand_hand_clearance_pass": clearance_pass,
                "settle_displacement_mm": float(
                    isaac["settle_displacement"][index] * 1000
                ),
                "disturbance_displacement_mm": float(
                    isaac["disturbance_displacement"][index] * 1000
                ),
                "isaac_success": isaac_success,
                "strict_success": strict_success,
            }
            sample_rows.append(row)
            if strict_success:
                retained.append(
                    {
                        "object_name": object_name,
                        "left_q": left_q[index],
                        "right_q": right_q[index],
                        "source_pair_index": pair_index,
                        "opposition_roll_degrees": roll_degrees,
                    }
                )

    object_rows = []
    for object_name in object_names:
        rows = [
            row for row in sample_rows
            if row["object_name"] == object_name
        ]
        successful_pairs = {
            row["source_pair_index"]
            for row in rows
            if row["strict_success"]
        }
        object_rows.append(
            {
                "object_name": object_name,
                "num_candidates": len(rows),
                "isaac_success_rate": np.mean(
                    [row["isaac_success"] for row in rows]
                ),
                "right_mesh_pass_rate": np.mean(
                    [row["right_mesh_pass"] for row in rows]
                ),
                "clearance_pass_rate": np.mean(
                    [row["hand_hand_clearance_pass"] for row in rows]
                ),
                "strict_success_rate": np.mean(
                    [row["strict_success"] for row in rows]
                ),
                "successful_source_pairs": len(successful_pairs),
            }
        )

    overall = {
        "num_candidates": len(sample_rows),
        "isaac_success_rate": np.mean(
            [row["isaac_success"] for row in sample_rows]
        ),
        "strict_success_rate": np.mean(
            [row["strict_success"] for row in sample_rows]
        ),
        "num_strict_success": len(retained),
    }
    write_csv(args.output_dir / "sample_results.csv", sample_rows)
    write_csv(args.output_dir / "object_summary.csv", object_rows)
    torch.save(retained, args.output_dir / "filtered_pairs.pt")
    write_report(args.output_dir / "REPORT.md", object_rows, overall)
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
