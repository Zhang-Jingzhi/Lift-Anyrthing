#!/usr/bin/env python3
"""Recover dense-geometry-safe candidates for the opposed/radial baseline."""

import argparse
import json
from pathlib import Path
import sys

import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_bimanual_pilot import hand_clearances
from migration_4090.generate_bidexgrasp_candidates import pose_audit
from utils.controller import controller
from utils.hand_model import create_hand_model


def side_variants(hand, seed, mesh, query, outward_values, aperture_values, args):
    _, base_inner = controller(hand.robot_name, seed, hand=hand)
    closing_direction = base_inner[6:] - seed[6:]
    rows = []
    for outward_mm in outward_values:
        for aperture_scale in aperture_values:
            q = seed.clone()
            radius = q[:2].norm().clamp_min(1e-8)
            q[:2] *= (radius + outward_mm / 1000.0) / radius
            q[6:] += aperture_scale * closing_direction
            outer_q, inner_q = controller(hand.robot_name, q, hand=hand)
            outer = pose_audit(
                hand, outer_q, mesh, query, mesh.centroid, args.contact_mm / 1000.0, 1.0
            )
            inner = pose_audit(
                hand, inner_q, mesh, query, mesh.centroid, args.contact_mm / 1000.0, 1.0
            )
            if not (
                outer["penetration_mm"] <= args.penetration_mm
                and inner["penetration_mm"] <= args.penetration_mm
                and inner["surface_distance_mm"] <= args.contact_mm
                and inner["contact_link_count"] >= args.minimum_contact_links
            ):
                continue
            score = (
                20.0 * inner["penetration_mm"] ** 2
                + 2.0 * inner["surface_distance_mm"]
                - 3.0 * inner["contact_link_count"]
                - 0.1 * inner["contact_point_count"]
                + 0.02 * abs(outward_mm)
                + 0.05 * abs(aperture_scale)
            )
            rows.append(
                {
                    "q": q,
                    "score": score,
                    "outward_mm": outward_mm,
                    "aperture_scale": aperture_scale,
                    "outer": outer,
                    "inner": inner,
                }
            )
    rows.sort(key=lambda row: row["score"])
    return rows[: args.side_variant_cap]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--parent-indices", type=int, nargs="+", required=True)
    parser.add_argument("--outward-mm", type=float, nargs="+", default=[0, 4, 8, 12, 16])
    parser.add_argument(
        "--aperture-scales", type=float, nargs="+", default=[-1, -0.5, 0, 0.5]
    )
    parser.add_argument("--contact-mm", type=float, default=2.0)
    parser.add_argument("--penetration-mm", type=float, default=2.0)
    parser.add_argument("--minimum-contact-links", type=int, default=1)
    parser.add_argument("--minimum-hand-clearance-mm", type=float, default=50.0)
    parser.add_argument("--side-variant-cap", type=int, default=8)
    parser.add_argument("--parent-candidate-cap", type=int, default=20)
    parser.add_argument(
        "--candidate-method",
        default="baseline_opposed_radial_dense_joint_recovery_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit_json.exists():
        raise RuntimeError("refusing to overwrite output")

    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == args.object)
    pool = entry["precomputed_bimanual_candidates"]
    dataset, name = args.object.split("+", 1)
    mesh_path = ROOT / "data/data_urdf/object" / dataset / name / f"{name}.stl"
    mesh = trimesh.load_mesh(mesh_path)
    query = trimesh.proximity.ProximityQuery(mesh)
    left_hand = create_hand_model("allegro_left", torch.device("cpu"))
    right_hand = create_hand_model("allegro_right", torch.device("cpu"))

    left_q = []
    right_q = []
    metadata = []
    audit_parents = []
    for parent_index in args.parent_indices:
        left = side_variants(
            left_hand,
            pool["left_q"][parent_index],
            mesh,
            query,
            args.outward_mm,
            args.aperture_scales,
            args,
        )
        right = side_variants(
            right_hand,
            pool["right_q"][parent_index],
            mesh,
            query,
            args.outward_mm,
            args.aperture_scales,
            args,
        )
        combinations = []
        for left_row in left:
            for right_row in right:
                outer_clearance, inner_clearance = hand_clearances(
                    left_hand,
                    left_row["q"].unsqueeze(0),
                    right_row["q"].unsqueeze(0),
                    right_hand=right_hand,
                )[0]
                clearance = min(outer_clearance, inner_clearance)
                if clearance < args.minimum_hand_clearance_mm:
                    continue
                combinations.append(
                    (
                        left_row["score"] + right_row["score"],
                        left_row,
                        right_row,
                        outer_clearance,
                        inner_clearance,
                    )
                )
        combinations.sort(key=lambda row: row[0])
        selected = combinations[: args.parent_candidate_cap]
        for rank, (score, left_row, right_row, outer_clearance, inner_clearance) in enumerate(selected):
            left_q.append(left_row["q"])
            right_q.append(right_row["q"])
            metadata.append(
                {
                    "candidate_method": args.candidate_method,
                    "source_parent_index": parent_index,
                    "local_variant_rank": rank,
                    "left_outward_offset_mm": left_row["outward_mm"],
                    "right_outward_offset_mm": right_row["outward_mm"],
                    "left_aperture_scale": left_row["aperture_scale"],
                    "right_aperture_scale": right_row["aperture_scale"],
                    "left_dense_contact_links": left_row["inner"]["contact_link_count"],
                    "right_dense_contact_links": right_row["inner"]["contact_link_count"],
                    "left_dense_penetration_mm": left_row["inner"]["penetration_mm"],
                    "right_dense_penetration_mm": right_row["inner"]["penetration_mm"],
                    "outer_hand_clearance_mm": outer_clearance,
                    "inner_hand_clearance_mm": inner_clearance,
                    "recovery_objective": score,
                }
            )
        audit_parents.append(
            {
                "source_parent_index": parent_index,
                "left_feasible_variants": len(left),
                "right_feasible_variants": len(right),
                "paired_candidates": len(selected),
            }
        )
    if not metadata:
        raise RuntimeError("no dense-safe recovery candidates")
    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_q),
        "right_q": torch.stack(right_q),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save([derived], args.output)
    audit = {
        "method": args.candidate_method,
        "object_name": args.object,
        "mesh_extents_mm": (mesh.extents * 1000.0).tolist(),
        "parent_indices": args.parent_indices,
        "outward_mm": args.outward_mm,
        "aperture_scales": args.aperture_scales,
        "parents": audit_parents,
        "candidate_count": len(metadata),
        "output": str(args.output.resolve()),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
