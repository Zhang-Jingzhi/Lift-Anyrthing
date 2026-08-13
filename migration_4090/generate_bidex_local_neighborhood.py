#!/usr/bin/env python3
"""Create an auditable Cartesian neighborhood around a verified BiDex seed."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


def rotate_pose(q, yaw_degrees):
    result = q.clone()
    yaw = Rotation.from_euler("Z", yaw_degrees, degrees=True).as_matrix()
    xyz = result[:3].detach().cpu().numpy()
    xyz[:2] = (yaw @ xyz)[:2]
    orientation = Rotation.from_euler(
        "XYZ", result[3:6].detach().cpu().numpy()
    ).as_matrix()
    result[:3] = torch.as_tensor(xyz, dtype=result.dtype)
    result[3:6] = torch.as_tensor(
        Rotation.from_matrix(yaw @ orientation).as_euler("XYZ"),
        dtype=result.dtype,
    )
    return result


def add_radial_offset(q, offset_mm):
    result = q.clone()
    direction = result[:2] / result[:2].norm().clamp_min(1e-8)
    result[:2] += direction * (offset_mm / 1000.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument(
        "--target-object",
        help="Target object entry; defaults to --object for same-size expansion.",
    )
    parser.add_argument(
        "--root-scale-ratio",
        type=float,
        default=1.0,
        help=(
            "Uniform target/source object scale ratio applied to both wrist "
            "translations before local refinement."
        ),
    )
    parser.add_argument(
        "--source-mesh",
        type=Path,
        help="Visual mesh of --object, required with surface-offset transfer.",
    )
    parser.add_argument(
        "--preserve-root-surface-offset",
        action="store_true",
        help=(
            "Scale the closest source-mesh surface point while preserving "
            "the absolute wrist-to-surface vector. This avoids scaling the "
            "Allegro hand standoff when only the object is enlarged."
        ),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--candidate-method",
        default="bidexgrasp_region_gws_local_success_neighborhood_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--yaw-degrees", type=float, nargs="+", default=[-6, -3, 0, 3, 6]
    )
    parser.add_argument(
        "--z-mm", type=float, nargs="+", default=[-4, -2, 0, 2, 4]
    )
    parser.add_argument(
        "--left-radial-mm", type=float, nargs="+", default=[-4, -2, 0, 2, 4]
    )
    parser.add_argument(
        "--right-radial-mm", type=float, nargs="+", default=[-4, -2, 0, 2, 4]
    )
    args = parser.parse_args()

    if args.output.exists() or args.audit_json.exists():
        raise RuntimeError("Refusing to overwrite output or audit JSON")
    if args.root_scale_ratio <= 0:
        raise ValueError("--root-scale-ratio must be positive")
    target_object = args.target_object or args.object
    dataset = torch.load(
        args.source_dataset, map_location="cpu", weights_only=False
    )
    samples = dataset["samples"] if isinstance(dataset, dict) else dataset
    sample = samples[args.sample_index]
    if sample["object_name"] != args.object:
        raise ValueError("Selected sample does not match --object")
    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == target_object)
    # For a strict/repeat-verified parent, the optimized final pose is the
    # pose that actually passed Isaac and realized-geometry checks.  Expanding
    # the original optimizer seed can move the center candidate outside that
    # verified basin and produce a repair pool with zero strict samples.  Keep
    # the historical seed behavior for non-strict parents, but promote the
    # final pose whenever the parent carries a strict-success metric.
    parent_metrics = sample.get("metrics", {})
    use_final_parent = bool(
        parent_metrics.get("strict_success")
        and "left_q" in sample
        and "right_q" in sample
    )
    parent_left_key = "left_q" if use_final_parent else "left_q_seed"
    parent_right_key = "right_q" if use_final_parent else "right_q_seed"
    left_seed = sample[parent_left_key].detach().cpu()
    right_seed = sample[parent_right_key].detach().cpu()
    left_seed = left_seed.clone()
    right_seed = right_seed.clone()
    if args.preserve_root_surface_offset:
        if args.source_mesh is None:
            raise ValueError(
                "--source-mesh is required with --preserve-root-surface-offset"
            )
        mesh = trimesh.load_mesh(args.source_mesh, process=False)
        roots = np.stack(
            (
                left_seed[:3].detach().cpu().numpy(),
                right_seed[:3].detach().cpu().numpy(),
            )
        )
        closest, _, _ = trimesh.proximity.closest_point(mesh, roots)
        transferred = closest * float(args.root_scale_ratio) + (roots - closest)
        left_seed[:3] = torch.as_tensor(transferred[0], dtype=left_seed.dtype)
        right_seed[:3] = torch.as_tensor(transferred[1], dtype=right_seed.dtype)
    else:
        left_seed[:3] *= float(args.root_scale_ratio)
        right_seed[:3] *= float(args.root_scale_ratio)
    left_candidates = []
    right_candidates = []
    metadata = []
    for yaw in args.yaw_degrees:
        left_rotated = rotate_pose(left_seed, yaw)
        right_rotated = rotate_pose(right_seed, yaw)
        for z_mm in args.z_mm:
            for left_radial_mm in args.left_radial_mm:
                for right_radial_mm in args.right_radial_mm:
                    left = add_radial_offset(left_rotated, left_radial_mm)
                    right = add_radial_offset(right_rotated, right_radial_mm)
                    left[2] += z_mm / 1000.0
                    right[2] += z_mm / 1000.0
                    left_candidates.append(left)
                    right_candidates.append(right)
                    metadata.append(
                        {
                            "source_pair_index": int(
                                sample.get("source_pair_index", -1)
                            ),
                            "left_index": int(
                                sample.get("left_source_index", -1)
                            ),
                            "right_index": int(
                                sample.get("right_source_index", -1)
                            ),
                            "opposition_roll_degrees": float(
                                sample.get("opposition_roll_degrees", 0.0)
                            ),
                            "candidate_method": (
                                args.candidate_method
                            ),
                            "parent_candidate_index": int(
                                sample.get("candidate_index", -1)
                            ),
                            "parent_gws_score": parent_metrics.get("gws_score"),
                            "transfer_source_object": args.object,
                            "transfer_target_object": target_object,
                            "transfer_root_scale_ratio": float(
                                args.root_scale_ratio
                            ),
                            "parent_pose_basis": (
                                "strict_final_pose" if use_final_parent else "seed_pose"
                            ),
                            "local_yaw_degrees": float(yaw),
                            "local_common_z_mm": float(z_mm),
                            "local_left_radial_mm": float(left_radial_mm),
                            "local_right_radial_mm": float(right_radial_mm),
                        }
                    )

    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_candidates),
        "right_q": torch.stack(right_candidates),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save([derived], args.output)
    audit = {
        "method": args.candidate_method,
        "source_dataset": str(args.source_dataset.resolve()),
        "source_sample_index": args.sample_index,
        "source_object_name": args.object,
        "object_name": target_object,
        "root_scale_ratio": args.root_scale_ratio,
        "preserve_root_surface_offset": args.preserve_root_surface_offset,
        "source_mesh": (
            str(args.source_mesh.resolve()) if args.source_mesh else None
        ),
        "yaw_degrees": args.yaw_degrees,
        "z_mm": args.z_mm,
        "left_radial_mm": args.left_radial_mm,
        "right_radial_mm": args.right_radial_mm,
        "num_candidates": len(left_candidates),
        "output": str(args.output.resolve()),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"saved={args.output.resolve()} candidates={len(left_candidates)}")


if __name__ == "__main__":
    main()
