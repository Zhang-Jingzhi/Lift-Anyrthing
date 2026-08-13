#!/usr/bin/env python3
"""Create a reproducible family of large, mildly anisotropic object scales."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--dataset", default="ycb")
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--target-prefix", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260857)
    parser.add_argument("--minimum-scale", type=float, default=0.85)
    parser.add_argument("--maximum-scale", type=float, default=1.15)
    parser.add_argument(
        "--isotropic",
        action="store_true",
        help="Sample one shared XYZ scale per object instead of three axis scales.",
    )
    parser.add_argument(
        "--minimum-extents-mm",
        type=float,
        nargs=3,
        default=(380.0, 370.0, 500.0),
    )
    parser.add_argument(
        "--target-longest-extent-mm",
        type=float,
        nargs=2,
        metavar=("MIN_MM", "MAX_MM"),
        help=(
            "Sample an isotropic scale whose resulting longest mesh extent "
            "lies in this interval. When set, --minimum-extents-mm and the "
            "ordinary scale interval are not used to choose the scale."
        ),
    )
    parser.add_argument("--output-vis", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.source_vis = args.source_vis.resolve()
    args.output_vis = args.output_vis.resolve()
    args.manifest_json = args.manifest_json.resolve()
    if args.count < 1:
        parser.error("--count must be positive")
    if not 0 < args.minimum_scale <= args.maximum_scale:
        parser.error("invalid scale interval")
    if args.target_longest_extent_mm is not None:
        low, high = args.target_longest_extent_mm
        if not 0 < low <= high:
            parser.error("invalid --target-longest-extent-mm interval")
        if not args.isotropic:
            parser.error("--target-longest-extent-mm requires --isotropic")
    if args.output_vis.exists() or args.manifest_json.exists():
        raise RuntimeError("refusing to overwrite family output")

    source_mesh_path = (
        args.repo
        / "data/data_urdf/object"
        / args.dataset
        / args.source_object
        / f"{args.source_object}.stl"
    )
    source_mesh = trimesh.load_mesh(source_mesh_path, force="mesh", process=False)
    source_extents_mm = np.asarray(source_mesh.extents) * 1000.0
    minimum_extents_mm = np.asarray(args.minimum_extents_mm, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    rows = []
    planned = []
    for index in range(args.count):
        required = minimum_extents_mm / source_extents_mm
        if args.target_longest_extent_mm is not None:
            target_longest_mm = rng.uniform(*args.target_longest_extent_mm)
            scale = np.full(3, target_longest_mm / source_extents_mm.max())
        elif args.isotropic:
            sampled_scalar = rng.uniform(args.minimum_scale, args.maximum_scale)
            scale = np.full(3, max(sampled_scalar, float(required.max())))
        else:
            sampled = rng.uniform(args.minimum_scale, args.maximum_scale, size=3)
            scale = np.maximum(sampled, required)
        target = f"{args.target_prefix}_{index:03d}"
        target_dir = args.repo / "data/data_urdf/object" / args.dataset / target
        if target_dir.exists():
            raise RuntimeError(f"refusing to overwrite {target_dir}")
        planned.append((target, scale))

    args.output_vis.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    create_script = args.repo / "scripts/create_scaled_bimanual_object.py"
    with tempfile.TemporaryDirectory(prefix="tro_scale_family_") as temp_dir:
        current_vis = args.source_vis
        for index, (target, scale) in enumerate(planned):
            next_vis = (
                args.output_vis
                if index == len(planned) - 1
                else Path(temp_dir) / f"family_{index:03d}.pt"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(create_script),
                    "--repo",
                    str(args.repo),
                    "--dataset",
                    args.dataset,
                    "--source-object",
                    args.source_object,
                    "--target-object",
                    target,
                    "--scale",
                    *(f"{value:.10f}" for value in scale),
                    "--source-vis",
                    str(current_vis),
                    "--output-vis",
                    str(next_vis),
                ],
                cwd=args.repo,
                check=True,
            )
            extents_mm = source_extents_mm * scale
            rows.append(
                {
                    "index": index,
                    "object_name": f"{args.dataset}+{target}",
                    "scale_from_source": scale.tolist(),
                    "extents_mm": extents_mm.tolist(),
                }
            )
            current_vis = next_vis

    manifest = {
        "method": (
            "deterministic_truncated_isotropic_uniform_scale_v2"
            if args.isotropic
            else "deterministic_truncated_anisotropic_uniform_scale_v1"
        ),
        "seed": args.seed,
        "source_object": f"{args.dataset}+{args.source_object}",
        "source_extents_mm": source_extents_mm.tolist(),
        "sampled_scale_interval": [args.minimum_scale, args.maximum_scale],
        "target_longest_extent_mm": args.target_longest_extent_mm,
        "minimum_extents_mm": minimum_extents_mm.tolist(),
        "objects": rows,
        "output_vis": str(args.output_vis),
    }
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
