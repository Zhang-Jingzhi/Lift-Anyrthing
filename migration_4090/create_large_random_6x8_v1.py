#!/usr/bin/env python3
"""Build the shared six-object, eight-size formal generation catalogue."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


OBJECTS = (
    ("ycb", "bleach_cleanser"),
    ("ycb", "cracker_box"),
    ("ycb", "pitcher_base"),
    ("contactdb", "piggy_bank"),
    ("ycb", "power_drill"),
    ("ycb", "toy_airplane"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--target-longest-extent-mm", type=float, nargs=2, default=(450.0, 620.0)
    )
    parser.add_argument("--target-mass-kg", type=float, default=0.25)
    parser.add_argument("--family-seed", type=int, default=20260900)
    parser.add_argument("--schedule-seed", type=int, default=20261000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    derived_root = repo / "migration_4090/derived/large_random_6x8_v1"
    result_root = repo / "migration_4090/results/large_random_6x8_v1"
    final_vis = derived_root / "source_vis_large_random_6x8_v1.pt"
    catalog_path = result_root / "catalog.json"
    schedule_path = result_root / "shared_schedule_6x100.json"
    if final_vis.exists() or catalog_path.exists() or schedule_path.exists():
        raise RuntimeError("refusing to overwrite formal family outputs")
    derived_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    family_script = repo / "migration_4090/create_random_scaled_object_family.py"
    current_vis = repo / "data/bimanual/source_vis.pt"
    manifests = []
    for object_index, (dataset, source_object) in enumerate(OBJECTS):
        prefix = f"{source_object}_formal_large_random_v1"
        manifest = result_root / f"{dataset}_{source_object}_family.json"
        output_vis = (
            final_vis
            if object_index == len(OBJECTS) - 1
            else derived_root / f"stage_{object_index + 1:02d}.pt"
        )
        subprocess.run(
            [
                sys.executable,
                str(family_script),
                "--repo",
                str(repo),
                "--dataset",
                dataset,
                "--source-object",
                source_object,
                "--source-vis",
                str(current_vis),
                "--target-prefix",
                prefix,
                "--count",
                str(args.count),
                "--seed",
                str(args.family_seed + object_index),
                "--isotropic",
                "--target-longest-extent-mm",
                *(str(value) for value in args.target_longest_extent_mm),
                "--output-vis",
                str(output_vis),
                "--manifest-json",
                str(manifest),
            ],
            cwd=repo,
            check=True,
        )
        current_vis = output_vis
        manifests.append(manifest)

    catalog_rows = []
    for manifest_path in manifests:
        family = json.loads(manifest_path.read_text())
        for row in family["objects"]:
            dataset, object_name = row["object_name"].split("+", 1)
            collision_path = (
                repo
                / "data/data_urdf/object"
                / dataset
                / object_name
                / "coacd_allinone.obj"
            )
            collision = trimesh.load_mesh(
                collision_path, force="mesh", process=False
            )
            collision_volume_m3 = float(abs(collision.volume))
            if collision_volume_m3 <= 0.0:
                raise RuntimeError(f"non-positive collision volume: {collision_path}")
            catalog_rows.append(
                {
                    **row,
                    "base_object_name": family["source_object"],
                    "family_seed": family["seed"],
                    "collision_volume_m3": collision_volume_m3,
                    "target_mass_kg": args.target_mass_kg,
                    "density_kg_m3": args.target_mass_kg / collision_volume_m3,
                    "collision_path": str(collision_path),
                }
            )

    catalogue = {
        "schema": "tro_grasp_large_random_shared_catalog_v1",
        "methods": ["baseline", "bidex_v3"],
        "base_object_count": len(OBJECTS),
        "sizes_per_object": args.count,
        "target_longest_extent_mm": list(args.target_longest_extent_mm),
        "target_mass_kg": args.target_mass_kg,
        "scaling": "isotropic",
        "source_vis": str(final_vis),
        "objects": catalog_rows,
    }
    catalog_path.write_text(json.dumps(catalogue, indent=2) + "\n")

    schedule = []
    for object_index, (dataset, source_object) in enumerate(OBJECTS):
        rng = np.random.default_rng(args.schedule_seed + object_index)
        size_indices = np.resize(np.arange(args.count), 100)
        rng.shuffle(size_indices)
        for sample_index, size_index in enumerate(size_indices.tolist()):
            object_name = (
                f"{dataset}+{source_object}_formal_large_random_v1_{size_index:03d}"
            )
            catalogue_row = next(
                row for row in catalog_rows if row["object_name"] == object_name
            )
            schedule.append(
                {
                    "base_object_name": f"{dataset}+{source_object}",
                    "sample_index": sample_index,
                    "sample_seed": args.schedule_seed + object_index * 1000 + sample_index,
                    "size_index": size_index,
                    "object_name": object_name,
                    "scale_from_source": catalogue_row["scale_from_source"],
                    "extents_mm": catalogue_row["extents_mm"],
                    "density_kg_m3": catalogue_row["density_kg_m3"],
                    "target_mass_kg": args.target_mass_kg,
                }
            )
    schedule_payload = {
        "schema": "tro_grasp_large_random_shared_schedule_6x100_v1",
        "shared_by_methods": ["baseline", "bidex_v3"],
        "schedule_seed": args.schedule_seed,
        "entries": schedule,
    }
    schedule_path.write_text(json.dumps(schedule_payload, indent=2) + "\n")
    print(json.dumps({
        "catalog": str(catalog_path),
        "schedule": str(schedule_path),
        "source_vis": str(final_vis),
        "assets": len(catalog_rows),
        "scheduled_samples_per_method": len(schedule),
    }, indent=2))


if __name__ == "__main__":
    main()
