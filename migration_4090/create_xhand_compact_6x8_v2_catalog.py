#!/usr/bin/env python3
"""Create the shared compact six-object XHand size catalogue.

The source meshes are never modified.  Each derived mesh is uniformly scaled,
centred, and given a deterministic tabletop orientation whose longest axis is
the opposed-hand grasp axis (world Y) and whose thinnest remaining axis is Z.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


OBJECTS = (
    ("sphere", "contactdb", "sphere_large_xhand_290mm_v1", 0.270, 0.310),
    ("cube", "contactdb", "cube_xlarge", 0.240, 0.290),
    ("cracker", "ycb", "cracker_box", 0.300, 0.380),
    ("bleach", "ycb", "bleach_cleanser", 0.300, 0.380),
    ("pitcher", "ycb", "pitcher_base", 0.280, 0.350),
    ("drill", "ycb", "power_drill", 0.300, 0.380),
)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
        raise RuntimeError(f"invalid source mesh: {path}")
    return mesh


def canonical_mesh(source: trimesh.Trimesh, target_y: float):
    mesh = source.copy()
    extents = np.asarray(mesh.extents, dtype=np.float64)
    long_axis = int(np.argmax(extents))
    remaining = [axis for axis in range(3) if axis != long_axis]
    vertical_axis = remaining[int(np.argmin(extents[remaining]))]
    forward_axis = next(axis for axis in remaining if axis != vertical_axis)
    rotation = np.zeros((3, 3), dtype=np.float64)
    rotation[0, forward_axis] = 1.0
    rotation[1, long_axis] = 1.0
    rotation[2, vertical_axis] = 1.0
    if np.linalg.det(rotation) < 0:
        rotation[0] *= -1.0
    vertices = (np.asarray(mesh.vertices) - mesh.bounds.mean(axis=0)) @ rotation.T
    scale = float(target_y / extents[long_axis])
    mesh.vertices = vertices * scale
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh, rotation, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--sizes", type=int, default=8)
    parser.add_argument("--family-seed", type=int, default=2026081001)
    parser.add_argument("--schedule-seed", type=int, default=2026081101)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "migration_4090/derived/xhand_compact_6x8_v2"
    result = repo / "migration_4090/results/xhand_compact_6x8_v2"
    catalog_path = result / "catalog.json"
    schedule_path = result / "shared_schedule_6x100.json"
    if out.exists() or catalog_path.exists() or schedule_path.exists():
        raise RuntimeError("refusing to overwrite xhand_compact_6x8_v2 outputs")
    out.mkdir(parents=True)
    result.mkdir(parents=True)

    rows = []
    for object_index, (key, dataset, source_name, low, high) in enumerate(OBJECTS):
        source_path = (
            repo / "data/data_urdf/object" / dataset / source_name / "coacd_allinone.obj"
        )
        source = load_mesh(source_path)
        rng = np.random.default_rng(args.family_seed + object_index)
        targets = np.linspace(low, high, args.sizes)
        targets += rng.uniform(-0.0025, 0.0025, size=args.sizes)
        targets = np.clip(targets, low, high)
        for size_index, target_y in enumerate(targets):
            mesh, rotation, scale = canonical_mesh(source, float(target_y))
            object_name = f"{dataset}+{source_name}_xhand_compact_v2_{size_index:03d}"
            mesh_dir = out / key / f"size_{size_index:03d}"
            mesh_dir.mkdir(parents=True)
            mesh_path = mesh_dir / "collision.obj"
            mesh.export(mesh_path)
            checked = load_mesh(mesh_path)
            if abs(float(checked.extents[1]) - float(target_y)) > 2e-5:
                raise RuntimeError(f"derived size mismatch: {mesh_path}")
            rows.append(
                {
                    "key": key,
                    "dataset": dataset,
                    "source_object_name": f"{dataset}+{source_name}",
                    "object_name": object_name,
                    "size_index": size_index,
                    "mesh_path": str(mesh_path),
                    "source_mesh_path": str(source_path),
                    "source_extents_m": np.asarray(source.extents).tolist(),
                    "canonical_rotation": rotation.tolist(),
                    "uniform_scale": scale,
                    "extents_m": np.asarray(checked.extents).tolist(),
                    "watertight": bool(checked.is_watertight),
                }
            )

    catalog = {
        "schema": "xhand_compact_shared_catalog_v2",
        "methods": ["baseline", "bidex_v3"],
        "object_count": len(OBJECTS),
        "sizes_per_object": args.sizes,
        "scaling": "uniform",
        "object_x_m": 0.50,
        "support_center_xyz_m": [0.50, 0.0, 0.70],
        "support_size_xy_m": [0.60, 0.80],
        "table_top_z_m": 0.71,
        "objects": rows,
    }
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")

    schedule_rows = []
    for object_index, (key, dataset, source_name, _, _) in enumerate(OBJECTS):
        rng = np.random.default_rng(args.schedule_seed + object_index)
        sizes = np.resize(np.arange(args.sizes), 100)
        rng.shuffle(sizes)
        for sample_index, size_index in enumerate(sizes.tolist()):
            row = next(
                item
                for item in rows
                if item["key"] == key and item["size_index"] == size_index
            )
            schedule_rows.append(
                {
                    "key": key,
                    "base_object_name": f"{dataset}+{source_name}",
                    "sample_index": sample_index,
                    "size_index": size_index,
                    "object_name": row["object_name"],
                    "sample_seed": args.schedule_seed + object_index * 1000 + sample_index,
                }
            )
    schedule_path.write_text(
        json.dumps(
            {
                "schema": "xhand_compact_shared_schedule_6x100_v2",
                "shared_by_methods": ["baseline", "bidex_v3"],
                "entries": schedule_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"catalog": str(catalog_path), "schedule": str(schedule_path), "assets": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
