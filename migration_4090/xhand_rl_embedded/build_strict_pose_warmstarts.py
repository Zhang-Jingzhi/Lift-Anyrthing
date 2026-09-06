#!/usr/bin/env python3
"""Create immutable RL-only warm starts from locked-mesh strict poses.

The copied poses are never accepted as production data.  They are used only
to put PPO in the contact basin for the resized sphere/pyramid meshes; the
embedded episode still has to discover and pass every formal gate before a
receipt can be materialized.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--sphere-small-source", type=Path, required=True)
    ap.add_argument("--pyramid-source", type=Path, required=True)
    ap.add_argument(
        "--legacy-source-dir",
        type=Path,
        help="include the four unchanged locked-mesh nominal warm starts from this directory",
    )
    args = ap.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    sources = {
        "sphere_small": args.sphere_small_source.resolve(),
        "pyramid": args.pyramid_source.resolve(),
    }
    if args.legacy_source_dir is not None:
        legacy_dir = args.legacy_source_dir.resolve()
        for object_name in ("sphere", "cracker_large", "cracker", "cube"):
            sources[object_name] = (legacy_dir / f"{object_name}.pt").resolve()
    index = {
        "schema": "xhand_rl_locked_mesh_strict_pose_warmstart_index_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "strict_validated": False,
        "formal_validation_pending": True,
        "legacy_data_reused_as_accepted": False,
        "objects": {},
    }

    for object_name, source_path in sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples", []) if isinstance(payload, dict) else []
        if not samples:
            raise RuntimeError(f"source has no samples: {source_path}")
        sample = copy.deepcopy(samples[0])
        row = manifest["objects"][object_name]
        sample["object_name"] = object_name
        sample["object_id"] = object_name
        sample["base_object_name"] = object_name
        sample["object_mesh_path"] = row["mesh"]["path"]
        sample["generation_backend"] = "legacy_derived_warm_start_only"
        sample["method"] = "rl_locked_mesh_strict_pose_warm_start"
        sample["strict_validated"] = False
        sample["formal_validation_pending"] = True
        sample["isaaclab_physical_validated"] = False
        sample["isaac_gym_physical_validated"] = False
        sample["strict_acceptance"] = False
        sample["legacy_data_reused_as_accepted"] = False
        sample["legacy_source_file"] = str(source_path)
        sample["legacy_source_sha256"] = sha256(source_path)
        sample["physical_parameters"] = {
            "target_mass_kg": float(manifest["physics"]["object_mass_kg"]),
            "friction": float(manifest["physics"]["friction"]),
            "source_legacy_physical_parameters": sample.get("physical_parameters"),
        }
        sample["warm_start_provenance"] = {
            "schema": "xhand_rl_locked_mesh_strict_pose_warm_start_v1",
            "source": str(source_path),
            "source_sha256": sha256(source_path),
            "not_an_accepted_sample": True,
            "formal_validation_pending": True,
        }
        output = {
            "schema": "xhand_rl_locked_mesh_strict_pose_warm_start_v1",
            "backend": "isaaclab_rsl_rl_ppo_embedded_physics_v2",
            "object": object_name,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "source_nominal": str(source_path),
            "source_nominal_sha256": sha256(source_path),
            "strict_validated": False,
            "formal_validation_pending": True,
            "legacy_data_reused_as_accepted": False,
            "samples": [sample],
        }
        destination = out / f"{object_name}.pt"
        torch.save(output, destination)
        index["objects"][object_name] = {
            "path": str(destination),
            "sha256": sha256(destination),
            "source": str(source_path),
            "source_sha256": sha256(source_path),
        }

    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
