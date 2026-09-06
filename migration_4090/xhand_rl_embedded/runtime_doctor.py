#!/usr/bin/env python3
"""Validate v2 USD assets after starting Isaac/Omniverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--kit-portable-root", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
from .launcher import configure_isolated_kit

configure_isolated_kit(args, label="runtime_doctor")
simulation_app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics

from .contracts import validate_manifest


def _counts(path: Path) -> dict[str, int]:
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"cannot open USD: {path}")
    result = {"articulations": 0, "rigid_bodies": 0, "collisions": 0, "meshes": 0, "instances": 0}
    for prim in stage.Traverse():
        result["articulations"] += int(prim.HasAPI(UsdPhysics.ArticulationRootAPI))
        result["rigid_bodies"] += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        result["collisions"] += int(prim.HasAPI(UsdPhysics.CollisionAPI))
        result["meshes"] += int(prim.IsA(UsdGeom.Mesh))
        result["instances"] += int(prim.IsInstance())
    return result


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    checks = {}
    assets = {"robot": Path(manifest["robot_usd"]["path"])}
    assets.update({name: Path(row["usd"]["path"]) for name, row in manifest["objects"].items()})
    for name, path in assets.items():
        counts = _counts(path)
        if name == "robot":
            passed = counts["articulations"] >= 1 and counts["rigid_bodies"] >= 1
        else:
            passed = counts["rigid_bodies"] >= 1
        passed = passed and counts["collisions"] >= 1 and counts["meshes"] >= 1 and counts["instances"] == 0
        checks[name] = {"pass": passed, "counts": counts}
    result = {
        "schema": "xhand_rl_embedded_runtime_doctor_v2",
        "checks": checks,
        "ready": all(bool(row["pass"]) for row in checks.values()),
    }
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
    simulation_app.close()
