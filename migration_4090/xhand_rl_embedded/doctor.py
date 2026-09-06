#!/usr/bin/env python3
"""Read-only dependency and contract check for v2."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

from .contracts import validate_manifest


def check(manifest_path: Path) -> dict:
    checks: dict[str, dict[str, object]] = {}
    try:
        validate_manifest(json.loads(manifest_path.read_text()), verify_files=True)
        checks["manifest_and_asset_hashes"] = {"pass": True}
    except Exception as error:
        checks["manifest_and_asset_hashes"] = {"pass": False, "error": str(error)}
    for module in ("isaacsim", "isaaclab", "isaaclab_rl", "rsl_rl", "torch", "numpy", "scipy"):
        spec = importlib.util.find_spec(module)
        checks[f"module:{module}"] = {"pass": spec is not None, "origin": None if spec is None else spec.origin}
    for package in ("isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch", "numpy", "scipy"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks[f"package:{package}"] = {"pass": version is not None, "version": version}
    return {
        "schema": "xhand_rl_embedded_doctor_v2",
        "python": sys.executable,
        "manifest": str(manifest_path.resolve()),
        "checks": checks,
        "ready": all(bool(row["pass"]) for row in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.manifest)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
