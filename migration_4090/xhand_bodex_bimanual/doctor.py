#!/usr/bin/env python3
"""Read-only readiness audit for the BODex-to-staged-RL boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from .contracts import BODexContractError, verify_bodex_checkout


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bodex-root", type=Path, required=True)
    parser.add_argument("--xhand-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checks: dict[str, object] = {}
    try:
        checks["source"] = verify_bodex_checkout(args.bodex_root)
    except BODexContractError as exc:
        checks["source_error"] = str(exc)
    checks["xhand_urdf"] = {
        "path": str(args.xhand_urdf.resolve()),
        "exists": args.xhand_urdf.is_file(),
    }
    modules = (
        "torch",
        "curobo",
        "torch_scatter",
        "coal",
        "coal_openmp_wrapper",
        "pxr",
        "trimesh",
    )
    checks["python"] = sys.executable
    checks["modules"] = {name: _module_available(name) for name in modules}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    checks["gpus"] = gpu
    ready = (
        "source" in checks
        and bool(checks["xhand_urdf"]["exists"])
        and all(checks["modules"].values())
        and bool(gpu)
    )
    report = {"schema": "xhand_bodex_bimanual_doctor_v1", "ready": ready, "checks": checks}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
