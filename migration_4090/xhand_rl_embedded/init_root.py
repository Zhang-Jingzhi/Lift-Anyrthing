#!/usr/bin/env python3
"""Create or verify an isolated v2 output root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import (
    EMBEDDED_BACKEND,
    LOCKED_OBJECTS,
    OMITTED_EXPENSIVE_GATES,
    PPO_REVISION,
    REWARD_REVISION,
    ROOT_MARKER_SCHEMA,
    TRAINING_REWARD_SCALE,
    sha256_file,
    validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nominal-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, verify_files=True)
    root = args.root.resolve()
    forbidden = (
        "xhand_requested_six_same_x_strict_1200",
        "objectflow_xhand_six_strict",
        "locked600",
        "objectflow_xhand_rl_prod_20260823",
    )
    if any(fragment in str(root) for fragment in forbidden):
        raise RuntimeError("v2 root may not overlap a legacy or v1 namespace")
    marker = root / "EMBEDDED_RL_PIPELINE_ROOT.json"
    expected = {
        "schema": ROOT_MARKER_SCHEMA,
        "backend": EMBEDDED_BACKEND,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reward_revision": REWARD_REVISION,
        "ppo_revision": PPO_REVISION,
        "training_reward_scale": TRAINING_REWARD_SCALE,
        "nominal_root": str(args.nominal_root.resolve()),
        "objects": list(LOCKED_OBJECTS),
        "target_per_object": int(manifest["target_per_object"]),
        "acceptance_profile": "embedded_physics_pass",
        "external_candidate_replay_validator": False,
        "strict_visual_mesh_accepted": False,
        "omitted_expensive_gates": list(OMITTED_EXPENSIVE_GATES),
        "legacy_data_reused_as_accepted": False,
    }
    if root.exists() and any(root.iterdir()):
        if not marker.is_file():
            raise RuntimeError(f"refusing non-empty unmarked v2 root: {root}")
        if json.loads(marker.read_text()) != expected:
            raise RuntimeError("existing v2 marker differs from this invocation")
    else:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2) + "\n")
    for object_name in LOCKED_OBJECTS:
        nominal = args.nominal_root / f"{object_name}.pt"
        if not nominal.is_file():
            raise FileNotFoundError(f"missing nominal seed: {nominal}")
    for name in ("accepted", "runs", "logs"):
        (root / name).mkdir(exist_ok=True)
    print(marker)


if __name__ == "__main__":
    main()
