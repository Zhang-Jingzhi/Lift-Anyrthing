"""Append-only storage that rejects exact and near-duplicate final grasps."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

import torch

from migration_4090.xhand_bodex_bimanual.diversity import is_near_duplicate


def _stored_samples(root: Path, object_name: str) -> list[dict[str, Any]]:
    samples = []
    for path in sorted((root / object_name).glob("sample_*/sample.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples.append(payload["sample"])
    return samples


def store_diverse_stage6_success(
    *,
    root: Path,
    object_name: str,
    sample: dict[str, Any],
    report: dict[str, Any],
    minimum_descriptor_distance: float = 0.035,
) -> Path | None:
    if report.get("stage", {}).get("stage_id") != 6 or report.get("stage_pass") is not True:
        raise RuntimeError("only a passing Stage 6 trajectory may enter the final dataset")
    object_root = root / object_name
    object_root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".diversity.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _stored_samples(root, object_name)
        if is_near_duplicate(
            sample, existing, minimum_distance=minimum_descriptor_distance
        ):
            return None
        target = object_root / f"sample_{len(existing):06d}"
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        torch.save(
            {
                "schema": "xhand_bodex_staged_diverse_sample_v1",
                "sample": sample,
                "report": report,
                "minimum_descriptor_distance": minimum_descriptor_distance,
            },
            target / "sample.pt",
        )
        (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return target
