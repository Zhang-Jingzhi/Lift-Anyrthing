#!/usr/bin/env python3
"""Merge audited staged-RL success shards into a balanced grasp dataset."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


SHARD_SCHEMA = "xhand_bodex_staged_success_state_shard_v1"
DATASET_SCHEMA = "xhand_bodex_staged_balanced_dataset_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantized_key(sample: dict[str, Any], quantum_rad: float) -> bytes:
    q = torch.as_tensor(sample["final_full_body_q"], dtype=torch.float32)
    if q.shape != (38,) or not torch.isfinite(q).all():
        raise RuntimeError("success sample must contain one finite 38D joint state")
    quantized = torch.round(q / quantum_rad).to(torch.int32).numpy()
    return quantized.tobytes()


def merge_shards(
    paths: Iterable[Path],
    *,
    target_count: int,
    quantum_rad: float,
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if quantum_rad <= 0.0:
        raise ValueError("quantum_rad must be positive")
    paths = sorted(Path(path) for path in paths)
    if not paths:
        raise RuntimeError("no success shards were supplied")

    invariants: dict[str, Any] | None = None
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[bytes] = set()
    raw_count = 0
    duplicate_count = 0
    sources = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SHARD_SCHEMA:
            raise RuntimeError(f"unexpected success shard schema: {path}")
        current = {
            key: payload.get(key)
            for key in (
                "object",
                "stage",
                "stage_spec",
                "checkpoint_sha256",
                "bodex_bank_sha256",
                "candidate_selection",
                "candidate_repeat_factors",
                "fixed_candidate_index",
                "nominal_pose_lock",
                "formal_active_action_group",
                "active_action_group",
                "residual_activation_phase",
                "residual_activation_close_fraction",
                "hand_residual_activation_phase",
                "hand_residual_activation_close_fraction",
                "residual_integration",
                "residual_limit_rad",
            )
        }
        if invariants is None:
            invariants = current
        elif current != invariants:
            raise RuntimeError(f"incompatible success shard: {path}")
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
                "episodes": int(payload["episodes"]),
                "successes": int(payload["successes"]),
            }
        )
        for offset, sample in enumerate(payload.get("samples", [])):
            raw_count += 1
            key = _quantized_key(sample, quantum_rad)
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            normalized = dict(sample)
            normalized["source_shard"] = str(path.resolve())
            normalized["source_shard_offset"] = offset
            buckets[int(sample["candidate_index"])].append(normalized)

    selected: list[dict[str, Any]] = []
    offsets = {candidate: 0 for candidate in buckets}
    candidates = sorted(buckets)
    while len(selected) < target_count:
        advanced = False
        for candidate in candidates:
            offset = offsets[candidate]
            if offset >= len(buckets[candidate]):
                continue
            selected.append(buckets[candidate][offset])
            offsets[candidate] += 1
            advanced = True
            if len(selected) >= target_count:
                break
        if not advanced:
            break
    if len(selected) < target_count:
        raise RuntimeError(
            f"only {len(selected)} quantized-unique successes; requested {target_count}"
        )

    selected_counts: dict[int, int] = defaultdict(int)
    for sample in selected:
        selected_counts[int(sample["candidate_index"])] += 1
    assert invariants is not None
    return {
        "schema": DATASET_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **invariants,
        "target_count": target_count,
        "selected_count": len(selected),
        "raw_success_count": raw_count,
        "quantized_duplicate_count": duplicate_count,
        "joint_quantization_rad": quantum_rad,
        "selection": "round_robin_candidate_balanced_after_quantized_dedup",
        "candidate_available_counts": {
            str(key): len(value) for key, value in sorted(buckets.items())
        },
        "candidate_selected_counts": {
            str(key): value for key, value in sorted(selected_counts.items())
        },
        "source_shards": sources,
        "samples": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", default=[])
    parser.add_argument("--shard-glob")
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--joint-quantization-rad", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.shard)
    if args.shard_glob:
        paths.extend(Path(path) for path in glob.glob(args.shard_glob))
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = merge_shards(
        paths,
        target_count=args.target_count,
        quantum_rad=args.joint_quantization_rad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {key: value for key, value in payload.items() if key != "samples"}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
