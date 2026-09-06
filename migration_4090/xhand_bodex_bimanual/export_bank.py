#!/usr/bin/env python3
"""Materialize validated BODex solver rows as a diverse RL candidate bank."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .contracts import (
    BODEX_BACKEND,
    BODEX_BANK_SCHEMA,
    BODEX_CURRICULUM_BANK_SCHEMA,
    sha256_file,
    validate_bodex_bank,
    verify_bodex_checkout,
)
from .diversity import diverse_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver-results",
        type=Path,
        action="append",
        required=True,
        help="repeat to merge multiple compatible BODex solver-result shards",
    )
    parser.add_argument("--bodex-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-descriptor-distance", type=float, default=0.035)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source = verify_bodex_checkout(args.bodex_root)
    raw_payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.solver_results
    ]
    raw = raw_payloads[0]
    rows = []
    for path, shard in zip(args.solver_results, raw_payloads):
        if not isinstance(shard, dict):
            raise RuntimeError(f"solver-results is not a mapping: {path}")
        for key in (
            "object",
            "stage_seed_profile",
            "supported_curriculum_stages",
            "requires_lift_ik",
        ):
            if shard.get(key) != raw.get(key):
                raise RuntimeError(f"incompatible solver-results field {key}: {path}")
        for sample in shard.get("samples", []):
            normalized = copy.deepcopy(sample)
            normalized["solver_result_source"] = str(path.resolve())
            normalized["source_candidate_id"] = str(sample.get("candidate_id", "candidate"))
            normalized["candidate_id"] = f"{path.stem}:{normalized['source_candidate_id']}"
            rows.append(normalized)
    if not rows:
        raise RuntimeError("solver-results contains no BODex samples")
    selected = diverse_subset(
        rows,
        minimum_distance=args.minimum_descriptor_distance,
        limit=args.limit,
    )
    stage_seed_profile = int(raw.get("stage_seed_profile", -1))
    supported_stages = raw.get("supported_curriculum_stages")
    requires_lift_ik = bool(raw.get("requires_lift_ik"))
    if not 1 <= stage_seed_profile <= 6:
        raise RuntimeError("solver-results lacks a valid stage_seed_profile")
    if not isinstance(supported_stages, list) or not supported_stages:
        raise RuntimeError("solver-results lacks supported_curriculum_stages")
    if stage_seed_profile <= 3:
        schema = BODEX_CURRICULUM_BANK_SCHEMA
        if requires_lift_ik:
            raise RuntimeError("pre-lift solver results cannot require lift IK")
    else:
        schema = BODEX_BANK_SCHEMA
        if not requires_lift_ik:
            raise RuntimeError("lift-ready solver results must require lift IK")
    payload = {
        "schema": schema,
        "generation_backend": BODEX_BACKEND,
        "object": args.object,
        "stage_seed_profile": stage_seed_profile,
        "supported_curriculum_stages": supported_stages,
        "requires_lift_ik": requires_lift_ik,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "solver_results": [str(path.resolve()) for path in args.solver_results],
        "solver_results_sha256": {
            str(path.resolve()): sha256_file(path) for path in args.solver_results
        },
        "minimum_descriptor_distance": args.minimum_descriptor_distance,
        "raw_count": len(rows),
        "selected_count": len(selected),
        "bodex_provenance": {
            **source,
            "joint_bimanual_optimization": True,
            "combined_grasp_matrix": True,
            "independent_single_hand_pairing": False,
            "paired_surface_seed_generator": "xhand_bodex_paired_surface_seeds_v1",
            "side_aware_pressure_constraints": True,
        },
        "samples": selected,
    }
    validate_bodex_bank(payload, expected_object=args.object, verify_source=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(payload, args.output)
    summary = args.output.with_suffix(".json")
    summary.write_text(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "schema",
                    "generation_backend",
                    "object",
                    "stage_seed_profile",
                    "supported_curriculum_stages",
                    "requires_lift_ik",
                    "created_at",
                    "solver_results",
                    "solver_results_sha256",
                    "minimum_descriptor_distance",
                    "raw_count",
                    "selected_count",
                    "bodex_provenance",
                )
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output.resolve()), "selected_count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
