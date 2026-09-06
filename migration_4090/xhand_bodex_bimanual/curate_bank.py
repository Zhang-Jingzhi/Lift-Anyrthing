#!/usr/bin/env python3
"""Curate a diverse, physically robust subset of a genuine BODex bank."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .contracts import load_bodex_bank, sha256_file, validate_bodex_bank
from .diversity import descriptor_distance, diverse_subset, grasp_descriptor


def diagnostic_candidate_rates(payload: dict[str, Any]) -> dict[str, float]:
    rows: dict[str, list[bool]] = defaultdict(list)
    for report in payload.get("reports", []):
        candidate_id = report.get("candidate_id")
        if isinstance(candidate_id, str):
            rows[candidate_id].append(bool(report.get("stage_pass")))
    return {
        candidate_id: sum(outcomes) / len(outcomes)
        for candidate_id, outcomes in rows.items()
        if outcomes
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--minimum-success-rate", type=float, default=0.70)
    parser.add_argument("--minimum-descriptor-distance", type=float, default=0.035)
    parser.add_argument("--minimum-count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("minimum success rate must be in [0, 1]")
    if args.minimum_descriptor_distance < 0.0 or args.minimum_count <= 0:
        raise ValueError("diversity distance must be non-negative and count positive")

    bank = load_bodex_bank(
        args.input_bank,
        expected_object=args.object,
        verify_source=False,
        intended_stage=args.stage,
    )
    bank_hash = sha256_file(args.input_bank)
    rates_by_diagnostic: dict[str, dict[str, float]] = {}
    diagnostic_hashes: dict[str, str] = {}
    for path in args.diagnostic:
        payload = json.loads(path.read_text())
        if payload.get("object") != args.object or int(payload.get("stage", -1)) != args.stage:
            raise RuntimeError(f"diagnostic object/stage mismatch: {path}")
        if payload.get("bodex_bank_sha256") != bank_hash:
            raise RuntimeError(f"diagnostic was not evaluated against input bank: {path}")
        if payload.get("candidate_selection") != "round_robin":
            raise RuntimeError(f"diagnostic did not use round-robin candidates: {path}")
        rates_by_diagnostic[str(path.resolve())] = diagnostic_candidate_rates(payload)
        diagnostic_hashes[str(path.resolve())] = sha256_file(path)

    eligible = []
    minimum_rates: dict[str, float] = {}
    for sample in bank["samples"]:
        candidate_id = str(sample.get("candidate_id"))
        rates = [rows.get(candidate_id, 0.0) for rows in rates_by_diagnostic.values()]
        minimum_rates[candidate_id] = min(rates)
        if all(rate >= args.minimum_success_rate for rate in rates):
            eligible.append(copy.deepcopy(sample))
    selected = diverse_subset(
        eligible,
        minimum_distance=args.minimum_descriptor_distance,
    )
    if len(selected) < args.minimum_count:
        raise RuntimeError(
            f"only {len(selected)} candidates pass robustness/diversity; "
            f"minimum is {args.minimum_count}"
        )
    pairwise_distances = [
        descriptor_distance(grasp_descriptor(selected[i]), grasp_descriptor(selected[j]))
        for i in range(len(selected))
        for j in range(i + 1, len(selected))
    ]
    payload = copy.deepcopy(bank)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["samples"] = selected
    payload["curation_provenance"] = {
        "schema": "xhand_bodex_physics_robust_diverse_curation_v1",
        "input_bank": str(args.input_bank.resolve()),
        "input_bank_sha256": bank_hash,
        "diagnostics_sha256": diagnostic_hashes,
        "diagnostic_candidate_success_rates": rates_by_diagnostic,
        "selection_rule": "minimum_success_rate_in_every_round_robin_diagnostic",
        "minimum_success_rate": args.minimum_success_rate,
        "minimum_descriptor_distance": args.minimum_descriptor_distance,
        "eligible_candidate_count": len(eligible),
        "selected_candidate_count": len(selected),
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "selected_minimum_success_rates": {
            row["candidate_id"]: minimum_rates[row["candidate_id"]] for row in selected
        },
        "minimum_selected_pairwise_descriptor_distance": (
            min(pairwise_distances) if pairwise_distances else None
        ),
    }
    validate_bodex_bank(payload, expected_object=args.object, intended_stage=args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(payload["curation_provenance"], indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "selected_count": len(selected),
                "selected_candidate_ids": [row["candidate_id"] for row in selected],
                "minimum_pairwise_descriptor_distance": min(pairwise_distances),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
