#!/usr/bin/env python3
"""Compose a four-mode BODex bank from hash-audited source candidates."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from migration_4090.xhand_bodex_bimanual.contracts import (
    load_bodex_bank,
    sha256_file,
    validate_bodex_bank,
)
from migration_4090.xhand_bodex_bimanual.diversity import (
    descriptor_distance,
    grasp_descriptor,
)


SCHEMA = "xhand_bodex_candidate_composition_v1"


def minimum_pairwise_descriptor_distance(samples: list[dict]) -> float:
    if len(samples) < 2:
        raise ValueError("candidate composition requires at least two samples")
    distances = [
        descriptor_distance(
            grasp_descriptor(samples[left]),
            grasp_descriptor(samples[right]),
        )
        for left in range(len(samples))
        for right in range(left + 1, len(samples))
    ]
    return min(distances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, action="append", required=True)
    parser.add_argument("--source-index", type=int, action="append", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--minimum-descriptor-distance", type=float, default=0.035)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.source_bank) != len(args.source_index):
        raise ValueError("source-bank and source-index counts must match")
    if len(args.source_bank) != 4:
        raise ValueError("the fixed staged policy ABI requires exactly four candidates")
    if args.minimum_descriptor_distance < 0.0:
        raise ValueError("minimum descriptor distance must be non-negative")

    reference_path = args.reference_bank.resolve()
    reference = load_bodex_bank(
        reference_path,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    selected = []
    rows = []
    for output_index, (source_path_arg, source_index) in enumerate(
        zip(args.source_bank, args.source_index)
    ):
        source_path = source_path_arg.resolve()
        source = load_bodex_bank(
            source_path,
            expected_object=args.object,
            verify_source=False,
            intended_stage=2,
        )
        if not 0 <= source_index < len(source["samples"]):
            raise IndexError(
                f"source index {source_index} is outside {source_path}"
            )
        sample = copy.deepcopy(source["samples"][source_index])
        selected.append(sample)
        rows.append(
            {
                "output_candidate_index": output_index,
                "candidate_id": sample["candidate_id"],
                "source_bank": str(source_path),
                "source_bank_sha256": sha256_file(source_path),
                "source_candidate_index": source_index,
            }
        )
    identifiers = [row["candidate_id"] for row in selected]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("composed candidate identifiers must be unique")
    minimum_distance = minimum_pairwise_descriptor_distance(selected)
    if minimum_distance < args.minimum_descriptor_distance:
        raise RuntimeError(
            "candidate composition violates diversity gate: "
            f"minimum descriptor distance {minimum_distance:.6f} < "
            f"{args.minimum_descriptor_distance:.6f}"
        )
    output = copy.deepcopy(reference)
    output["created_at"] = datetime.now(timezone.utc).isoformat()
    output["samples"] = selected
    output["candidate_composition_provenance"] = {
        "schema": SCHEMA,
        "created_at": output["created_at"],
        "reference_bank": str(reference_path),
        "reference_bank_sha256": sha256_file(reference_path),
        "candidate_rows": rows,
        "minimum_pairwise_descriptor_distance": minimum_distance,
        "required_minimum_descriptor_distance": args.minimum_descriptor_distance,
        "joint_bimanual_bodex_candidates_only": True,
        "non_promotional_until_isaac_validated": True,
    }
    validate_bodex_bank(
        output,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = args.output.with_suffix(".json")
    report.write_text(
        json.dumps(output["candidate_composition_provenance"], indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "candidate_rows": rows,
                "minimum_pairwise_descriptor_distance": minimum_distance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
