#!/usr/bin/env python3
"""Assemble a candidate-conditioned pregrasp bank without changing grasps."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from migration_4090.xhand_bodex_bimanual.contracts import (
    load_bodex_bank,
    sha256_file,
    validate_bodex_bank,
)


SCHEMA = "xhand_bodex_candidate_conditioned_pregrasp_bank_v1"
IMMUTABLE_SAMPLE_FIELDS = (
    "candidate_id",
    "joint_names",
    "full_body_q",
    "lift_full_body_q",
    "bodex_result",
)


def verify_compatible_source(reference: dict, candidate: dict, index: int) -> None:
    """Reject any source that changes BODex identity or final trajectories."""

    for field in IMMUTABLE_SAMPLE_FIELDS:
        if candidate.get(field) != reference.get(field):
            raise RuntimeError(
                f"candidate {index} source changes immutable field {field}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument(
        "--candidate-pregrasp-bank",
        type=Path,
        nargs="+",
        required=True,
        help="one compatible bank per candidate, in the fixed candidate order",
    )
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    reference_path = args.reference_bank.resolve()
    reference = load_bodex_bank(
        reference_path,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    if len(args.candidate_pregrasp_bank) != len(reference["samples"]):
        raise ValueError(
            "candidate pregrasp source count must match reference candidate count"
        )

    sources = [
        load_bodex_bank(
            path.resolve(),
            expected_object=args.object,
            verify_source=False,
            intended_stage=2,
        )
        for path in args.candidate_pregrasp_bank
    ]
    for path, payload in zip(args.candidate_pregrasp_bank, sources):
        if len(payload["samples"]) != len(reference["samples"]):
            raise RuntimeError(f"candidate source has wrong bank size: {path}")

    output = copy.deepcopy(reference)
    rows = []
    for index, reference_sample in enumerate(reference["samples"]):
        source_path = args.candidate_pregrasp_bank[index].resolve()
        source_sample = sources[index]["samples"][index]
        verify_compatible_source(reference_sample, source_sample, index)
        selected_pregrasp = list(
            source_sample.get(
                "controller_pregrasp_full_body_q",
                source_sample["pregrasp_full_body_q"],
            )
        )
        output["samples"][index]["pregrasp_full_body_q"] = selected_pregrasp
        controller_pregrasp_overridden = (
            "controller_grasp_full_body_q" in output["samples"][index]
        )
        if controller_pregrasp_overridden:
            # Keep the reference bank's already-audited controller grasp, but
            # let each candidate start from the empirically safer pregrasp.
            # The mixed path remains non-promotional until the authoritative
            # PhysX smoke passes.
            output["samples"][index][
                "controller_pregrasp_full_body_q"
            ] = list(selected_pregrasp)
        output["samples"][index]["candidate_conditioned_pregrasp_source"] = {
            "schema": SCHEMA,
            "source_bank": str(source_path),
            "source_bank_sha256": sha256_file(source_path),
            "controller_pregrasp_overridden": controller_pregrasp_overridden,
            "source_pregrasp_derivation": source_sample.get(
                "pregrasp_derivation"
            ),
        }
        rows.append(
            {
                "candidate_index": index,
                "candidate_id": reference_sample["candidate_id"],
                "source_bank": str(source_path),
                "source_bank_sha256": sha256_file(source_path),
                "requested_standoff_m": (
                    source_sample.get("pregrasp_derivation", {}).get(
                        "requested_standoff_m", 0.0
                    )
                ),
                "controller_pregrasp_overridden": controller_pregrasp_overridden,
            }
        )

    output["candidate_conditioned_pregrasp_derivation"] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_bank": str(reference_path),
        "reference_bank_sha256": sha256_file(reference_path),
        "final_bodex_grasps_unchanged": True,
        "candidate_rows": rows,
        "non_promotional_until_isaac_validated": True,
    }
    validate_bodex_bank(
        output,
        expected_object=args.object,
        verify_source=False,
        intended_stage=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.output)
    report = args.output.with_suffix(".json")
    report.write_text(
        json.dumps(output["candidate_conditioned_pregrasp_derivation"], indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output),
                "report": str(report.resolve()),
                "candidate_rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
