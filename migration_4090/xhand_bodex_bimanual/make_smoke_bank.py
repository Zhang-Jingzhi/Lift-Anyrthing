#!/usr/bin/env python3
"""Build an explicitly non-BODex fixture for Isaac environment smoke tests only."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from .contracts import SMOKE_BACKEND, SMOKE_BANK_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=2)
    args = parser.parse_args()
    if args.copies <= 0:
        raise ValueError("copies must be positive")
    source = torch.load(args.nominal, map_location="cpu", weights_only=False)
    sample = copy.deepcopy(source["samples"][0])
    sample["object_name"] = args.object
    sample["generation_backend"] = SMOKE_BACKEND
    sample["test_fixture"] = True
    samples = []
    for index in range(args.copies):
        row = copy.deepcopy(sample)
        row["smoke_fixture_index"] = index
        samples.append(row)
    payload = {
        "schema": SMOKE_BANK_SCHEMA,
        "generation_backend": SMOKE_BACKEND,
        "object": args.object,
        "test_fixture": True,
        "training_allowed": False,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(payload, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
