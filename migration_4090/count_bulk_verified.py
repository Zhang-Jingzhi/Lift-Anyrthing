#!/usr/bin/env python3
"""Count unique repeat-verified poses below one object's seed root."""

import argparse
from pathlib import Path

import torch

from select_bulk_repeat_candidates import pose_key
from formal_large_random_protocol import is_formal_protocol_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    keys = set()
    raw = 0
    files = sorted(args.input_root.glob("seed_*/repeat_verified/verified_dataset.pt"))
    accepted_files = 0
    skipped_files = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "large_random_6x100_v1" in str(args.input_root):
            if not is_formal_protocol_manifest(payload.get("manifest", {})):
                skipped_files += 1
                continue
        accepted_files += 1
        for sample in payload.get("samples", []):
            raw += 1
            keys.add(pose_key(sample))
    print(len(keys))
    print(
        f"files={accepted_files} skipped_protocol={skipped_files} "
        f"raw={raw} unique={len(keys)}",
        file=__import__("sys").stderr,
    )


if __name__ == "__main__":
    main()
