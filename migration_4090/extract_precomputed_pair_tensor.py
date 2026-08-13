#!/usr/bin/env python3
"""Extract left/right pair tensors from one precomputed candidate payload."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--left-output", type=Path, required=True)
    parser.add_argument("--right-output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    entry = payload[0] if isinstance(payload, list) else payload
    if "precomputed_bimanual_candidates" in entry:
        pairs = entry["precomputed_bimanual_candidates"]
        left_q = pairs["left_q"]
        right_q = pairs["right_q"]
    elif "left_q_final" in entry and "right_q_final" in entry:
        left_q = entry["left_q_final"]
        right_q = entry["right_q_final"]
    else:
        raise KeyError("No precomputed or realized left/right tensor pair")
    args.left_output.parent.mkdir(parents=True, exist_ok=True)
    args.right_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(left_q, args.left_output)
    torch.save(right_q, args.right_output)
    print(f"extracted={len(left_q)}")


if __name__ == "__main__":
    main()
