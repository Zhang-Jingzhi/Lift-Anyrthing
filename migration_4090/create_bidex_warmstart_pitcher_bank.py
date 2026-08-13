#!/usr/bin/env python3
"""Create a BiDex-v3 local-refinement bank around a validated region center.

The source pose is a screened region center, not a final dataset import: the
remaining candidates receive paired local finger/wrist perturbations.  Keeping
the exact center as candidate 0 gives the smoke gate a deterministic anchor;
formal generation can discard that center and use the refined variants.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch


def perturb_q(sample, rng):
    names = list(sample["joint_names"])
    values = dict(zip(names, sample["full_body_q"]))
    # Paired perturbations retain the opposed two-hand structure.
    for joint in (
        "index_bend_joint", "index_joint1", "index_joint2", "mid_joint1",
        "mid_joint2", "ring_joint1", "ring_joint2", "pinky_joint1",
        "pinky_joint2", "thumb_bend_joint", "thumb_rota_joint1",
        "thumb_rota_joint2",
    ):
        delta = float(rng.uniform(-0.012, 0.012))
        values[f"left_hand_{joint}"] += delta
        values[f"right_hand_{joint}"] += delta
    for joint in ("j6", "j7"):
        delta = float(rng.uniform(-0.010, 0.010))
        values[f"left_{joint}"] += delta
        values[f"right_{joint}"] += delta
    sample["full_body_q"] = [float(values[n]) for n in names]
    if "pregrasp_full_body_q" in sample:
        values = dict(zip(names, sample["pregrasp_full_body_q"]))
        for joint in ("j6", "j7"):
            delta = float(rng.uniform(-0.004, 0.004))
            values[f"left_{joint}"] += delta
            values[f"right_{joint}"] += delta
        sample["pregrasp_full_body_q"] = [float(values[n]) for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--object", default="pitcher")
    parser.add_argument("--source-index", type=int, default=2)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    source = payload["samples"][args.source_index]
    samples = []
    for index in range(args.count):
        sample = copy.deepcopy(source)
        sample["method"] = "bidex_v3"
        sample["compact_v2"] = {
            **sample.get("compact_v2", {}),
            "candidate_index": index,
            "method": "bidex_v3",
            "paired_local_proposals": 12,
            "warm_start_source": str(args.source),
            "warm_start_source_index": args.source_index,
            "warm_start_exact_center": index == 0,
        }
        if index:
            perturb_q(sample, np.random.default_rng(args.seed + index))
        samples.append(sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "xhand_compact_candidate_bank_v3_bidex_warmstart",
            "object": args.object,
            "method": "bidex_v3",
            "size_index": source.get("object_transform", {}).get("size_index", 4),
            "samples": samples,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "count": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
