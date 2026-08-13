#!/usr/bin/env python3
"""Convert repeat-verified samples into reusable precomputed wrist seeds.

The formal verifier stores both the realized final hand pose and the exact
pre-closure seed.  Local candidate expansion must start from ``*_q_seed`` so
that the normal closure controller and every downstream strict gate are run
again, rather than replaying a post-contact realized pose.
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-source-vis", type=Path, required=True)
    parser.add_argument("--verified-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument(
        "--candidate-method",
        default="repeat_verified_seed_parent_v1",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    verified = torch.load(
        args.verified_dataset, map_location="cpu", weights_only=False
    )
    samples = verified.get("samples", [])
    if not samples:
        raise RuntimeError("verified dataset contains no samples")
    indices = args.indices if args.indices is not None else list(range(len(samples)))
    invalid = [index for index in indices if not 0 <= index < len(samples)]
    if invalid:
        raise IndexError(
            f"verified sample indices out of range: {invalid}; available={len(samples)}"
        )

    entries = torch.load(
        args.template_source_vis, map_location="cpu", weights_only=False
    )
    if not isinstance(entries, list) or len(entries) != 1:
        raise RuntimeError("expected exactly one template source-vis entry")
    entry = dict(entries[0])

    left_rows = []
    right_rows = []
    metadata = []
    for output_index, sample_index in enumerate(indices):
        sample = samples[sample_index]
        if sample.get("object_name") != entry.get("object_name"):
            raise ValueError(
                "verified/template object mismatch: "
                f"{sample.get('object_name')} != {entry.get('object_name')}"
            )
        metrics = sample.get("metrics", {})
        if metrics.get("repeat_success_rate") != 1.0:
            raise RuntimeError(
                f"sample {sample_index} is not repeat verified at 100%"
            )
        if int(metrics.get("repeat_count", 0)) != 3:
            raise RuntimeError(
                f"sample {sample_index} does not have three total rollouts"
            )
        if not metrics.get("repeat_bilateral_physical_contact_all_phases", False):
            raise RuntimeError(
                f"sample {sample_index} lacks bilateral physical contact"
            )

        left_q = sample["left_q_seed"].detach().cpu().clone()
        right_q = sample["right_q_seed"].detach().cpu().clone()
        if left_q.shape != (22,) or right_q.shape != (22,):
            raise RuntimeError(
                f"invalid seed shapes: {left_q.shape}, {right_q.shape}"
            )
        left_rows.append(left_q)
        right_rows.append(right_q)
        metadata.append(
            {
                "source_pair_index": output_index,
                "candidate_method": args.candidate_method,
                "verified_source_dataset": str(args.verified_dataset.resolve()),
                "verified_source_sample_index": sample_index,
                "verified_candidate_index": int(sample.get("candidate_index", -1)),
                "verified_repeat_count": int(metrics["repeat_count"]),
                "verified_repeat_success_rate": float(
                    metrics["repeat_success_rate"]
                ),
                "verified_repeat_max_6dir_mm": float(
                    metrics["repeat_max_6dir_mm"]
                ),
                "verified_repeat_max_gravity_mm": float(
                    metrics["repeat_max_gravity_mm"]
                ),
                "verified_repeat_min_lift_mm": float(
                    metrics["repeat_min_lift_mm"]
                ),
                "verified_bilateral_physical_contact_all_phases": True,
            }
        )

    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_rows),
        "right_q": torch.stack(right_rows),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(
        f"saved {len(metadata)} repeat-verified seed parents -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
