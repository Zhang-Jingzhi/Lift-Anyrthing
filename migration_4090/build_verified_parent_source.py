#!/usr/bin/env python3
"""Convert repeat-verified samples into an auditable precomputed source pool."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-dataset", type=Path, required=True)
    parser.add_argument("--template-source", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    verified = torch.load(
        args.verified_dataset, map_location="cpu", weights_only=False
    )
    samples = verified.get("samples", [])
    if not samples:
        raise RuntimeError("verified dataset has no samples")
    template_payload = torch.load(
        args.template_source, map_location="cpu", weights_only=False
    )
    matches = [
        entry for entry in template_payload if entry.get("object_name") == args.object
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one template entry for {args.object}")

    entry = dict(matches[0])
    metadata = []
    for output_index, sample in enumerate(samples):
        metrics = sample.get("metrics", {})
        metadata.append(
            {
                "source_pair_index": output_index,
                "candidate_method": "repeat_verified_high_v2_parent_v1",
                "verified_source_dataset": str(args.verified_dataset.resolve()),
                "verified_source_sample_index": output_index,
                "verified_candidate_index": int(sample.get("candidate_index", -1)),
                "verified_repeat_count": int(metrics.get("repeat_count", 0)),
                "verified_repeat_success_rate": float(
                    metrics.get("repeat_success_rate", 0.0)
                ),
                "verified_repeat_max_6dir_mm": float(
                    metrics.get("repeat_max_6dir_mm", float("inf"))
                ),
                "verified_repeat_max_gravity_mm": float(
                    metrics.get("repeat_max_gravity_mm", float("inf"))
                ),
                "verified_repeat_min_lift_mm": float(
                    metrics.get("repeat_min_lift_mm", float("-inf"))
                ),
                "verified_bilateral_physical_contact_all_phases": bool(
                    metrics.get(
                        "repeat_bilateral_physical_contact_all_phases", False
                    )
                ),
            }
        )
    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(
            [sample["left_q_seed"].detach().cpu() for sample in samples]
        ),
        "right_q": torch.stack(
            [sample["right_q_seed"].detach().cpu() for sample in samples]
        ),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(f"saved {len(samples)} verified parents -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
