#!/usr/bin/env python3
"""Convert strict first-pass samples into exploratory pre-closure seeds.

This helper is deliberately separate from ``extract_verified_seed_candidates``:
these parents are allowed to have only one strict rollout and therefore may be
used only to build a new candidate neighborhood.  Every descendant still has
to pass the normal full physics, ablation, and 3/3 repeat-verification gates.
"""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-source-vis", type=Path, required=True)
    parser.add_argument("--strict-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument(
        "--candidate-method",
        default="strict_first_pass_seed_parent_v1",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    payload = torch.load(
        args.strict_dataset, map_location="cpu", weights_only=False
    )
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError("strict dataset contains no samples")
    indices = args.indices if args.indices is not None else list(range(len(samples)))
    invalid = [index for index in indices if not 0 <= index < len(samples)]
    if invalid:
        raise IndexError(
            f"strict sample indices out of range: {invalid}; available={len(samples)}"
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
                "strict/template object mismatch: "
                f"{sample.get('object_name')} != {entry.get('object_name')}"
            )
        metrics = sample.get("metrics", {})
        required_true = (
            "strict_success",
            "isaac_success",
            "gravity_success",
            "realized_pose_pass",
            "realized_dual_contact_pass",
            "bimanual_required",
        )
        failed = [name for name in required_true if not metrics.get(name, False)]
        if failed:
            raise RuntimeError(
                f"sample {sample_index} is not a strict first-pass parent: {failed}"
            )
        if metrics.get("left_only_gravity_success", False):
            raise RuntimeError(f"sample {sample_index} passes left-only ablation")
        if metrics.get("right_only_gravity_success", False):
            raise RuntimeError(f"sample {sample_index} passes right-only ablation")

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
                "strict_source_dataset": str(args.strict_dataset.resolve()),
                "strict_source_sample_index": sample_index,
                "strict_source_candidate_index": int(
                    sample.get("candidate_index", -1)
                ),
                "strict_source_lift_mm": float(
                    metrics["lift_displacement_mm"]
                ),
                "strict_source_gravity_mm": float(
                    metrics["gravity_displacement_mm"]
                ),
                "strict_source_max_6dir_mm": float(
                    metrics["max_direction_displacement_mm"]
                ),
                "strict_source_height_difference_mm": float(
                    metrics["realized_root_height_difference_mm"]
                ),
                "strict_source_left_contact_links": int(
                    metrics["left_realized_contact_link_count"]
                ),
                "strict_source_right_contact_links": int(
                    metrics["right_realized_contact_link_count"]
                ),
                "requires_new_repeat_verification": True,
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
        f"saved {len(metadata)} strict first-pass seed parents -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
