#!/usr/bin/env python3
"""Rank, deduplicate, and cap strict candidates before repeat rollouts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from formal_large_random_protocol import is_formal_protocol_manifest


def pose_key(sample):
    values = torch.cat(
        [sample["left_q_seed"].detach().cpu(), sample["right_q_seed"].detach().cpu()]
    )
    return tuple(torch.round(values * 100_000).to(torch.int64).tolist())


def rank_key(sample):
    metrics = sample.get("metrics", {})
    left_links = int(
        metrics.get(
            "repeat_min_left_contact_links",
            metrics.get("left_realized_contact_link_count", 0),
        )
    )
    right_links = int(
        metrics.get(
            "repeat_min_right_contact_links",
            metrics.get("right_realized_contact_link_count", 0),
        )
    )
    penetration = max(
        float(
            metrics.get(
                "repeat_max_left_penetration_mm",
                metrics.get("left_realized_penetration_mm", float("inf")),
            )
        ),
        float(
            metrics.get(
                "repeat_max_right_penetration_mm",
                metrics.get("right_realized_penetration_mm", float("inf")),
            )
        ),
    )
    disturbance = float(
        metrics.get(
            "repeat_max_6dir_mm",
            metrics.get("max_direction_displacement_mm", float("inf")),
        )
    )
    gravity = float(
        metrics.get(
            "repeat_max_gravity_mm",
            metrics.get("gravity_displacement_mm", float("inf")),
        )
    )
    lift = float(
        metrics.get(
            "repeat_min_lift_mm",
            metrics.get("lift_displacement_mm", float("-inf")),
        )
    )
    clearance = float(
        metrics.get(
            "repeat_min_hand_clearance_mm",
            metrics.get("realized_hand_clearance_mm", float("-inf")),
        )
    )
    parent_gravity = float(
        metrics.get("verified_repeat_max_gravity_mm", 0.0)
    )
    parent_disturbance = float(
        metrics.get("verified_repeat_max_6dir_mm", 0.0)
    )
    parent_lift = float(
        metrics.get("verified_repeat_min_lift_mm", 0.0)
    )
    perturbation = sum(
        abs(float(metrics.get(name, 0.0) or 0.0)) * scale
        for name, scale in (
            ("high_vhacd_common_yaw_degrees", 1.0),
            ("high_vhacd_left_yaw_delta_degrees", 0.5),
            ("high_vhacd_right_yaw_delta_degrees", 0.5),
            ("high_vhacd_common_z_mm", 1.0),
            ("high_vhacd_left_z_delta_mm", 1.0),
            ("high_vhacd_right_z_delta_mm", 1.0),
            ("high_vhacd_left_radial_delta_mm", 1.0),
            ("high_vhacd_right_radial_delta_mm", 1.0),
            ("high_vhacd_left_tangential_delta_mm", 1.0),
            ("high_vhacd_right_tangential_delta_mm", 1.0),
            ("high_vhacd_left_joint_delta_rad", 20.0),
            ("high_vhacd_right_joint_delta_rad", 20.0),
            ("high_vhacd_left_flexion_delta_rad", 20.0),
            ("high_vhacd_right_flexion_delta_rad", 20.0),
        )
    )
    # Prefer balanced bilateral contact first, then total contact redundancy.
    return (
        -min(left_links, right_links),
        parent_gravity,
        parent_disturbance,
        -parent_lift,
        -(left_links + right_links),
        perturbation,
        penetration,
        disturbance,
        gravity,
        -lift,
        -clearance,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument(
        "--local-quality-json",
        type=Path,
        help=(
            "Optional output from audit_decoupled_force_closure.py for the "
            "source dataset. Local wrench pass and residual are used as "
            "ranking preferences before expensive repeat rollouts."
        ),
    )
    parser.add_argument(
        "--exclude-root",
        type=Path,
        help=(
            "Exclude pose keys already present in seed_*/repeat_verified/"
            "verified_dataset.pt below this size root."
        ),
    )
    parser.add_argument(
        "--max-historical-selection-attempts",
        type=int,
        default=2,
        help=(
            "When --exclude-root is set, stop retrying a pose after it has "
            "appeared in this many prior formal repeat_selection.pt files. "
            "The default keeps one stochastic retry while preventing a "
            "small set of repeat failures from permanently blocking new "
            "candidates."
        ),
    )
    args = parser.parse_args()
    if args.max_historical_selection_attempts < 1:
        raise ValueError("--max-historical-selection-attempts must be positive")

    if args.output.exists():
        raise FileExistsError(args.output)
    payload = torch.load(args.source_dataset, map_location="cpu", weights_only=False)
    samples = payload.get("samples", [])
    local_quality = None
    local_quality_pass = None
    local_quality_residual = None
    if args.local_quality_json is not None:
        local_quality = json.loads(args.local_quality_json.read_text())
        rows = local_quality.get("samples", [])
        if len(rows) != len(samples):
            raise RuntimeError(
                "local-quality row count does not match source samples: "
                f"{len(rows)} != {len(samples)}"
            )
        local_quality_pass = {
            int(row["sample_index"]): bool(
                row.get("decoupled_force_closure_pass", False)
            )
            for row in rows
        }
        local_quality_residual = {
            int(row["sample_index"]): max(
                float(row["left"]["mean_wrench_residual"]),
                float(row["right"]["mean_wrench_residual"]),
            )
            for row in rows
        }
        if sorted(local_quality_pass) != list(range(len(samples))):
            raise RuntimeError(
                "local-quality sample_index values are not contiguous"
            )
    excluded = set()
    excluded_files = 0
    historical_selection_files = 0
    historical_selection_attempts = Counter()
    if args.exclude_root is not None:
        for path in sorted(
            args.exclude_root.glob("seed_*/repeat_verified/verified_dataset.pt")
        ):
            previous = torch.load(path, map_location="cpu", weights_only=False)
            excluded_files += 1
            for sample in previous.get("samples", []):
                excluded.add(pose_key(sample))
        for path in sorted(args.exclude_root.glob("seed_*/repeat_selection.pt")):
            previous = torch.load(path, map_location="cpu", weights_only=False)
            manifest = previous.get("manifest", {})
            if "large_random_6x100_v1" in str(args.exclude_root) and not is_formal_protocol_manifest(manifest):
                continue
            historical_selection_files += 1
            for sample in previous.get("samples", []):
                historical_selection_attempts[pose_key(sample)] += 1
        excluded.update(
            key
            for key, attempts in historical_selection_attempts.items()
            if attempts >= args.max_historical_selection_attempts
        )
    unique = {}
    def selection_rank(sample, sample_index):
        if local_quality_pass is None:
            quality_prefix = (0, 0.0)
        else:
            quality_prefix = (
                0 if local_quality_pass[sample_index] else 1,
                local_quality_residual[sample_index],
            )
        return quality_prefix + rank_key(sample)

    for sample_index, sample in enumerate(samples):
        key = pose_key(sample)
        if key in excluded:
            continue
        previous = unique.get(key)
        if (
            previous is None
            or selection_rank(sample, sample_index)
            < selection_rank(previous[0], previous[1])
        ):
            unique[key] = (sample, sample_index)
    ranked_rows = sorted(
        unique.values(), key=lambda row: selection_rank(row[0], row[1])
    )[: args.max_samples]
    ranked = [sample for sample, _ in ranked_rows]
    manifest = dict(payload.get("manifest", {}))
    manifest.update(
        {
            "bulk_selection_source": str(args.source_dataset.resolve()),
            "bulk_selection_input_samples": len(samples),
            "bulk_selection_local_quality_json": (
                str(args.local_quality_json.resolve())
                if args.local_quality_json is not None
                else None
            ),
            "bulk_selection_local_quality_passed": (
                sum(local_quality_pass.values())
                if local_quality_pass is not None
                else None
            ),
            "bulk_selection_excluded_verified_files": excluded_files,
            "bulk_selection_excluded_pose_keys": len(excluded),
            "bulk_selection_historical_selection_files": historical_selection_files,
            "bulk_selection_historical_attempted_pose_keys": len(
                historical_selection_attempts
            ),
            "bulk_selection_max_historical_selection_attempts": (
                args.max_historical_selection_attempts
            ),
            "bulk_selection_excluded_attempted_pose_keys": sum(
                attempts >= args.max_historical_selection_attempts
                for attempts in historical_selection_attempts.values()
            ),
            "bulk_selection_unique_samples": len(unique),
            "bulk_selection_output_samples": len(ranked),
            "bulk_selection_rank": (
                "local_quality_pass,local_quality_residual,balanced_contact,"
                "parent_gravity,parent_disturbance,parent_lift,total_contact,"
                "perturbation,penetration,disturbance,gravity,lift,clearance"
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"version": payload.get("version"), "manifest": manifest, "samples": ranked},
        args.output,
    )
    print(
        f"selected {len(ranked)}/{len(samples)} strict samples "
        f"({len(unique)} unique) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
