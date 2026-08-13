#!/usr/bin/env python3
"""Summarize or strictly validate formal large-random generation progress."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from select_bulk_repeat_candidates import pose_key
from formal_large_random_protocol import (
    PROTOCOL_ID,
    assert_formal_protocol_manifest,
    is_formal_protocol_manifest,
)


BASE_OBJECTS = (
    "ycb+bleach_cleanser",
    "ycb+cracker_box",
    "ycb+pitcher_base",
    "contactdb+piggy_bank",
    "ycb+power_drill",
    "ycb+toy_airplane",
)
METHODS = ("baseline", "bidex_v3")
SIZE_TARGETS = (13, 13, 13, 13, 12, 12, 12, 12)


def slug(name):
    return name.replace("+", "_")


def verified_progress(base_root):
    raw = 0
    keys = set()
    files = list(base_root.glob("size_*/seed_*/repeat_verified/verified_dataset.pt"))
    accepted_files = 0
    skipped_files = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not is_formal_protocol_manifest(payload.get("manifest", {})):
            skipped_files += 1
            continue
        accepted_files += 1
        for sample in payload.get("samples", []):
            raw += 1
            keys.add((sample["object_name"], pose_key(sample)))
    return accepted_files, skipped_files, raw, len(keys)


def assembled_progress(base_root):
    """Count only rows already committed to the fixed 100-row size schedule."""
    size_counts = {}
    for size_index, target in enumerate(SIZE_TARGETS):
        path = (
            base_root
            / f"size_{size_index:03d}"
            / f"assembled_target_{target:02d}"
            / "bimanual_dataset.pt"
        )
        count = 0
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples = payload.get("samples", [])
            if len(samples) != target:
                raise RuntimeError(
                    f"assembled size count mismatch: {path}: "
                    f"{len(samples)} != {target}"
                )
            count = len(samples)
        size_counts[str(size_index)] = count
    return size_counts, sum(size_counts.values())


def validate_final(path, method, base_object):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest", {})
    if manifest.get("formal_protocol_id") != PROTOCOL_ID:
        raise RuntimeError(
            f"{method}/{base_object}: final protocol id mismatch"
        )
    samples = payload.get("samples", [])
    if len(samples) != 100:
        raise RuntimeError(f"{method}/{base_object}: {len(samples)} != 100")
    schedule_indices = []
    sizes = Counter()
    identities = set()
    for index, sample in enumerate(samples):
        metrics = sample.get("metrics", {})
        required_true = (
            "strict_success",
            "geometry_pass",
            "isaac_success",
            "bimanual_required",
            "realized_pose_pass",
            "repeat_bilateral_physical_contact_all_phases",
        )
        failed = [key for key in required_true if metrics.get(key) is not True]
        if float(metrics.get("repeat_success_rate", 0.0)) != 1.0:
            failed.append("repeat_success_rate")
        if int(metrics.get("repeat_count", 0)) != 3:
            failed.append("repeat_count")
        if metrics.get("repeat_any_left_only_gravity_success") is not False:
            failed.append("left_only_ablation")
        if metrics.get("repeat_any_right_only_gravity_success") is not False:
            failed.append("right_only_ablation")
        numeric_limits = {
            "repeat_max_left_penetration_mm": (None, 2.0),
            "repeat_max_right_penetration_mm": (None, 2.0),
            "repeat_min_hand_clearance_mm": (2.0, None),
            "repeat_min_lift_mm": (20.0, None),
            "repeat_max_gravity_mm": (None, 12.5),
            "repeat_max_6dir_mm": (None, 15.0),
        }
        for key, (lower, upper) in numeric_limits.items():
            value = float(metrics.get(key, float("nan")))
            if not math.isfinite(value):
                failed.append(key + "_nonfinite")
            elif lower is not None and value < lower:
                failed.append(key + "_low")
            elif upper is not None and value > upper:
                failed.append(key + "_high")
        if failed:
            raise RuntimeError(f"{method}/{base_object} sample={index} failed={failed}")
        provenance = sample.get("provenance", {})
        schedule_indices.append(int(provenance["formal_schedule_sample_index"]))
        sizes[int(provenance["formal_size_index"])] += 1
        identity = (sample["object_name"], pose_key(sample))
        if identity in identities:
            raise RuntimeError(f"{method}/{base_object}: duplicate pose at {index}")
        identities.add(identity)
    if sorted(schedule_indices) != list(range(100)):
        raise RuntimeError(f"{method}/{base_object}: schedule indices invalid")
    return {"samples": 100, "size_counts": dict(sorted(sizes.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("graph_exp/bimanual_data/large_random_6x100_v1"),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {"schema": "tro_grasp_large_random_6x100_progress_v1", "methods": {}}
    complete = True
    for method in METHODS:
        method_summary = {}
        for base_object in BASE_OBJECTS:
            base_root = args.root / method / slug(base_object)
            final = base_root / "final_100/bimanual_dataset.pt"
            files, skipped, raw, unique = verified_progress(base_root)
            assembled_sizes, assembled_count = assembled_progress(base_root)
            row = {
                "repeat_verified_files": files,
                "repeat_verified_skipped_legacy_files": skipped,
                "repeat_verified_raw": raw,
                "repeat_verified_unique": unique,
                "scheduled_assembled_size_counts": assembled_sizes,
                "scheduled_assembled_count": assembled_count,
                "final_dataset": str(final.resolve()),
                "final_exists": final.is_file(),
            }
            if final.is_file():
                row["validation"] = validate_final(final, method, base_object)
            else:
                complete = False
                if args.strict:
                    raise RuntimeError(f"missing final dataset: {final}")
            method_summary[base_object] = row
        summary["methods"][method] = method_summary
        summary.setdefault("scheduled_assembled_by_method", {})[method] = sum(
            row["scheduled_assembled_count"] for row in method_summary.values()
        )
    summary["complete"] = complete
    summary["final_sample_count"] = sum(
        row.get("validation", {}).get("samples", 0)
        for method in summary["methods"].values()
        for row in method.values()
    )
    summary["scheduled_assembled_sample_count"] = sum(
        summary["scheduled_assembled_by_method"].values()
    )
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
