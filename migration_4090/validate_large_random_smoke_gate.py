#!/usr/bin/env python3
"""Validate repeat-verified large-object smoke outputs beyond exit status."""

import argparse
import json
import math
from pathlib import Path

import torch

from formal_large_random_protocol import assert_formal_protocol_manifest


def validate_dataset(path: Path, method: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_formal_protocol_manifest(payload.get("manifest", {}), str(path))
    samples = payload.get("samples", [])
    if not samples:
        raise RuntimeError(f"{method}: no repeat-verified samples in {path}")
    checked = []
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
        finite_keys = (
            "repeat_min_lift_mm",
            "repeat_max_gravity_mm",
            "repeat_max_6dir_mm",
            "repeat_max_left_penetration_mm",
            "repeat_max_right_penetration_mm",
            "repeat_min_hand_clearance_mm",
            "object_mass_kg",
        )
        nonfinite = [
            key
            for key in finite_keys
            if not math.isfinite(float(metrics.get(key, float("nan"))))
        ]
        if float(metrics.get("repeat_success_rate", 0.0)) != 1.0:
            failed.append("repeat_success_rate")
        if int(metrics.get("repeat_count", 0)) != 3:
            failed.append("repeat_count")
        if metrics.get("repeat_any_left_only_gravity_success") is not False:
            failed.append("left_only_ablation")
        if metrics.get("repeat_any_right_only_gravity_success") is not False:
            failed.append("right_only_ablation")
        if float(metrics.get("repeat_max_left_penetration_mm", float("inf"))) > 2.0:
            failed.append("left_penetration")
        if float(metrics.get("repeat_max_right_penetration_mm", float("inf"))) > 2.0:
            failed.append("right_penetration")
        if float(metrics.get("repeat_min_hand_clearance_mm", float("-inf"))) <= 2.0:
            failed.append("hand_clearance")
        if failed or nonfinite:
            raise RuntimeError(
                f"{method} sample {index} failed={failed} nonfinite={nonfinite}"
            )
        checked.append(
            {
                "sample_index": index,
                "object_name": sample["object_name"],
                "repeat_count": int(metrics["repeat_count"]),
                "repeat_success_rate": float(metrics["repeat_success_rate"]),
                "lift_mm": float(metrics["repeat_min_lift_mm"]),
                "gravity_mm": float(metrics["repeat_max_gravity_mm"]),
                "max_6dir_mm": float(metrics["repeat_max_6dir_mm"]),
                "left_links": int(metrics["repeat_min_left_contact_links"]),
                "right_links": int(metrics["repeat_min_right_contact_links"]),
                "left_penetration_mm": float(
                    metrics["repeat_max_left_penetration_mm"]
                ),
                "right_penetration_mm": float(
                    metrics["repeat_max_right_penetration_mm"]
                ),
                "hand_clearance_mm": float(metrics["repeat_min_hand_clearance_mm"]),
                "object_mass_kg": float(metrics["object_mass_kg"]),
                "local_cooperative_quality_pass": bool(
                    metrics.get("local_cooperative_quality_pass", False)
                ),
            }
        )
    return checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bidex", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "schema": "tro_grasp_large_random_smoke_gate_v1",
        "baseline_dataset": str(args.baseline.resolve()),
        "bidex_dataset": str(args.bidex.resolve()),
        "baseline": validate_dataset(args.baseline, "baseline"),
        "bidex_v3": validate_dataset(args.bidex, "bidex_v3"),
        "gate_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
