#!/usr/bin/env python3
"""Strictly validate and compare the two 600-sample bulk corpora."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


METRICS = {
    "min_contact_links": lambda m: min(
        m["left_realized_contact_link_count"], m["right_realized_contact_link_count"]
    ),
    "total_contact_links": lambda m: (
        m["left_realized_contact_link_count"] + m["right_realized_contact_link_count"]
    ),
    "max_penetration_mm": lambda m: max(
        m["left_realized_penetration_mm"], m["right_realized_penetration_mm"]
    ),
    "hand_clearance_mm": lambda m: m["realized_hand_clearance_mm"],
    "lift_mm": lambda m: m["lift_displacement_mm"],
    "gravity_mm": lambda m: m["gravity_displacement_mm"],
    "max_6dir_mm": lambda m: m["max_direction_displacement_mm"],
    "object_mass_kg": lambda m: m["object_mass_kg"],
    "repeat_min_contact_links": lambda m: min(
        m["repeat_min_left_contact_links"], m["repeat_min_right_contact_links"]
    ),
    "repeat_max_penetration_mm": lambda m: max(
        m["repeat_max_left_penetration_mm"],
        m["repeat_max_right_penetration_mm"],
    ),
    "repeat_min_hand_clearance_mm": lambda m: m[
        "repeat_min_hand_clearance_mm"
    ],
    "repeat_min_lift_mm": lambda m: m["repeat_min_lift_mm"],
    "repeat_max_gravity_mm": lambda m: m["repeat_max_gravity_mm"],
    "repeat_max_6dir_mm": lambda m: m["repeat_max_6dir_mm"],
}


def validate_sample(sample, label, index):
    m = sample["metrics"]
    failures = []
    checks = {
        "strict_success": bool(m.get("strict_success")),
        "bimanual_required": bool(m.get("bimanual_required")),
        "left_contact": m.get("left_realized_contact_link_count", 0) >= 1,
        "right_contact": m.get("right_realized_contact_link_count", 0) >= 1,
        "penetration": max(
            m.get("left_realized_penetration_mm", math.inf),
            m.get("right_realized_penetration_mm", math.inf),
        ) <= 2.0,
        "clearance": m.get("realized_hand_clearance_mm", -math.inf) > 2.0,
        "lift": m.get("lift_displacement_mm", -math.inf) >= 30.0,
        "gravity": m.get("gravity_displacement_mm", math.inf) <= 10.0,
        "six_direction": m.get("max_direction_displacement_mm", math.inf) <= 15.0,
        "left_only_failed": not bool(m.get("left_only_gravity_success", True)),
        "right_only_failed": not bool(m.get("right_only_gravity_success", True)),
        "repeat_count": m.get("repeat_count") == 3,
        "repeat_success": m.get("repeat_success_rate") == 1.0,
        "repeat_lift": m.get("repeat_min_lift_mm", -math.inf) >= 30.0,
        "repeat_gravity": m.get("repeat_max_gravity_mm", math.inf) <= 10.0,
        "repeat_six_direction": m.get("repeat_max_6dir_mm", math.inf) <= 15.0,
        "repeat_left_contact": m.get("repeat_min_left_contact_links", 0) >= 1,
        "repeat_right_contact": m.get("repeat_min_right_contact_links", 0) >= 1,
        "repeat_left_penetration": (
            m.get("repeat_max_left_penetration_mm", math.inf) <= 2.0
        ),
        "repeat_right_penetration": (
            m.get("repeat_max_right_penetration_mm", math.inf) <= 2.0
        ),
        "repeat_clearance": m.get("repeat_min_hand_clearance_mm", -math.inf) > 2.0,
        "repeat_left_only_failed": not bool(
            m.get("repeat_any_left_only_gravity_success", True)
        ),
        "repeat_right_only_failed": not bool(
            m.get("repeat_any_right_only_gravity_success", True)
        ),
        "physical_contact_repeat_count": m.get(
            "physical_contact_repeat_count", 0
        ) == 2,
        "bilateral_physical_contact": bool(
            m.get("repeat_bilateral_physical_contact_all_phases", False)
        ),
    }
    for phase in ("closure", "lifted", "settled"):
        for side in ("left", "right"):
            checks[f"{phase}_{side}_active_physical_contact"] = (
                m.get(f"repeat_min_{phase}_active_{side}_points", 0) >= 1
            )
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    for name, getter in METRICS.items():
        try:
            if not math.isfinite(float(getter(m))):
                failures.append(f"finite_{name}")
        except (KeyError, TypeError, ValueError):
            failures.append(f"present_{name}")
    if failures:
        raise RuntimeError(f"{label} sample {index} failed: {failures}")


def describe(method, payload):
    samples = payload["samples"]
    if len(samples) != 600:
        raise RuntimeError(f"{method}: expected 600, found {len(samples)}")
    counts = Counter(sample["object_name"] for sample in samples)
    if len(counts) != 6 or set(counts.values()) != {100}:
        raise RuntimeError(f"{method}: bad object counts {dict(counts)}")
    for index, sample in enumerate(samples):
        validate_sample(sample, method, index)
    result = {"method": method, "samples": len(samples), "objects": len(counts)}
    for name, getter in METRICS.items():
        values = np.asarray([float(getter(sample["metrics"])) for sample in samples])
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_p10"] = float(np.percentile(values, 10))
        result[f"{name}_p90"] = float(np.percentile(values, 90))
        result[f"{name}_min"] = float(values.min())
        result[f"{name}_max"] = float(values.max())
    result["counts_by_object"] = dict(sorted(counts.items()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bidex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    results = []
    for method, path in (("baseline", args.baseline), ("bidex_style", args.bidex)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        results.append(describe(method, payload))
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    flat = [{k: v for k, v in row.items() if k != "counts_by_object"} for row in results]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Bulk irregular-object synthesis comparison",
        "",
        "Both corpora passed the strict 600 = 6 objects x 100 records audit. Each",
        "record passed three rollouts, bilateral contact, lift, six independent",
        "disturbances, penetration/clearance gates, and both single-hand ablations.",
        "",
        "| method | 3-rollout min links mean | 3-rollout max penetration mean (mm) | 3-rollout min lift mean (mm) | 3-rollout max gravity mean (mm) | 3-rollout max 6-dir mean (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['method']} | {row['repeat_min_contact_links_mean']:.3f} | "
            f"{row['repeat_max_penetration_mm_mean']:.3f} | "
            f"{row['repeat_min_lift_mm_mean']:.3f} | "
            f"{row['repeat_max_gravity_mm_mean']:.3f} | "
            f"{row['repeat_max_6dir_mm_mean']:.3f} |"
        )
    lines += [
        "",
        "Dynamic displacement values must be interpreted together with object mass;",
        "the two methods currently use method-specific densities established by the",
        "successful smoke tests, so contact redundancy and geometric safety are the",
        "cleaner direct comparison.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
