#!/usr/bin/env python3
"""Aggregate repeated lift-bridge evaluations into robust promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from migration_4090.xhand_bodex_bimanual.contracts import (
    load_bodex_bank,
    sha256_file,
)
from migration_4090.xhand_bodex_bimanual.diversity import (
    descriptor_distance,
    grasp_descriptor,
)

from .throughput import THROUGHPUT_METRICS


SCREEN_METRICS = (
    "stage2_contact_pass",
    "physically_bounded_micro_lift",
    *THROUGHPUT_METRICS,
)


def _minimum_bank_distance(samples: list[dict[str, Any]]) -> float:
    if len(samples) < 2:
        raise ValueError("multi-seed screen requires at least two candidates")
    return min(
        descriptor_distance(
            grasp_descriptor(samples[left]),
            grasp_descriptor(samples[right]),
        )
        for left in range(len(samples))
        for right in range(left + 1, len(samples))
    )


def aggregate_evaluations(
    payloads: list[dict[str, Any]],
    *,
    minimum_seed_count: int,
    minimum_descriptor_distance: float,
    bank_descriptor_distance: float,
    minimum_per_candidate_controlled: float,
    minimum_per_candidate_ready: float,
    minimum_overall_sustained: float,
    minimum_sustained_per_hour: float,
    required_production_height_m: float,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("at least one evaluation is required")
    if minimum_seed_count <= 0:
        raise ValueError("minimum seed count must be positive")
    seeds = [int(payload["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("evaluation seeds must be unique")
    if len(seeds) < minimum_seed_count:
        raise RuntimeError(
            f"only {len(seeds)} unique seeds; minimum is {minimum_seed_count}"
        )
    first = payloads[0]
    invariants = (
        "object",
        "bodex_bank_sha256",
        "checkpoint_sha256",
        "policy_mode",
        "zero_action_candidates",
    )
    profile_name = first["profile"]["name"]
    candidate_ids = [
        first["candidate_rates"][str(index)]["candidate_id"]
        for index in range(len(first["candidate_rates"]))
    ]
    for payload in payloads[1:]:
        for key in invariants:
            if payload.get(key) != first.get(key):
                raise RuntimeError(f"evaluation invariant changed: {key}")
        if payload["profile"]["name"] != profile_name:
            raise RuntimeError("evaluation profile changed")
        current_ids = [
            payload["candidate_rates"][str(index)]["candidate_id"]
            for index in range(len(payload["candidate_rates"]))
        ]
        if current_ids != candidate_ids:
            raise RuntimeError("candidate order changed across evaluations")

    candidate_summary: dict[str, Any] = {}
    for index, candidate_id in enumerate(candidate_ids):
        rows = [payload["candidate_rates"][str(index)] for payload in payloads]
        metric_summary = {}
        for metric in SCREEN_METRICS:
            values = [float(row[metric]) for row in rows]
            total_episodes = sum(int(row["episodes"]) for row in rows)
            total_successes = sum(
                int(round(float(row[metric]) * int(row["episodes"])))
                for row in rows
            )
            metric_summary[metric] = {
                "minimum_rate": min(values),
                "maximum_rate": max(values),
                "weighted_rate": total_successes / total_episodes,
                "rates_by_seed": {
                    str(seed): value for seed, value in zip(seeds, values)
                },
            }
        candidate_summary[str(index)] = {
            "candidate_id": candidate_id,
            "episodes": sum(int(row["episodes"]) for row in rows),
            "metrics": metric_summary,
        }

    total_episodes = sum(int(payload["episodes"]) for payload in payloads)
    overall = {}
    for metric in SCREEN_METRICS:
        successes = sum(
            int(round(float(payload["overall_rates"][metric]) * int(payload["episodes"])))
            for payload in payloads
        )
        overall[metric] = {
            "successes": successes,
            "episodes": total_episodes,
            "weighted_rate": successes / total_episodes,
            "rates_by_seed": {
                str(payload["seed"]): float(payload["overall_rates"][metric])
                for payload in payloads
            },
        }

    timing_available = all("throughput" in payload for payload in payloads)
    throughput = {"available": timing_available}
    sustained_per_hour = 0.0
    if timing_available:
        total_wall_time = sum(
            float(payload["throughput"]["end_to_end_wall_time_s"])
            for payload in payloads
        )
        sustained_successes = overall["sustained_micro_lift"]["successes"]
        sustained_per_hour = 3600.0 * sustained_successes / total_wall_time
        throughput.update(
            {
                "end_to_end_wall_time_s": total_wall_time,
                "episodes_per_hour": 3600.0 * total_episodes / total_wall_time,
                "sustained_micro_lift_per_hour": sustained_per_hour,
            }
        )

    diversity_pass = bank_descriptor_distance >= minimum_descriptor_distance
    per_candidate_controlled_pass = all(
        row["metrics"]["controlled_micro_lift"]["minimum_rate"]
        >= minimum_per_candidate_controlled
        for row in candidate_summary.values()
    )
    per_candidate_ready_pass = all(
        row["metrics"]["stage4_ready_micro_lift"]["minimum_rate"]
        >= minimum_per_candidate_ready
        for row in candidate_summary.values()
    )
    bridge_screen_pass = (
        diversity_pass
        and per_candidate_controlled_pass
        and per_candidate_ready_pass
    )
    target_height_m = float(first["target_height_m"])
    production_pass = (
        bridge_screen_pass
        and target_height_m >= required_production_height_m
        and overall["sustained_micro_lift"]["weighted_rate"]
        >= minimum_overall_sustained
        and timing_available
        and sustained_per_hour >= minimum_sustained_per_hour
    )
    return {
        "schema": "xhand_bodex_multi_seed_lift_screen_v1",
        "object": first["object"],
        "profile": profile_name,
        "target_height_m": target_height_m,
        "bodex_bank": first["bodex_bank"],
        "bodex_bank_sha256": first["bodex_bank_sha256"],
        "checkpoint": first["checkpoint"],
        "checkpoint_sha256": first["checkpoint_sha256"],
        "policy_mode": first["policy_mode"],
        "zero_action_candidates": first["zero_action_candidates"],
        "seeds": seeds,
        "seed_count": len(seeds),
        "total_episodes": total_episodes,
        "bank_minimum_pairwise_descriptor_distance": bank_descriptor_distance,
        "candidate_summary": candidate_summary,
        "overall_summary": overall,
        "throughput": throughput,
        "thresholds": {
            "minimum_seed_count": minimum_seed_count,
            "minimum_descriptor_distance": minimum_descriptor_distance,
            "minimum_per_candidate_controlled": minimum_per_candidate_controlled,
            "minimum_per_candidate_ready": minimum_per_candidate_ready,
            "minimum_overall_sustained": minimum_overall_sustained,
            "minimum_sustained_per_hour": minimum_sustained_per_hour,
            "required_production_height_m": required_production_height_m,
        },
        "decisions": {
            "diversity_pass": diversity_pass,
            "per_candidate_controlled_pass": per_candidate_controlled_pass,
            "per_candidate_ready_pass": per_candidate_ready_pass,
            "bridge_screen_pass": bridge_screen_pass,
            "production_pass": production_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seed-count", type=int, default=3)
    parser.add_argument("--minimum-descriptor-distance", type=float, default=0.035)
    parser.add_argument("--minimum-per-candidate-controlled", type=float, default=0.25)
    parser.add_argument("--minimum-per-candidate-ready", type=float, default=0.03)
    parser.add_argument("--minimum-overall-sustained", type=float, default=0.05)
    parser.add_argument("--minimum-sustained-per-hour", type=float, default=100.0)
    parser.add_argument("--required-production-height-m", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.evaluation]
    bank_path = Path(payloads[0]["bodex_bank"])
    if sha256_file(bank_path) != payloads[0]["bodex_bank_sha256"]:
        raise RuntimeError("evaluation BODex bank hash no longer matches")
    bank = load_bodex_bank(
        bank_path,
        expected_object=payloads[0]["object"],
        verify_source=False,
        intended_stage=2,
    )
    result = aggregate_evaluations(
        payloads,
        minimum_seed_count=args.minimum_seed_count,
        minimum_descriptor_distance=args.minimum_descriptor_distance,
        bank_descriptor_distance=_minimum_bank_distance(bank["samples"]),
        minimum_per_candidate_controlled=args.minimum_per_candidate_controlled,
        minimum_per_candidate_ready=args.minimum_per_candidate_ready,
        minimum_overall_sustained=args.minimum_overall_sustained,
        minimum_sustained_per_hour=args.minimum_sustained_per_hour,
        required_production_height_m=args.required_production_height_m,
    )
    result["evaluations"] = [str(path.resolve()) for path in args.evaluation]
    result["evaluation_sha256"] = {
        str(path.resolve()): sha256_file(path) for path in args.evaluation
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), **result["decisions"]}, indent=2))


if __name__ == "__main__":
    main()
