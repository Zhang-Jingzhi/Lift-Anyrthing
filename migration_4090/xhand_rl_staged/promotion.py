#!/usr/bin/env python3
"""Fail-closed promotion check over consecutive fixed evaluation windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    PROMOTION_CONSECUTIVE_WINDOWS,
    PROMOTION_SUCCESS_RATE,
    STAGED_EVALUATION_SCHEMA,
)


def promotion_decision(
    evaluations: Iterable[dict[str, Any]],
    *,
    stage: int,
    object_name: str,
    success_rate: float = PROMOTION_SUCCESS_RATE,
    consecutive_windows: int = PROMOTION_CONSECUTIVE_WINDOWS,
    evaluation_num_envs: int = 64,
    evaluation_episodes: int = 256,
) -> dict[str, Any]:
    rows = [
        row
        for row in evaluations
        if row.get("schema") == STAGED_EVALUATION_SCHEMA
        and int(row.get("stage", -1)) == stage
        and row.get("object") == object_name
    ]
    selected = rows[-consecutive_windows:]
    protocol_valid = [
        int(row.get("num_envs", -1)) == evaluation_num_envs
        and int(row.get("episodes", -1)) == evaluation_episodes
        and row.get("candidate_selection") == "round_robin"
        and row.get("deterministic_policy") is True
        for row in selected
    ]
    passed = (
        len(selected) == consecutive_windows
        and all(protocol_valid)
        and all(float(row.get("success_rate", 0.0)) >= success_rate for row in selected)
        and all(row.get("earlier_gates_no_regression") is True for row in selected)
    )
    return {
        "schema": "xhand_rl_staged_promotion_decision_v1",
        "object": object_name,
        "stage": stage,
        "required_success_rate": success_rate,
        "required_consecutive_windows": consecutive_windows,
        "required_evaluation_num_envs": evaluation_num_envs,
        "required_evaluation_episodes": evaluation_episodes,
        "windows_available": len(selected),
        "window_success_rates": [float(row.get("success_rate", 0.0)) for row in selected],
        "window_protocol_valid": protocol_valid,
        "promote": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.evaluations.glob("evaluation_*.json")):
        rows.append(json.loads(path.read_text()))
    decision = promotion_decision(
        rows,
        stage=args.stage,
        object_name=args.object,
        evaluation_num_envs=args.num_envs,
        evaluation_episodes=args.episodes,
    )
    rendered = json.dumps(decision, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if decision["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
