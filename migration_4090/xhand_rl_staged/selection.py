"""Pure checkpoint selection helpers for balanced BODex curricula."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def candidate_success_rates(evaluation: dict[str, Any]) -> dict[int, float]:
    """Return deterministic per-candidate stage success rates.

    Formal evaluation assigns candidates in an equal round-robin schedule.  A
    separate rate is still necessary because the overall mean can hide a
    policy that improves one BODex seed while destroying another one.
    """

    totals: dict[int, int] = defaultdict(int)
    successes: dict[int, int] = defaultdict(int)
    for report in evaluation.get("reports", ()):
        candidate = int(report["candidate_index"])
        totals[candidate] += 1
        successes[candidate] += int(
            report.get("stage_pass") is True
            and report.get("terminated_early") is not True
        )
    if not totals:
        raise ValueError("checkpoint selection requires per-candidate reports")
    return {
        candidate: successes[candidate] / count
        for candidate, count in sorted(totals.items())
    }


def balanced_checkpoint_score(evaluation: dict[str, Any]) -> float:
    """Balance aggregate performance with the weakest genuine BODex seed."""

    rates = candidate_success_rates(evaluation)
    return 0.5 * float(evaluation["success_rate"]) + 0.5 * min(rates.values())


def checkpoint_selection_decision(
    evaluations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Choose a champion without letting a weak candidate hide in the mean."""

    rows = [row for row in evaluations if row.get("checkpoint")]
    if not rows:
        raise ValueError("checkpoint selection requires evaluated checkpoints")
    scored = []
    for index, row in enumerate(rows):
        rates = candidate_success_rates(row)
        scored.append(
            (
                balanced_checkpoint_score(row),
                min(rates.values()),
                float(row["success_rate"]),
                index,
                row,
                rates,
            )
        )
    selected = max(scored, key=lambda item: item[:4])
    latest = scored[-1]
    return {
        "schema": "xhand_rl_staged_checkpoint_selection_v1",
        "evaluations_considered": len(rows),
        "selected_checkpoint": str(selected[4]["checkpoint"]),
        "selected_success_rate": float(selected[4]["success_rate"]),
        "selected_candidate_success_rates": {
            str(key): value for key, value in selected[5].items()
        },
        "selected_balanced_score": selected[0],
        "latest_checkpoint": str(latest[4]["checkpoint"]),
        "latest_success_rate": float(latest[4]["success_rate"]),
        "latest_candidate_success_rates": {
            str(key): value for key, value in latest[5].items()
        },
        "latest_balanced_score": latest[0],
        "latest_selected": selected[3] == latest[3],
    }
