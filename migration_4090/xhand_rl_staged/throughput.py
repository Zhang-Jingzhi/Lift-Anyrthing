"""Pure throughput helpers for BODex lift-bridge evaluations."""

from __future__ import annotations

from typing import Any


THROUGHPUT_METRICS = (
    "controlled_micro_lift",
    "stage4_ready_micro_lift",
    "sustained_micro_lift",
)


def evaluation_throughput(
    *,
    episodes: int,
    overall_rates: dict[str, float],
    end_to_end_wall_time_s: float,
    rollout_wall_time_s: float,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if end_to_end_wall_time_s <= 0.0 or rollout_wall_time_s <= 0.0:
        raise ValueError("wall times must be positive")
    counts = {
        metric: int(round(float(overall_rates[metric]) * episodes))
        for metric in THROUGHPUT_METRICS
    }
    return {
        "episodes": episodes,
        "end_to_end_wall_time_s": float(end_to_end_wall_time_s),
        "rollout_wall_time_s": float(rollout_wall_time_s),
        "episodes_per_hour_end_to_end": 3600.0
        * episodes
        / end_to_end_wall_time_s,
        "episodes_per_hour_rollout_only": 3600.0 * episodes / rollout_wall_time_s,
        "accepted_counts": counts,
        "accepted_per_hour_end_to_end": {
            metric: 3600.0 * count / end_to_end_wall_time_s
            for metric, count in counts.items()
        },
        "accepted_per_hour_rollout_only": {
            metric: 3600.0 * count / rollout_wall_time_s
            for metric, count in counts.items()
        },
        "production_contract_note": (
            "micro-lift throughput is a bridge KPI; final production acceptance "
            "still requires the configured full lift height, hold duration, "
            "trajectory safety, and diversity gate"
        ),
    }
