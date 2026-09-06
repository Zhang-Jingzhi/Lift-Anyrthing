"""Deterministic candidate sampling utilities for diverse BODex training."""

from __future__ import annotations


def parse_candidate_repeat_factors(
    value: str | None, *, candidate_count: int
) -> tuple[int, ...]:
    """Parse positive repeat factors while retaining every bank candidate."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if value is None:
        return (1,) * candidate_count
    try:
        factors = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("candidate repeat factors must be comma-separated integers") from exc
    if len(factors) != candidate_count:
        raise ValueError(
            "candidate repeat factor count must match the BODex candidate count: "
            f"{len(factors)} != {candidate_count}"
        )
    if any(factor <= 0 for factor in factors):
        raise ValueError("candidate repeat factors must all be positive")
    return factors


def parse_candidate_indices(
    value: str | None, *, candidate_count: int
) -> tuple[int, ...] | None:
    """Parse an optional unique candidate subset in stable numeric order."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if value is None:
        return None
    try:
        indices = tuple(sorted({int(part.strip()) for part in value.split(",")}))
    except ValueError as exc:
        raise ValueError("candidate indices must be comma-separated integers") from exc
    if not indices:
        raise ValueError("at least one candidate index is required")
    if any(index < 0 or index >= candidate_count for index in indices):
        raise ValueError("candidate index is outside the BODex bank")
    return indices
