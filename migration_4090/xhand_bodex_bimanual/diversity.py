"""Pose descriptors and near-neighbour filtering for bimanual grasp banks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def grasp_descriptor(sample: dict[str, Any]) -> np.ndarray:
    """Return a scale-balanced descriptor over both arm and hand postures."""

    names = list(sample["joint_names"])
    q = np.asarray(sample["full_body_q"], dtype=np.float64)
    if q.shape != (len(names),):
        raise ValueError("full_body_q does not match joint_names")
    arm = np.asarray([q[index] for index, name in enumerate(names) if "_hand_" not in name and (name.startswith("left_") or name.startswith("right_"))])
    hand = np.asarray([q[index] for index, name in enumerate(names) if "_hand_" in name])
    if arm.size != 14 or hand.size != 24:
        raise ValueError("descriptor expects 14 arm and 24 XHand joints")
    return np.concatenate((arm / np.pi, hand / 1.9)).astype(np.float32)


def descriptor_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("descriptor shapes differ")
    return float(np.sqrt(np.mean(np.square(first - second))))


def is_near_duplicate(
    candidate: dict[str, Any],
    existing: Iterable[dict[str, Any]],
    *,
    minimum_distance: float = 0.035,
) -> bool:
    descriptor = grasp_descriptor(candidate)
    return any(
        descriptor_distance(descriptor, grasp_descriptor(sample)) < minimum_distance
        for sample in existing
    )


def diverse_subset(
    samples: Iterable[dict[str, Any]],
    *,
    minimum_distance: float = 0.035,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample in samples:
        if not is_near_duplicate(sample, selected, minimum_distance=minimum_distance):
            selected.append(sample)
            if limit is not None and len(selected) >= limit:
                break
    return selected
