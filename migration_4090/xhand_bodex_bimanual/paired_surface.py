"""Generate opposed, side-aware surface seeds for BODex's two transferred links."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# Object-frame tangent references for the two mounted XHands.  Both palm
# links use local +X as the inward normal, but the fixed arm/hand mounts have
# different reachable wrist roll conventions.  These fixed directions were
# calibrated from palm-link axes only (not joint targets) in a previously
# validated Isaac grasp.  They define orientation about the sampled surface
# normal; positions and all 38 joint values are still solved by BODex.
LEFT_TANGENT_REFERENCE = np.asarray((-1.0, 0.0, 0.0), dtype=np.float64)
RIGHT_TANGENT_REFERENCE = np.asarray((0.315, 0.0, 0.949), dtype=np.float64)


@dataclass(frozen=True)
class PairedSurfaceSeed:
    left_position_m: tuple[float, float, float]
    right_position_m: tuple[float, float, float]
    left_rotation_6d: tuple[float, float, float, float, float, float]
    right_rotation_6d: tuple[float, float, float, float, float, float]
    opposition_cosine: float
    span_m: float
    source_surface_indices: tuple[int, int]
    lateral_fraction: float = 0.0
    side_normal_min_component: float = 0.0
    x_offset_m: float = 0.0
    z_offset_m: float = 0.0


def _rotation_6d_from_surface_normal(
    normal: np.ndarray, tangent_reference: np.ndarray
) -> np.ndarray:
    first = -np.asarray(normal, dtype=np.float64)
    first /= max(np.linalg.norm(first), 1.0e-12)
    reference = np.asarray(tangent_reference, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("tangent reference must contain three finite values")
    reference /= max(np.linalg.norm(reference), 1.0e-12)
    if abs(float(np.dot(first, reference))) > 0.95:
        fallback = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(first, fallback))) > 0.95:
            fallback = np.asarray([0.0, 1.0, 0.0])
        reference = fallback
    second = reference - float(np.dot(reference, first)) * first
    second /= max(np.linalg.norm(second), 1.0e-12)
    return np.concatenate((first, second))


def pair_surface_samples(
    positions: np.ndarray,
    normals: np.ndarray,
    *,
    count: int,
    seed: int,
    standoff_m: float = 0.015,
    minimum_opposition_cosine: float = 0.75,
    minimum_span_m: float = 0.04,
    minimum_side_normal_component: float = 0.55,
    minimum_lateral_fraction: float = 0.75,
    maximum_x_offset_m: float = 0.075,
    maximum_z_offset_m: float = 0.075,
) -> list[PairedSurfaceSeed]:
    """Pair genuinely lateral positive/negative-Y points into bimanual seeds.

    Merely splitting points by their Y coordinate is insufficient: an
    opposed-normal search can otherwise pair a point near the top of a round
    object with one near the bottom.  Such a pair is mathematically opposed
    but does not describe the left/right side approach required by the fixed
    dual-arm XHand setup.
    """

    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or normals.shape != positions.shape:
        raise ValueError("positions and normals must both have shape [N, 3]")
    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(positions).all() or not np.isfinite(normals).all():
        raise ValueError("positions and normals must be finite")
    if not 0.0 <= minimum_side_normal_component <= 1.0:
        raise ValueError("minimum_side_normal_component must be in [0, 1]")
    if not 0.0 <= minimum_lateral_fraction <= 1.0:
        raise ValueError("minimum_lateral_fraction must be in [0, 1]")
    if min(maximum_x_offset_m, maximum_z_offset_m) < 0.0:
        raise ValueError("maximum cross-axis offsets must be non-negative")
    normal_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if bool(np.any(normal_lengths <= 1.0e-12)):
        raise ValueError("surface normals must be non-zero")
    normalized = normals / normal_lengths
    center_y = float(np.median(positions[:, 1]))
    left_indices = np.flatnonzero(
        (positions[:, 1] >= center_y)
        & (normalized[:, 1] >= minimum_side_normal_component)
    )
    right_indices = np.flatnonzero(
        (positions[:, 1] < center_y)
        & (normalized[:, 1] <= -minimum_side_normal_component)
    )
    if not len(left_indices) or not len(right_indices):
        raise RuntimeError(
            "mesh surface has no samples satisfying the left/right side-normal constraint"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(left_indices)
    seeds: list[PairedSurfaceSeed] = []
    used: set[tuple[int, int]] = set()
    used_right: set[int] = set()
    for left_index in left_indices:
        deltas = positions[right_indices] - positions[left_index]
        opposition = -(normalized[right_indices] @ normalized[left_index])
        spans = np.linalg.norm(deltas, axis=-1)
        lateral_fraction = np.abs(deltas[:, 1]) / spans.clip(1.0e-12)
        valid = (
            (opposition >= minimum_opposition_cosine)
            & (spans >= minimum_span_m)
            & (lateral_fraction >= minimum_lateral_fraction)
            & (np.abs(deltas[:, 0]) <= maximum_x_offset_m)
            & (np.abs(deltas[:, 2]) <= maximum_z_offset_m)
        )
        valid_rows = np.flatnonzero(valid)
        if not len(valid_rows):
            continue
        available_rows = np.asarray(
            [row for row in valid_rows if int(right_indices[row]) not in used_right],
            dtype=np.int64,
        )
        if not len(available_rows):
            continue
        candidates = right_indices[available_rows]
        valid_deltas = deltas[available_rows]
        valid_spans = spans[available_rows]
        valid_lateral_fraction = lateral_fraction[available_rows]
        cross_axis_fraction = (
            np.abs(valid_deltas[:, 0]) + np.abs(valid_deltas[:, 2])
        ) / valid_spans.clip(1.0e-12)
        valid_scores = (
            2.0 * opposition[available_rows]
            + valid_lateral_fraction
            - 0.25 * cross_axis_fraction
        )
        order = np.argsort(valid_scores)[::-1]
        top = order[: min(16, len(order))]
        selected = int(rng.choice(top))
        right_index = int(candidates[selected])
        key = (int(left_index), right_index)
        if key in used:
            continue
        used.add(key)
        used_right.add(right_index)
        left_position = positions[left_index] + standoff_m * normalized[left_index]
        right_position = positions[right_index] + standoff_m * normalized[right_index]
        selected_delta = valid_deltas[selected]
        selected_span = float(valid_spans[selected])
        seeds.append(
            PairedSurfaceSeed(
                left_position_m=tuple(float(value) for value in left_position),
                right_position_m=tuple(float(value) for value in right_position),
                left_rotation_6d=tuple(
                    float(value)
                    for value in _rotation_6d_from_surface_normal(
                        normalized[left_index], LEFT_TANGENT_REFERENCE
                    )
                ),
                right_rotation_6d=tuple(
                    float(value)
                    for value in _rotation_6d_from_surface_normal(
                        normalized[right_index], RIGHT_TANGENT_REFERENCE
                    )
                ),
                opposition_cosine=float(opposition[available_rows][selected]),
                span_m=selected_span,
                source_surface_indices=key,
                lateral_fraction=float(valid_lateral_fraction[selected]),
                side_normal_min_component=float(
                    min(normalized[left_index, 1], -normalized[right_index, 1])
                ),
                x_offset_m=float(abs(selected_delta[0])),
                z_offset_m=float(abs(selected_delta[2])),
            )
        )
        if len(seeds) >= count:
            break
    if len(seeds) < count:
        raise RuntimeError(f"only found {len(seeds)} valid opposed pairs; requested {count}")
    return sorted(
        seeds,
        key=lambda pair: (
            2.0 * pair.opposition_cosine
            + pair.lateral_fraction
            + 0.5 * pair.side_normal_min_component
            - 0.25 * (pair.x_offset_m + pair.z_offset_m) / max(pair.span_m, 1.0e-12)
        ),
        reverse=True,
    )


def sample_mesh_pairs(
    mesh_path: str | Path,
    *,
    count: int,
    seed: int,
    surface_samples: int = 20000,
    standoff_m: float = 0.015,
    minimum_opposition_cosine: float = 0.75,
    minimum_span_m: float = 0.04,
    minimum_side_normal_component: float = 0.55,
    minimum_lateral_fraction: float = 0.75,
    maximum_x_offset_m: float = 0.075,
    maximum_z_offset_m: float = 0.075,
) -> list[PairedSurfaceSeed]:
    import trimesh

    mesh = trimesh.load_mesh(Path(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    positions, face_indices = trimesh.sample.sample_surface(
        mesh, surface_samples, seed=np.random.default_rng(seed)
    )
    normals = np.asarray(mesh.face_normals)[face_indices]
    return pair_surface_samples(
        positions,
        normals,
        count=count,
        seed=seed,
        standoff_m=standoff_m,
        minimum_opposition_cosine=minimum_opposition_cosine,
        minimum_span_m=minimum_span_m,
        minimum_side_normal_component=minimum_side_normal_component,
        minimum_lateral_fraction=minimum_lateral_fraction,
        maximum_x_offset_m=maximum_x_offset_m,
        maximum_z_offset_m=maximum_z_offset_m,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--standoff-m", type=float, default=0.015)
    parser.add_argument("--minimum-opposition-cosine", type=float, default=0.75)
    parser.add_argument("--minimum-span-m", type=float, default=0.04)
    parser.add_argument("--minimum-side-normal-component", type=float, default=0.55)
    parser.add_argument("--minimum-lateral-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-x-offset-m", type=float, default=0.075)
    parser.add_argument("--maximum-z-offset-m", type=float, default=0.075)
    args = parser.parse_args()
    pairs = sample_mesh_pairs(
        args.mesh,
        count=args.count,
        surface_samples=args.surface_samples,
        seed=args.seed,
        standoff_m=args.standoff_m,
        minimum_opposition_cosine=args.minimum_opposition_cosine,
        minimum_span_m=args.minimum_span_m,
        minimum_side_normal_component=args.minimum_side_normal_component,
        minimum_lateral_fraction=args.minimum_lateral_fraction,
        maximum_x_offset_m=args.maximum_x_offset_m,
        maximum_z_offset_m=args.maximum_z_offset_m,
    )
    payload = {
        "schema": "xhand_bodex_paired_surface_seeds_v2",
        "mesh": str(args.mesh.resolve()),
        "seed": args.seed,
        "count": len(pairs),
        "pairing_constraints": {
            "minimum_opposition_cosine": args.minimum_opposition_cosine,
            "minimum_span_m": args.minimum_span_m,
            "minimum_side_normal_component": args.minimum_side_normal_component,
            "minimum_lateral_fraction": args.minimum_lateral_fraction,
            "maximum_x_offset_m": args.maximum_x_offset_m,
            "maximum_z_offset_m": args.maximum_z_offset_m,
            "standoff_m": args.standoff_m,
        },
        "seeds": [asdict(pair) for pair in pairs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "count": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
