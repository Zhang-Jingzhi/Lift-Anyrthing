#!/usr/bin/env python3
"""Create a small deterministic local neighborhood around BiDex candidates.

The output keeps the normal ``source_vis`` entry schema so it can be passed
directly to ``generate_bimanual_pilot.py``.  Only wrist radial standoff and
the 16 hand joint seeds are perturbed; hand identity and the optimized wrist
orientations are preserved.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def radial_shift(q, delta_mm):
    result = q.clone()
    direction = result[:2] / result[:2].norm().clamp_min(1e-8)
    result[:2] += direction * (float(delta_mm) / 1000.0)
    return result


def tangential_shift(q, delta_mm):
    """Translate a wrist along the object-frame XY tangent at its root."""
    result = q.clone()
    radial = result[:2] / result[:2].norm().clamp_min(1e-8)
    tangent = torch.stack((-radial[1], radial[0]))
    result[:2] += tangent * (float(delta_mm) / 1000.0)
    return result


def common_yaw_shift(q, yaw_degrees):
    """Rotate a hand root pose about the object-frame Z axis."""
    result = q.clone()
    yaw = Rotation.from_euler("Z", float(yaw_degrees), degrees=True).as_matrix()
    xyz = result[:3].detach().cpu().numpy().astype(np.float64, copy=True)
    xyz = yaw @ xyz
    orientation = Rotation.from_euler(
        "XYZ", result[3:6].detach().cpu().numpy()
    ).as_matrix()
    result[:3] = torch.as_tensor(xyz, dtype=result.dtype)
    result[3:6] = torch.as_tensor(
        Rotation.from_matrix(yaw @ orientation).as_euler("XYZ"),
        dtype=result.dtype,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-method",
        default="bidex_high_vhacd_local_neighborhood_v1",
        help="Provenance label stored on every generated candidate.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--indices", type=int, nargs="+")
    selection.add_argument(
        "--top-k",
        type=int,
        help=(
            "Expand the first K ranked candidates (or all available rows when "
            "fewer than K were produced)."
        ),
    )
    parser.add_argument(
        "--radial-mm", type=float, nargs="+", default=[-6, -4, -2, 0]
    )
    parser.add_argument("--left-radial-mm", type=float, nargs="+")
    parser.add_argument("--right-radial-mm", type=float, nargs="+")
    parser.add_argument(
        "--radial-delta-pair-mm",
        type=float,
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help=(
            "Repeatable left/right radial offset pairs. When supplied, "
            "these pairs replace the Cartesian radial grid."
        ),
    )
    parser.add_argument("--left-tangential-mm", type=float, nargs="+")
    parser.add_argument("--right-tangential-mm", type=float, nargs="+")
    parser.add_argument(
        "--common-yaw-degrees", type=float, nargs="+", default=[0.0]
    )
    parser.add_argument(
        "--yaw-delta-pair-degrees",
        type=float,
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help=(
            "Repeatable independent left/right object-frame yaw deltas. "
            "When omitted, --common-yaw-degrees is applied to both hands."
        ),
    )
    parser.add_argument("--common-z-mm", type=float, nargs="+", default=[0.0])
    parser.add_argument("--left-z-mm", type=float, nargs="+")
    parser.add_argument("--right-z-mm", type=float, nargs="+")
    parser.add_argument(
        "--z-delta-pair-mm",
        type=float,
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help=(
            "Repeatable left/right root-height delta pair. When supplied, "
            "these pairs replace the Cartesian left/right Z grid."
        ),
    )
    parser.add_argument(
        "--joint-delta-rad", type=float, nargs="+", default=[-0.04, 0, 0.04]
    )
    parser.add_argument(
        "--joint-delta-pair-rad",
        type=float,
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help=(
            "Repeatable left/right joint-seed delta pair. When supplied, "
            "these pairs replace the symmetric --joint-delta-rad grid."
        ),
    )
    parser.add_argument(
        "--flexion-delta-pair-rad",
        type=float,
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help=(
            "Repeatable extra deltas for the three flexion joints in each "
            "four-joint digit group, leaving the four ab/adduction joints "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--left-specific-joint-delta-rad",
        type=float,
        nargs=2,
        action="append",
        metavar=("JOINT_INDEX", "DELTA"),
        help=(
            "Alternative single-joint perturbations for the left Allegro "
            "seed. JOINT_INDEX is zero-based within the 16 actuated joints."
        ),
    )
    parser.add_argument(
        "--right-specific-joint-delta-rad",
        type=float,
        nargs=2,
        action="append",
        metavar=("JOINT_INDEX", "DELTA"),
        help=(
            "Alternative single-joint perturbations for the right Allegro "
            "seed. JOINT_INDEX is zero-based within the 16 actuated joints."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    entries = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(entries, list) or len(entries) != 1:
        raise RuntimeError("expected one source_vis entry")
    entry = dict(entries[0])
    source = entry["precomputed_bimanual_candidates"]
    left_source = source["left_q"]
    right_source = source["right_q"]
    source_metadata = source.get("metadata", [])
    if args.top_k is not None:
        if args.top_k <= 0:
            raise ValueError("--top-k must be positive")
        indices = list(range(min(args.top_k, len(left_source))))
    elif args.indices is not None:
        indices = args.indices
    else:
        # Preserve the smoke-search default that was validated on pitcher003.
        indices = [1, 2]
    if not indices:
        raise RuntimeError("no BiDex candidates are available to expand")
    invalid = [index for index in indices if not 0 <= index < len(left_source)]
    if invalid:
        raise IndexError(
            f"candidate indices out of range: {invalid}; available={len(left_source)}"
        )
    joint_delta_pairs = (
        [tuple(pair) for pair in args.joint_delta_pair_rad]
        if args.joint_delta_pair_rad
        else [(value, value) for value in args.joint_delta_rad]
    )
    flexion_delta_pairs = (
        [tuple(pair) for pair in args.flexion_delta_pair_rad]
        if args.flexion_delta_pair_rad
        else [(0.0, 0.0)]
    )
    flexion_indices = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
    left_specific_joint_variants = args.left_specific_joint_delta_rad or [
        (None, 0.0)
    ]
    right_specific_joint_variants = args.right_specific_joint_delta_rad or [
        (None, 0.0)
    ]
    for side, variants in (
        ("left", left_specific_joint_variants),
        ("right", right_specific_joint_variants),
    ):
        invalid = [index for index, _ in variants if index is not None and not 0 <= int(index) < 16]
        if invalid:
            raise ValueError(f"{side} specific joint indices out of range: {invalid}")
    left_radial_mm = args.left_radial_mm or args.radial_mm
    right_radial_mm = args.right_radial_mm or args.radial_mm
    radial_delta_pairs = (
        [tuple(pair) for pair in args.radial_delta_pair_mm]
        if args.radial_delta_pair_mm
        else [
            (left_delta, right_delta)
            for left_delta in left_radial_mm
            for right_delta in right_radial_mm
        ]
    )
    left_tangential_mm = args.left_tangential_mm or [0.0]
    right_tangential_mm = args.right_tangential_mm or [0.0]
    left_z_mm = args.left_z_mm or [0.0]
    right_z_mm = args.right_z_mm or [0.0]
    z_delta_pairs = (
        [tuple(pair) for pair in args.z_delta_pair_mm]
        if args.z_delta_pair_mm
        else [
            (left_delta, right_delta)
            for left_delta in left_z_mm
            for right_delta in right_z_mm
        ]
    )
    yaw_delta_pairs = (
        [tuple(pair) for pair in args.yaw_delta_pair_degrees]
        if args.yaw_delta_pair_degrees
        else [(value, value) for value in args.common_yaw_degrees]
    )

    left_rows = []
    right_rows = []
    metadata = []
    for source_index in indices:
        for left_yaw_degrees, right_yaw_degrees in yaw_delta_pairs:
            left_yaw = common_yaw_shift(
                left_source[source_index], left_yaw_degrees
            )
            right_yaw = common_yaw_shift(
                right_source[source_index], right_yaw_degrees
            )
            for common_z_mm in args.common_z_mm:
                left_common = left_yaw.clone()
                right_common = right_yaw.clone()
                left_common[2] += float(common_z_mm) / 1000.0
                right_common[2] += float(common_z_mm) / 1000.0
                for left_z_delta_mm, right_z_delta_mm in z_delta_pairs:
                    left_shifted = left_common.clone()
                    right_shifted = right_common.clone()
                    left_shifted[2] += float(left_z_delta_mm) / 1000.0
                    right_shifted[2] += float(right_z_delta_mm) / 1000.0
                    for left_mm, right_mm in radial_delta_pairs:
                            for left_tangent_mm in left_tangential_mm:
                                for right_tangent_mm in right_tangential_mm:
                                    for left_joint_delta, right_joint_delta in joint_delta_pairs:
                                        for left_flexion_delta, right_flexion_delta in flexion_delta_pairs:
                                            for left_specific_index, left_specific_delta in left_specific_joint_variants:
                                                for right_specific_index, right_specific_delta in right_specific_joint_variants:
                                                    left_q = tangential_shift(
                                                        radial_shift(left_shifted, left_mm),
                                                        left_tangent_mm,
                                                    )
                                                    right_q = tangential_shift(
                                                        radial_shift(right_shifted, right_mm),
                                                        right_tangent_mm,
                                                    )
                                                    left_q[6:] += float(left_joint_delta)
                                                    right_q[6:] += float(right_joint_delta)
                                                    left_q[
                                                        [6 + index for index in flexion_indices]
                                                    ] += float(left_flexion_delta)
                                                    right_q[
                                                        [6 + index for index in flexion_indices]
                                                    ] += float(right_flexion_delta)
                                                    if left_specific_index is not None:
                                                        left_q[6 + int(left_specific_index)] += float(
                                                            left_specific_delta
                                                        )
                                                    if right_specific_index is not None:
                                                        right_q[6 + int(right_specific_index)] += float(
                                                            right_specific_delta
                                                        )
                                                    left_rows.append(left_q)
                                                    right_rows.append(right_q)
                                                    base = (
                                                        dict(source_metadata[source_index])
                                                        if source_index < len(source_metadata)
                                                        else {}
                                                    )
                                                    metadata.append(
                                                        {
                                                    **base,
                                                    "candidate_method": args.candidate_method,
                                                    "source_pair_index": len(metadata),
                                                    "high_vhacd_local_parent_index": source_index,
                                                    "high_vhacd_common_yaw_degrees": (
                                                        float(left_yaw_degrees)
                                                        if left_yaw_degrees
                                                        == right_yaw_degrees
                                                        else None
                                                    ),
                                                    "high_vhacd_left_yaw_delta_degrees": float(
                                                        left_yaw_degrees
                                                    ),
                                                    "high_vhacd_right_yaw_delta_degrees": float(
                                                        right_yaw_degrees
                                                    ),
                                                    "high_vhacd_common_z_mm": float(common_z_mm),
                                                    "high_vhacd_left_z_delta_mm": float(
                                                        left_z_delta_mm
                                                    ),
                                                    "high_vhacd_right_z_delta_mm": float(
                                                        right_z_delta_mm
                                                    ),
                                                    "high_vhacd_left_radial_delta_mm": float(
                                                        left_mm
                                                    ),
                                                    "high_vhacd_right_radial_delta_mm": float(
                                                        right_mm
                                                    ),
                                                    "high_vhacd_left_tangential_delta_mm": float(
                                                        left_tangent_mm
                                                    ),
                                                    "high_vhacd_right_tangential_delta_mm": float(
                                                        right_tangent_mm
                                                    ),
                                                    "high_vhacd_joint_delta_rad": (
                                                        float(left_joint_delta)
                                                        if left_joint_delta
                                                        == right_joint_delta
                                                        else None
                                                    ),
                                                    "high_vhacd_left_joint_delta_rad": float(
                                                        left_joint_delta
                                                    ),
                                                    "high_vhacd_right_joint_delta_rad": float(
                                                        right_joint_delta
                                                    ),
                                                    "high_vhacd_left_flexion_delta_rad": float(
                                                        left_flexion_delta
                                                    ),
                                                    "high_vhacd_right_flexion_delta_rad": float(
                                                        right_flexion_delta
                                                    ),
                                                    "high_vhacd_left_specific_joint_index": (
                                                        int(left_specific_index)
                                                        if left_specific_index is not None
                                                        else None
                                                    ),
                                                    "high_vhacd_left_specific_joint_delta_rad": float(
                                                        left_specific_delta
                                                    ),
                                                    "high_vhacd_right_specific_joint_index": (
                                                        int(right_specific_index)
                                                        if right_specific_index is not None
                                                        else None
                                                    ),
                                                    "high_vhacd_right_specific_joint_delta_rad": float(
                                                        right_specific_delta
                                                    ),
                                                }
                                            )

    entry["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_rows),
        "right_q": torch.stack(right_rows),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([entry], args.output)
    print(
        f"saved {len(metadata)} local candidates "
        f"method={args.candidate_method} -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
