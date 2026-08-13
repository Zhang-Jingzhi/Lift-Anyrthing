#!/usr/bin/env python3
"""Paper-faithful BiDexGrasp-style initialization for strict TRO validation.

The official BiDexGrasp code is not public as of 2026-08-05.  This script
implements the two public synthesis ideas needed for an auditable comparison:
GWS-ranked surface-region pairs and decoupled per-hand local optimization.
The resulting candidates are not labeled successful here; the unchanged TRO
Isaac pipeline performs the final geometry, lift, disturbance, and ablations.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_bimanual_pilot import (
    closure_path_min_world_z_mm,
    hand_clearances,
    place_on_ray,
)
from scripts.audit_decoupled_force_closure import (
    grasp_matrix,
    wrench_coverage,
)
from utils.controller import controller
from utils.hand_model import create_hand_model


def enhanced_bidex(args):
    return args.enhanced_bidex_v2 or args.enhanced_bidex_v3


def method_name(args):
    if args.enhanced_bidex_v3:
        return "bidexgrasp_multi_parent_gws_joint_local_quality_fps_v3"
    if args.enhanced_bidex_v2:
        return "bidexgrasp_region_gws_ik_decoupled_local_v2"
    return "bidexgrasp_region_gws_decoupled_local_v1"


def contact_digit(link_name):
    """Map Allegro link names to one of its four physical digits."""
    if not link_name.startswith("link_"):
        return None
    try:
        link_index = int(link_name.split("_", 1)[1].split(".", 1)[0])
    except ValueError:
        return None
    return min(link_index // 4, 3)


def farthest_point_indices(points, count, seed):
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(points)))]
    minimum = np.full(len(points), np.inf)
    for _ in range(1, min(count, len(points))):
        delta = points - points[selected[-1]]
        minimum = np.minimum(minimum, np.einsum("ij,ij->i", delta, delta))
        selected.append(int(np.argmax(minimum)))
    return np.asarray(selected, dtype=np.int64)


def tangent_basis(normal):
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, helper)
    first /= np.linalg.norm(first) + 1e-12
    second = np.cross(normal, first)
    second /= np.linalg.norm(second) + 1e-12
    return first, second


def region_wrenches(points, normals, center, radius, friction):
    columns = []
    for point, outward in zip(points, normals):
        inward = -outward / (np.linalg.norm(outward) + 1e-12)
        first, second = tangent_basis(inward)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            force = inward + friction * (sx * first + sy * second)
            force /= np.linalg.norm(force) + 1e-12
            torque = np.cross(point - center, force) / max(radius, 1e-6)
            columns.append(np.concatenate([force, torque]))
    return np.stack(columns, axis=1)


def gws_metrics(left_region, right_region, center, radius, friction):
    matrix = np.concatenate(
        [
            region_wrenches(
                left_region["gws_points"],
                left_region["gws_normals"],
                center,
                radius,
                friction,
            ),
            region_wrenches(
                right_region["gws_points"],
                right_region["gws_normals"],
                center,
                radius,
                friction,
            ),
        ],
        axis=1,
    )
    disturbances = np.concatenate([np.eye(6), -np.eye(6)], axis=0)
    # This is the support-function form of Eq. (3).  Taking the worst of the
    # signed orthogonal directions prevents a one-sided wrench set ranking.
    support = np.asarray(
        [float(np.max(direction @ matrix)) for direction in disturbances]
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma_min = float(singular_values[-1])
    isotropy = float(sigma_min / max(float(singular_values[0]), 1e-12))
    return {
        "gws_score": float(support.min()),
        "gws_mean_support": float(support.mean()),
        "gws_sigma_min": sigma_min,
        "gws_isotropy": isotropy,
    }


def build_regions(mesh, args):
    rng_state = np.random.get_state()
    np.random.seed(args.seed)
    points, faces = trimesh.sample.sample_surface_even(
        mesh, args.surface_points
    )
    np.random.set_state(rng_state)
    normals = mesh.face_normals[faces]
    anchor_indices = farthest_point_indices(points, args.anchors, args.seed)
    center = np.asarray(mesh.bounding_box.centroid)
    bottom, top = mesh.bounds[:, 2]
    height = top - bottom
    region_radius = float(
        np.clip(args.region_scale * np.linalg.norm(mesh.extents), 0.025, 0.08)
    )
    hull_query = trimesh.proximity.ProximityQuery(mesh.convex_hull)
    _, hull_distance, _ = hull_query.on_surface(points[anchor_indices])
    regions = []
    for anchor_order, point_index in enumerate(anchor_indices):
        anchor = points[point_index]
        if not (
            bottom + args.minimum_height_fraction * height
            <= anchor[2]
            <= bottom + args.maximum_height_fraction * height
        ):
            continue
        radial = anchor[:2] - center[:2]
        if np.linalg.norm(radial) < args.minimum_radial_fraction * max(
            mesh.extents[0], mesh.extents[1]
        ):
            continue
        # BiDexGrasp removes concave regions.  Distance to the convex hull is
        # a deterministic mesh-only proxy for that public criterion.
        if hull_distance[anchor_order] > 0.30 * region_radius:
            continue
        distance = np.linalg.norm(points - anchor, axis=1)
        neighborhood = np.argsort(distance)[: args.region_points]
        sample = neighborhood[
            np.linspace(0, len(neighborhood) - 1, args.gws_contacts).astype(int)
        ]
        mean_normal = normals[neighborhood].mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-12
        normal_alignment = np.clip(normals[neighborhood] @ mean_normal, -1.0, 1.0)
        regions.append(
            {
                "anchor_index": int(anchor_order),
                "anchor": anchor,
                "normal": mean_normal,
                "normal_consistency": float(normal_alignment.mean()),
                "gws_points": points[sample],
                "gws_normals": normals[sample],
            }
        )
    return regions, region_radius


def rank_region_pairs(mesh, regions, args):
    center = np.asarray(mesh.bounding_box.centroid)
    radius = float(np.linalg.norm(mesh.extents) * 0.5)
    minimum_separation = args.minimum_pair_separation_fraction * max(
        mesh.extents[0], mesh.extents[1]
    )
    rows = []
    for left_index, left in enumerate(regions):
        left_xy = left["anchor"][:2] - center[:2]
        left_xy /= np.linalg.norm(left_xy) + 1e-12
        for right_index in range(left_index + 1, len(regions)):
            right = regions[right_index]
            right_xy = right["anchor"][:2] - center[:2]
            right_xy /= np.linalg.norm(right_xy) + 1e-12
            separation = float(
                np.linalg.norm(left["anchor"] - right["anchor"])
            )
            opposition = float(np.dot(left_xy, right_xy))
            normal_opposition = float(np.dot(left["normal"], right["normal"]))
            left_radial_normal = float(np.dot(left["normal"][:2], left_xy))
            right_radial_normal = float(np.dot(right["normal"][:2], right_xy))
            vertical_separation = abs(
                float(left["anchor"][2] - right["anchor"][2])
            )
            if (
                separation < minimum_separation
                or opposition > -0.35
                or (
                    enhanced_bidex(args)
                    and normal_opposition > args.maximum_normal_opposition_cosine
                )
                or (
                    enhanced_bidex(args)
                    and min(left_radial_normal, right_radial_normal)
                    < args.minimum_radial_normal_alignment
                )
                or vertical_separation
                < args.minimum_vertical_separation_mm / 1000.0
                or vertical_separation
                > args.maximum_vertical_separation_mm / 1000.0
            ):
                continue
            metrics = gws_metrics(left, right, center, radius, args.friction)
            rows.append(
                {
                    "left_region_index": left_index,
                    "right_region_index": right_index,
                    "separation_m": separation,
                    "vertical_separation_m": vertical_separation,
                    "horizontal_opposition_cosine": opposition,
                    "surface_normal_opposition_cosine": normal_opposition,
                    "minimum_radial_normal_alignment": min(
                        left_radial_normal, right_radial_normal
                    ),
                    **metrics,
                }
            )
    if enhanced_bidex(args):
        rows.sort(
            key=lambda row: (
                row["gws_score"],
                row["gws_isotropy"],
                row["gws_sigma_min"],
                -abs(row["horizontal_opposition_cosine"] + 1.0),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: row["gws_score"], reverse=True)
    return rows[: args.region_pairs]


def transformed_links(hand, q):
    links, _ = hand.get_transformed_links_pc(q)
    return {
        name: points.detach().cpu().numpy()
        for name, points in links.items()
    }


def pose_audit(hand, q, mesh, query, region_anchor, contact_m, friction):
    links = transformed_links(hand, q)
    all_points = np.concatenate(list(links.values()), axis=0)
    closest, distance, triangle = query.on_surface(all_points)
    # A nearest-face normal is not a valid inside/outside test near concave
    # features (for example, the cavity around a pitcher handle).  It caused
    # exterior fingertip points to be reported as deeply penetrating.  Match
    # the downstream TRO geometry audit: only points classified inside the
    # closed mesh contribute a penetration depth.
    inside = mesh.contains(all_points)
    penetration = float(distance[inside].max()) if inside.any() else 0.0
    contact_mask = distance <= contact_m
    contacts = []
    contact_links = []
    contact_digits = set()
    contact_positions = []
    offset = 0
    for link_name, points in links.items():
        link_distance = distance[offset : offset + len(points)]
        link_closest = closest[offset : offset + len(points)]
        link_triangle = triangle[offset : offset + len(points)]
        indices = np.flatnonzero(link_distance <= contact_m)
        if len(indices):
            contact_links.append(link_name)
            digit = contact_digit(link_name)
            if digit is not None:
                contact_digits.add(digit)
            for index in indices[np.argsort(link_distance[indices])[:4]]:
                inward = -mesh.face_normals[link_triangle[index]]
                inward /= np.linalg.norm(inward) + 1e-12
                contacts.append((link_closest[index], inward, link_name))
                contact_positions.append(link_closest[index])
        offset += len(points)
    radius = float(np.linalg.norm(mesh.extents) * 0.5)
    matrix = grasp_matrix(
        contacts,
        np.asarray(mesh.bounding_box.centroid),
        radius,
        friction,
    )
    residuals, _ = wrench_coverage(matrix)
    region_distance = (
        float(np.linalg.norm(closest[contact_mask] - region_anchor, axis=1).mean())
        if contact_mask.any()
        else float(distance.min() + contact_m)
    )
    if len(contact_positions) > 1:
        positions = np.asarray(contact_positions)
        contact_spread_mm = float(
            np.linalg.norm(positions[:, None] - positions[None, :], axis=-1).max()
            * 1000.0
        )
    else:
        contact_spread_mm = 0.0
    return {
        "penetration_mm": penetration * 1000.0,
        "surface_distance_mm": float(distance.min() * 1000.0),
        "contact_link_count": len(contact_links),
        "contact_point_count": int(contact_mask.sum()),
        "contact_digit_count": len(contact_digits),
        "contact_spread_mm": contact_spread_mm,
        "mean_wrench_residual": float(np.mean(residuals)),
        "max_wrench_residual": float(np.max(residuals)),
        "region_distance_mm": region_distance * 1000.0,
        "min_world_z_mm": float(all_points[:, 2].min() * 1000.0),
    }


def hand_objective(outer, inner, support_z_mm, args):
    base = float(
        40.0 * max(0.0, outer["penetration_mm"] - args.penetration_mm) ** 2
        + 8.0 * max(0.0, inner["surface_distance_mm"] - args.contact_mm) ** 2
        + 6.0 * max(0, args.minimum_contact_links - inner["contact_link_count"])
        + 5.0 * inner["mean_wrench_residual"]
        + 0.02 * inner["region_distance_mm"]
        + 20.0 * max(
            0.0,
            support_z_mm + args.support_clearance_mm - outer["min_world_z_mm"],
        )
    )
    if not enhanced_bidex(args):
        return base
    # V2 strongly prefers redundant, well-centered contacts instead of
    # allowing a single near-surface point to dominate the local ranking.
    return float(
        base
        + 80.0
        * max(0.0, inner["penetration_mm"] - args.penetration_mm) ** 2
        + 18.0
        * max(0, args.target_contact_links - inner["contact_link_count"]) ** 2
        + 0.35 * max(0, args.target_contact_points - inner["contact_point_count"])
        + 12.0 * max(0, args.target_contact_digits - inner["contact_digit_count"]) ** 2
        + 0.12 * max(0.0, args.target_contact_spread_mm - inner["contact_spread_mm"])
        + 3.0 * inner["max_wrench_residual"]
        + 0.06 * inner["region_distance_mm"]
    )


def reachability_pass(outer, inner, support_z_mm, args):
    """TRO has free 6-DoF hand roots, so this is a hand-workspace IK proxy.

    It verifies that a sourced Allegro joint posture can close from a
    collision-free outer pose onto the selected region with redundant links.
    Arm IK cannot be checked because this project contains no arm model.
    """
    return bool(
        outer["penetration_mm"] <= args.penetration_mm
        and inner["penetration_mm"] <= args.penetration_mm
        and inner["surface_distance_mm"] <= args.contact_mm
        and inner["contact_link_count"] >= args.minimum_contact_links
        and inner["contact_point_count"] >= args.minimum_contact_points
        and inner["contact_digit_count"] >= args.minimum_contact_digits
        and outer["min_world_z_mm"]
        >= support_z_mm + args.support_clearance_mm - 1e-6
    )


def reachability_failures(outer, inner, support_z_mm, args):
    """Return auditable reasons for rejecting a decoupled hand solution."""
    failures = []
    checks = (
        ("outer_penetration", outer["penetration_mm"] <= args.penetration_mm),
        ("inner_penetration", inner["penetration_mm"] <= args.penetration_mm),
        ("surface_distance", inner["surface_distance_mm"] <= args.contact_mm),
        ("contact_links", inner["contact_link_count"] >= args.minimum_contact_links),
        ("contact_points", inner["contact_point_count"] >= args.minimum_contact_points),
        ("contact_digits", inner["contact_digit_count"] >= args.minimum_contact_digits),
        (
            "table_clearance",
            outer["min_world_z_mm"]
            >= support_z_mm + args.support_clearance_mm - 1e-6,
        ),
    )
    for name, passed in checks:
        if not passed:
            failures.append(name)
    return failures


def evaluate_seed(hand, q, mesh, query, anchor, support_z_mm, args):
    outer_q, inner_q = controller(hand.robot_name, q, hand=hand)
    outer = pose_audit(
        hand, outer_q, mesh, query, anchor, args.contact_mm / 1000.0, args.friction
    )
    inner = pose_audit(
        hand, inner_q, mesh, query, anchor, args.contact_mm / 1000.0, args.friction
    )
    return hand_objective(outer, inner, support_z_mm, args), outer, inner


def refine_finger_joints(
    hand,
    best,
    mesh,
    query,
    region_anchor,
    support_z_mm,
    args,
):
    """Coordinate-descent refinement over the 16 Allegro finger joints.

    The v1 approximation optimized only the free wrist pose.  A region can be
    reachable by the hand while still failing that approximation because the
    source posture came from another object.  Refining the finger joints after
    wrist placement is the missing decoupled grasp-optimization step.
    """
    if not enhanced_bidex(args) or not args.v2_joint_steps_degrees:
        return best
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=best[1].dtype)
    upper = torch.as_tensor(upper, dtype=best[1].dtype)
    for step_degrees in args.v2_joint_steps_degrees:
        step = math.radians(step_degrees)
        # Use Gauss-Seidel coordinate descent: every accepted joint update is
        # immediately visible to the following joint's evaluation.
        for joint_index in range(6, len(best[1])):
            center_q = best[1].clone()
            for direction in (-1.0, 1.0):
                q = center_q.clone()
                q[joint_index] = torch.clamp(
                    q[joint_index] + direction * step,
                    lower[joint_index],
                    upper[joint_index],
                )
                objective, outer, inner = evaluate_seed(
                    hand,
                    q,
                    mesh,
                    query,
                    region_anchor,
                    support_z_mm,
                    args,
                )
                row = (
                    objective,
                    q,
                    best[2],
                    best[3],
                    best[4],
                    outer,
                    inner,
                )
                if row[0] < best[0]:
                    best = row
    return best


def refine_global_aperture(
    hand,
    best,
    mesh,
    query,
    region_anchor,
    support_z_mm,
    args,
):
    """Search a coordinated collision-free finger aperture.

    A large object can trap coordinate descent because moving only one joint
    does not remove simultaneous penetration by the other fingers.  The TRO
    controller already supplies a valid opening direction, so search along it
    first and let the existing contact/link objective retain useful closure.
    """
    if not enhanced_bidex(args) or not args.v2_aperture_scales:
        return best
    center_q = best[1].clone()
    open_q, _ = controller(hand.robot_name, center_q, hand=hand)
    open_direction = open_q[6:] - center_q[6:]
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=center_q.dtype)
    upper = torch.as_tensor(upper, dtype=center_q.dtype)
    for scale in args.v2_aperture_scales:
        q = center_q.clone()
        q[6:] = torch.clamp(
            center_q[6:] + float(scale) * open_direction,
            lower[6:],
            upper[6:],
        )
        objective, outer, inner = evaluate_seed(
            hand,
            q,
            mesh,
            query,
            region_anchor,
            support_z_mm,
            args,
        )
        row = (
            objective,
            q,
            best[2],
            best[3],
            best[4],
            outer,
            inner,
        )
        if row[0] < best[0]:
            best = row
    return best


def simultaneous_finger_variants(
    hand,
    best,
    mesh,
    query,
    region_anchor,
    support_z_mm,
    side,
    region_index,
    args,
):
    """Perturb all four digits together and retain robust local optima.

    V2's coordinate descent can only change one joint at a time.  That is a
    poor local proposal for a large surface: useful grasps often require two
    or more fingers to open, curl, or splay together.  V3 therefore samples a
    deterministic block perturbation over every finger, evaluates the whole
    closed-hand configuration, and keeps a small quality-ranked side pool.
    """
    if not args.enhanced_bidex_v3 or args.v3_finger_perturbations <= 0:
        return [best]
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=best[1].dtype)
    upper = torch.as_tensor(upper, dtype=best[1].dtype)
    side_offset = 0 if side == "left" else 1_000_003
    rng = np.random.default_rng(
        args.seed + side_offset + 7_919 * int(region_index) + int(best[2])
    )
    # A rejected wrist/joint optimum is still a useful parent for V3 joint
    # exploration, but it must never leak into the returned feasible pool.
    # This is particularly important for large, anisotropically scaled
    # objects where the sourced posture often starts with only one contacting
    # digit and needs a coordinated multi-finger correction.
    rows = []
    if reachability_pass(best[5], best[6], support_z_mm, args):
        rows.append(best)
    max_radians = math.radians(args.v3_finger_perturb_degrees)
    outer_q, _ = controller(hand.robot_name, best[1], hand=hand)
    opening = (outer_q[6:] - best[1][6:]).reshape(4, 4)
    # Move opposite to the controller's opening direction when asking a digit
    # to close. Fall back to the positive joint direction for zero entries so
    # that every deterministic proposal remains active.
    closing_sign = -torch.sign(opening)
    closing_sign[closing_sign == 0] = 1
    deterministic_digit_weights = (
        (1.0, 1.0, 1.0, 1.0),  # coordinated extra closure
        (-1.0, -1.0, -1.0, -1.0),  # coordinated opening/penetration escape
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 0.0),
    )
    for perturbation_index in range(args.v3_finger_perturbations):
        # Systematic proposals repair missing contacts before randomized
        # exploration: all digits, individual digits, then complementary
        # digit groups. Additional samples diversify the neighborhood.
        deterministic_count = 2 * len(deterministic_digit_weights)
        if perturbation_index < deterministic_count:
            pattern_index = perturbation_index % len(deterministic_digit_weights)
            amplitude_scale = 0.5 if perturbation_index < len(
                deterministic_digit_weights
            ) else 1.0
            weights = torch.as_tensor(
                deterministic_digit_weights[pattern_index],
                dtype=best[1].dtype,
            ).reshape(4, 1)
            delta = closing_sign * weights * max_radians * amplitude_scale
        else:
            digit_offsets = rng.uniform(-max_radians, max_radians, size=(4, 1))
            joint_noise = rng.normal(0.0, max_radians * 0.28, size=(4, 4))
            delta = torch.as_tensor(
                digit_offsets + joint_noise, dtype=best[1].dtype
            )
        # Distal joints receive the full curl proposal; the ab/adduction joint
        # receives a smaller displacement to avoid implausible finger splay.
        delta[:, 0] *= 0.35
        q = best[1].clone()
        q[6:] = torch.clamp(
            q[6:] + delta.reshape(-1),
            lower[6:],
            upper[6:],
        )
        objective, outer, inner = evaluate_seed(
            hand, q, mesh, query, region_anchor, support_z_mm, args
        )
        row = (
            objective,
            q,
            best[2],
            best[3],
            best[4],
            outer,
            inner,
        )
        if reachability_pass(outer, inner, support_z_mm, args):
            rows.append(row)
    unique = {}
    for row in rows:
        key = tuple(torch.round(row[1] * 1e5).to(torch.int64).tolist())
        previous = unique.get(key)
        if previous is None or row[0] < previous[0]:
            unique[key] = row
    return sorted(unique.values(), key=lambda row: row[0])[
        : args.v3_side_variants
    ]


def optimize_hand(
    hand,
    seeds,
    source_indices,
    region,
    mesh,
    query,
    side,
    support_z_mm,
    args,
):
    center = np.asarray(mesh.bounding_box.centroid)
    direction = region["anchor"][:2] - center[:2]
    direction /= np.linalg.norm(direction) + 1e-12
    target_direction = np.array([direction[0], direction[1], 0.0])
    tangent_direction = np.array([-direction[1], direction[0]])
    surface_radius = float(np.linalg.norm(region["anchor"][:2] - center[:2]))
    coarse_bests = []
    coarse_by_standoff = {}
    coarse_rolls = args.left_rolls if side == "left" else args.right_rolls
    for source_offset, seed in enumerate(seeds):
        source_index = int(source_indices[source_offset])
        for roll_degrees in coarse_rolls:
            aligned = place_on_ray(seed, target_direction, roll_degrees)
            aligned[2] = float(region["anchor"][2])
            local_best = None
            for standoff_mm in np.arange(
                args.minimum_standoff_mm,
                args.maximum_standoff_mm + 0.1,
                args.standoff_step_mm,
            ):
                q = aligned.clone()
                radius = surface_radius + standoff_mm / 1000.0
                q[:2] = torch.as_tensor(direction * radius, dtype=q.dtype)
                objective, outer, inner = evaluate_seed(
                    hand, q, mesh, query, region["anchor"], support_z_mm, args
                )
                row = (
                    objective,
                    q,
                    source_index,
                    float(roll_degrees),
                    float(standoff_mm),
                    outer,
                    inner,
                )
                coarse_by_standoff.setdefault(float(standoff_mm), []).append(
                    row
                )
                if local_best is None or row[0] < local_best[0]:
                    local_best = row
            coarse_bests.append(local_best)

    variants = []
    rejected = []
    if args.enhanced_bidex_v3:
        # Multiple opposition rolls must survive initialization, not merely be
        # evaluated and then collapsed by the standoff-wise minimum. Keep the
        # best source/standoff parent for every requested roll first; only
        # then fill any remaining side-parent budget by objective.
        selected_coarse = []
        selected_keys = set()
        for roll_degrees in coarse_rolls:
            roll_rows = [
                row
                for row in coarse_bests
                if abs(row[3] - float(roll_degrees)) < 1e-8
            ]
            if not roll_rows:
                continue
            row = min(roll_rows, key=lambda candidate: candidate[0])
            key = tuple(torch.round(row[1] * 1e5).to(torch.int64).tolist())
            if key not in selected_keys:
                selected_coarse.append(row)
                selected_keys.add(key)
        for row in sorted(coarse_bests, key=lambda candidate: candidate[0]):
            if len(selected_coarse) >= args.hand_variants:
                break
            key = tuple(torch.round(row[1] * 1e5).to(torch.int64).tolist())
            if key not in selected_keys:
                selected_coarse.append(row)
                selected_keys.add(key)
        selected_coarse = selected_coarse[: args.hand_variants]
    elif args.preserve_standoff_diversity:
        # Pair-level coordination needs alternatives at different wrist
        # radii. Keeping only each source/roll's best radius made the
        # clearance penalty ineffective because all returned modes used
        # nearly identical hand-to-object standoffs.
        diverse = [
            min(rows, key=lambda row: row[0])
            for _, rows in sorted(coarse_by_standoff.items())
        ]
        selected_coarse = sorted(diverse, key=lambda row: row[0])[
            : args.hand_variants
        ]
    else:
        selected_coarse = sorted(coarse_bests, key=lambda row: row[0])[
            : args.hand_variants
        ]
    for coarse in selected_coarse:
        best = coarse
        # Local coordinate refinement around each diverse source/roll mode.
        tangent_offsets = (
            (-6.0, 0.0, 6.0) if enhanced_bidex(args) else (0.0,)
        )
        refinement_steps = (
            ((8.0, 5.0), (4.0, 2.5), (2.0, 1.0))
            if enhanced_bidex(args)
            else ((5.0, 3.0),)
        )
        if enhanced_bidex(args):
            refinement_steps = refinement_steps[: args.v2_refinement_levels]
        for z_step_mm, radial_step_mm in refinement_steps:
            center_best = best
            for delta_z_mm in (-z_step_mm, 0.0, z_step_mm):
                for delta_radius_mm in (
                    -radial_step_mm,
                    0.0,
                    radial_step_mm,
                ):
                    for delta_tangent_mm in tangent_offsets:
                        q = center_best[1].clone()
                        q[2] += delta_z_mm / 1000.0
                        q[:2] += torch.as_tensor(
                            (
                                direction * delta_radius_mm
                                + tangent_direction * delta_tangent_mm
                            )
                            / 1000.0,
                            dtype=q.dtype,
                        )
                        objective, outer, inner = evaluate_seed(
                            hand,
                            q,
                            mesh,
                            query,
                            region["anchor"],
                            support_z_mm,
                            args,
                        )
                        row = (
                            objective,
                            q,
                            coarse[2],
                            coarse[3],
                            center_best[4] + delta_radius_mm,
                            outer,
                            inner,
                        )
                        if row[0] < best[0]:
                            best = row

        best = refine_global_aperture(
            hand,
            best,
            mesh,
            query,
            region["anchor"],
            support_z_mm,
            args,
        )
        best = refine_finger_joints(
            hand,
            best,
            mesh,
            query,
            region["anchor"],
            support_z_mm,
            args,
        )

        # Project onto the hard tabletop collision constraint using the full
        # outer-to-inner trajectory, not just its endpoint configurations.
        for _ in range(2):
            outer_q, inner_q = controller(hand.robot_name, best[1], hand=hand)
            path_min_z_mm = closure_path_min_world_z_mm(
                hand,
                outer_q,
                inner_q,
                sample_count=5,
            )
            minimum_z_mm = min(
                best[5]["min_world_z_mm"],
                best[6]["min_world_z_mm"],
                path_min_z_mm,
            )
            deficit_mm = (
                support_z_mm + args.support_clearance_mm - minimum_z_mm
            )
            if deficit_mm <= 0.0:
                break
            q = best[1].clone()
            q[2] += (deficit_mm + 0.25) / 1000.0
            objective, outer, inner = evaluate_seed(
                hand, q, mesh, query, region["anchor"], support_z_mm, args
            )
            best = (
                objective,
                q,
                best[2],
                best[3],
                best[4],
                outer,
                inner,
            )
        if args.enhanced_bidex_v3:
            variants.extend(
                simultaneous_finger_variants(
                    hand,
                    best,
                    mesh,
                    query,
                    region["anchor"],
                    support_z_mm,
                    side,
                    region["anchor_index"],
                    args,
                )
            )
        elif not enhanced_bidex(args) or reachability_pass(
            best[5], best[6], support_z_mm, args
        ):
            variants.append(best)
        else:
            rejected.append(best)
        if args.enhanced_bidex_v3 and not reachability_pass(
            best[5], best[6], support_z_mm, args
        ):
            rejected.append(best)
    if enhanced_bidex(args) and not variants and rejected:
        best_rejected = min(rejected, key=lambda row: row[0])
        outer, inner = best_rejected[5], best_rejected[6]
        print(
            f"region={region['anchor_index']} side={side} best_rejected "
            f"failures={reachability_failures(outer, inner, support_z_mm, args)} "
            f"outer_pen={outer['penetration_mm']:.2f}mm "
            f"inner_pen={inner['penetration_mm']:.2f}mm "
            f"surface={inner['surface_distance_mm']:.2f}mm "
            f"links={inner['contact_link_count']} "
            f"points={inner['contact_point_count']} "
            f"outer_min_z={outer['min_world_z_mm']:.2f}mm",
            flush=True,
        )
    variants.sort(key=lambda row: row[0])
    if args.enhanced_bidex_v3:
        variants = variants[: args.v3_side_variants]
    return variants


def serializable_region(region):
    return {
        "anchor_index": region["anchor_index"],
        "anchor_m": region["anchor"].tolist(),
        "normal": region["normal"].tolist(),
    }


def quality_constrained_fps(left_candidates, right_candidates, metadata, args):
    """Select a quality-gated, parent-balanced and pose-diverse V3 subset."""
    if not args.enhanced_bidex_v3:
        return list(range(len(metadata)))
    by_parent = {}
    for index, row in enumerate(metadata):
        by_parent.setdefault(row["region_pair_order"], []).append(index)

    # Quality is constrained within each GWS parent, so a globally dominant
    # easy region cannot erase every other parent before diversity selection.
    eligible = []
    for parent_indices in by_parent.values():
        ranked = sorted(
            parent_indices,
            key=lambda index: metadata[index]["pair_local_objective"],
        )
        keep = max(1, int(math.ceil(len(ranked) * args.v3_quality_keep_fraction)))
        eligible.extend(ranked[:keep])
    eligible = sorted(
        eligible,
        key=lambda index: metadata[index]["pair_local_objective"],
    )
    target = min(args.v3_final_candidates, len(eligible))
    if target <= 0:
        return []

    feature_rows = []
    for index in eligible:
        left = left_candidates[index].detach().cpu().numpy().astype(np.float64)
        right = right_candidates[index].detach().cpu().numpy().astype(np.float64)
        feature = np.concatenate([left, right])
        # Translation, wrist orientation, and finger articulation should all
        # contribute materially despite their different numerical scales.
        per_hand_scale = np.concatenate(
            [np.full(3, 0.035), np.full(3, 0.30), np.full(len(left) - 6, 0.20)]
        )
        feature_rows.append(feature / np.concatenate([per_hand_scale, per_hand_scale]))
    features = np.asarray(feature_rows)
    eligible_position = {index: position for position, index in enumerate(eligible)}
    selected = [eligible[0]]
    minimum_distance = np.linalg.norm(features - features[0], axis=1)

    # First cover distinct quality-qualified parents, then run ordinary FPS.
    represented = {metadata[selected[0]]["region_pair_order"]}
    while len(selected) < target:
        unrepresented = [
            index
            for index in eligible
            if index not in selected
            and metadata[index]["region_pair_order"] not in represented
        ]
        pool = unrepresented or [
            index for index in eligible if index not in selected
        ]
        next_index = max(
            pool,
            key=lambda index: (
                minimum_distance[eligible_position[index]],
                -metadata[index]["pair_local_objective"],
            ),
        )
        next_position = eligible_position[next_index]
        selected.append(next_index)
        represented.add(metadata[next_index]["region_pair_order"])
        distance = np.linalg.norm(features - features[next_position], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)

    for selection_rank, index in enumerate(selected):
        metadata[index]["v3_diversity_selection_rank"] = selection_rank
        metadata[index]["v3_quality_keep_fraction"] = args.v3_quality_keep_fraction
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument("--minimum-object-horizontal-span-mm", type=float, default=0.0)
    parser.add_argument("--minimum-object-height-mm", type=float, default=0.0)
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--region-points", type=int, default=256)
    parser.add_argument("--gws-contacts", type=int, default=5)
    parser.add_argument("--region-pairs", type=int, default=8)
    parser.add_argument("--region-scale", type=float, default=0.22)
    parser.add_argument("--enhanced-bidex-v2", action="store_true")
    parser.add_argument("--enhanced-bidex-v3", action="store_true")
    parser.add_argument(
        "--v2-refinement-levels", type=int, choices=(1, 2, 3), default=3
    )
    parser.add_argument(
        "--v2-joint-steps-degrees",
        type=float,
        nargs="*",
        default=[6.0, 3.0],
        help="Finger-joint coordinate-descent step sizes for enhanced v2.",
    )
    parser.add_argument(
        "--v2-aperture-scales",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 1.5, 2.0],
        help=(
            "Multiples of the controller opening direction searched before "
            "per-joint refinement for enhanced v2."
        ),
    )
    parser.add_argument(
        "--v3-parent-candidate-cap",
        type=int,
        default=12,
        help="Maximum retained candidates from any one GWS region-pair parent.",
    )
    parser.add_argument("--v3-finger-perturbations", type=int, default=8)
    parser.add_argument("--v3-finger-perturb-degrees", type=float, default=4.0)
    parser.add_argument("--v3-side-variants", type=int, default=6)
    parser.add_argument(
        "--v3-quality-keep-fraction",
        type=float,
        default=0.6,
        help="Per-parent quality fraction admitted to final FPS selection.",
    )
    parser.add_argument("--v3-final-candidates", type=int, default=48)
    parser.add_argument(
        "--maximum-normal-opposition-cosine", type=float, default=-0.25
    )
    parser.add_argument(
        "--minimum-radial-normal-alignment", type=float, default=0.30
    )
    parser.add_argument("--minimum-height-fraction", type=float, default=0.18)
    parser.add_argument("--maximum-height-fraction", type=float, default=0.72)
    parser.add_argument("--minimum-radial-fraction", type=float, default=0.25)
    parser.add_argument(
        "--minimum-pair-separation-fraction", type=float, default=0.65
    )
    parser.add_argument("--minimum-vertical-separation-mm", type=float, default=0.0)
    parser.add_argument("--maximum-vertical-separation-mm", type=float, default=45.0)
    parser.add_argument("--minimum-standoff-mm", type=float, default=35.0)
    parser.add_argument("--maximum-standoff-mm", type=float, default=115.0)
    parser.add_argument("--standoff-step-mm", type=float, default=10.0)
    parser.add_argument("--left-rolls", type=float, nargs="+", default=[0.0, 90.0])
    parser.add_argument("--right-rolls", type=float, nargs="+", default=[0.0, 90.0])
    parser.add_argument("--maximum-source-seeds", type=int, default=5)
    parser.add_argument("--hand-variants", type=int, default=1)
    parser.add_argument("--pair-variants", type=int, default=1)
    parser.add_argument(
        "--stop-after-candidates",
        type=int,
        default=0,
        help=(
            "Stop after this many feasible paired candidates; zero evaluates "
            "every ranked region pair. Useful for smoke tests."
        ),
    )
    parser.add_argument("--preserve-standoff-diversity", action="store_true")
    parser.add_argument("--minimum-pair-clearance-mm", type=float, default=2.0)
    parser.add_argument(
        "--maximum-pair-clearance-mm", type=float, default=float("inf")
    )
    parser.add_argument("--contact-mm", type=float, default=2.0)
    parser.add_argument("--penetration-mm", type=float, default=2.0)
    parser.add_argument("--minimum-contact-links", type=int, default=3)
    parser.add_argument("--minimum-contact-points", type=int, default=4)
    parser.add_argument("--target-contact-links", type=int, default=3)
    parser.add_argument("--target-contact-points", type=int, default=12)
    parser.add_argument("--minimum-contact-digits", type=int, default=1)
    parser.add_argument("--target-contact-digits", type=int, default=2)
    parser.add_argument("--target-contact-spread-mm", type=float, default=18.0)
    parser.add_argument("--target-pair-clearance-mm", type=float, default=22.0)
    parser.add_argument(
        "--maximum-horizontal-root-cosine",
        type=float,
        default=1.0,
        help=(
            "V3-only final-protocol guard on the horizontal wrist-root "
            "cosine. Use -0.8 for genuinely opposed side grasps."
        ),
    )
    parser.add_argument(
        "--maximum-root-z-mm",
        type=float,
        default=float("inf"),
        help="V3-only maximum absolute wrist-root z coordinate.",
    )
    parser.add_argument(
        "--maximum-root-height-difference-mm",
        type=float,
        default=float("inf"),
        help="V3-only maximum left/right wrist-root height difference.",
    )
    parser.add_argument("--support-clearance-mm", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--left-robot-name", default="allegro_left")
    parser.add_argument("--right-robot-name", default="allegro_right")
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    if args.enhanced_bidex_v2 and args.enhanced_bidex_v3:
        parser.error("Choose only one of --enhanced-bidex-v2/--enhanced-bidex-v3")
    if args.enhanced_bidex_v3 and not 10 <= args.v3_parent_candidate_cap <= 20:
        parser.error("V3 parent candidate cap must be within the requested 10--20 range")
    if not 0.0 < args.v3_quality_keep_fraction <= 1.0:
        parser.error("--v3-quality-keep-fraction must be in (0, 1]")

    started = time.monotonic()
    entries = torch.load(args.source_vis, map_location="cpu", weights_only=False)
    entry = next(row for row in entries if row["object_name"] == args.object)
    dataset_name, mesh_name = args.object.split("+", 1)
    mesh_path = (
        ROOT
        / "data/data_urdf/object"
        / dataset_name
        / mesh_name
        / f"{mesh_name}.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    extents_mm = np.asarray(mesh.extents) * 1000.0
    if max(extents_mm[:2]) < args.minimum_object_horizontal_span_mm:
        raise RuntimeError(
            f"Object horizontal span {max(extents_mm[:2]):.2f} mm is below "
            f"the required {args.minimum_object_horizontal_span_mm:.2f} mm"
        )
    if extents_mm[2] < args.minimum_object_height_mm:
        raise RuntimeError(
            f"Object height {extents_mm[2]:.2f} mm is below the required "
            f"{args.minimum_object_height_mm:.2f} mm"
        )
    query = trimesh.proximity.ProximityQuery(mesh)
    left_hand = create_hand_model(args.left_robot_name, torch.device("cpu"))
    right_hand = create_hand_model(args.right_robot_name, torch.device("cpu"))
    all_seeds = entry["predict_q"].detach().cpu()
    source_rng = np.random.default_rng(args.seed)
    source_indices = source_rng.choice(
        len(all_seeds),
        size=min(args.maximum_source_seeds, len(all_seeds)),
        replace=False,
    )
    seeds = all_seeds[torch.as_tensor(source_indices, dtype=torch.long)]

    regions, region_radius = build_regions(mesh, args)
    pairs = rank_region_pairs(mesh, regions, args)
    if not pairs:
        raise RuntimeError("No feasible GWS region pairs")
    print(
        f"regions={len(regions)} selected_pairs={len(pairs)} "
        f"region_radius_mm={region_radius * 1000:.2f}",
        flush=True,
    )

    left_candidates = []
    right_candidates = []
    metadata = []
    support_z_mm = float(mesh.bounds[0, 2] * 1000.0)
    # Region pairs heavily reuse the same surface regions.  The decoupled
    # single-hand optimization is deterministic for a given hand and region,
    # so cache it instead of repeating dozens of FK/proximity evaluations.
    # The cache is side-specific because the left/right Allegro assets are
    # genuinely mirrored rather than aliases of one URDF.
    variant_cache = {}

    def cached_variants(side, region_index):
        key = (side, region_index)
        if key not in variant_cache:
            hand = left_hand if side == "left" else right_hand
            variant_cache[key] = optimize_hand(
                hand,
                seeds,
                source_indices,
                regions[region_index],
                mesh,
                query,
                side,
                support_z_mm,
                args,
            )
        return variant_cache[key]

    for pair_order, pair in enumerate(pairs):
        left_region = regions[pair["left_region_index"]]
        right_region = regions[pair["right_region_index"]]
        left_variants = cached_variants("left", pair["left_region_index"])
        right_variants = cached_variants("right", pair["right_region_index"])
        if not left_variants or not right_variants:
            print(
                f"pair={pair_order} rejected_by_reachability_filter "
                f"left={len(left_variants)} right={len(right_variants)}",
                flush=True,
            )
            continue
        combinations = []
        for left in left_variants:
            for right in right_variants:
                left_xy = left[1][:2]
                right_xy = right[1][:2]
                horizontal_root_cosine = float(
                    torch.dot(left_xy, right_xy)
                    / (
                        left_xy.norm().clamp_min(1e-8)
                        * right_xy.norm().clamp_min(1e-8)
                    )
                )
                maximum_root_z_mm = float(
                    torch.maximum(left[1][2].abs(), right[1][2].abs())
                    * 1000.0
                )
                root_height_difference_mm = float(
                    (left[1][2] - right[1][2]).abs() * 1000.0
                )
                if args.enhanced_bidex_v3 and not (
                    horizontal_root_cosine
                    <= args.maximum_horizontal_root_cosine
                    and maximum_root_z_mm <= args.maximum_root_z_mm
                    and root_height_difference_mm
                    <= args.maximum_root_height_difference_mm
                ):
                    continue
                outer_clearance_mm, inner_clearance_mm = hand_clearances(
                    left_hand,
                    left[1].unsqueeze(0),
                    right[1].unsqueeze(0),
                    right_hand=right_hand,
                )[0]
                minimum_clearance_mm = min(
                    outer_clearance_mm, inner_clearance_mm
                )
                if args.enhanced_bidex_v3 and not (
                    args.minimum_pair_clearance_mm
                    <= minimum_clearance_mm
                    <= args.maximum_pair_clearance_mm
                ):
                    continue
                pair_objective = (
                    left[0]
                    + right[0]
                    + 20.0
                    * max(
                        0.0,
                        args.minimum_pair_clearance_mm
                        - minimum_clearance_mm,
                    )
                    ** 2
                    + 0.5
                    * max(
                        0.0,
                        minimum_clearance_mm
                        - args.maximum_pair_clearance_mm,
                    )
                    ** 2
                )
                if enhanced_bidex(args):
                    link_balance = abs(
                        left[6]["contact_link_count"]
                        - right[6]["contact_link_count"]
                    )
                    pair_objective += (
                        0.08
                        * (
                            minimum_clearance_mm
                            - args.target_pair_clearance_mm
                        )
                        ** 2
                        + 8.0 * link_balance
                        - 35.0 * pair["gws_score"]
                        - 20.0 * pair["gws_isotropy"]
                    )
                combinations.append(
                    (
                        pair_objective,
                        left,
                        right,
                        outer_clearance_mm,
                        inner_clearance_mm,
                        horizontal_root_cosine,
                        maximum_root_z_mm,
                        root_height_difference_mm,
                    )
                )
        if not combinations:
            print(
                f"pair={pair_order} rejected_by_pair_clearance_filter",
                flush=True,
            )
            continue
        parent_cap = (
            args.v3_parent_candidate_cap
            if args.enhanced_bidex_v3
            else args.pair_variants
        )
        selected_combinations = sorted(combinations, key=lambda row: row[0])[
            :parent_cap
        ]
        for local_variant_rank, (
            pair_objective,
            left,
            right,
            outer_clearance_mm,
            inner_clearance_mm,
            horizontal_root_cosine,
            maximum_root_z_mm,
            root_height_difference_mm,
        ) in enumerate(selected_combinations):
            left_candidates.append(left[1])
            right_candidates.append(right[1])
            metadata.append(
                {
                    "source_pair_index": pair_order,
                    "region_pair_order": pair_order,
                    "local_variant_rank": local_variant_rank,
                    "left_index": left[2],
                    "right_index": right[2],
                    "left_opposition_roll_degrees": left[3],
                    "right_opposition_roll_degrees": right[3],
                    "opposition_roll_degrees": right[3],
                    "candidate_method": method_name(args),
                    "gws_score": pair["gws_score"],
                    "gws_isotropy": pair["gws_isotropy"],
                    "gws_sigma_min": pair["gws_sigma_min"],
                    "ik_reachability_filter_pass": True,
                    "region_separation_mm": pair["separation_m"] * 1000.0,
                    "region_vertical_separation_mm": pair[
                        "vertical_separation_m"
                    ]
                    * 1000.0,
                    "region_opposition_cosine": pair[
                        "horizontal_opposition_cosine"
                    ],
                    "left_local_objective": left[0],
                    "right_local_objective": right[0],
                    "left_local_standoff_mm": left[4],
                    "right_local_standoff_mm": right[4],
                    "left_local_contact_links": left[6][
                        "contact_link_count"
                    ],
                    "right_local_contact_links": right[6][
                        "contact_link_count"
                    ],
                    "left_local_contact_digits": left[6]["contact_digit_count"],
                    "right_local_contact_digits": right[6]["contact_digit_count"],
                    "left_local_contact_spread_mm": left[6]["contact_spread_mm"],
                    "right_local_contact_spread_mm": right[6]["contact_spread_mm"],
                    "left_local_wrench_residual": left[6][
                        "mean_wrench_residual"
                    ],
                    "right_local_wrench_residual": right[6][
                        "mean_wrench_residual"
                    ],
                    "local_outer_hand_clearance_mm": outer_clearance_mm,
                    "local_inner_hand_clearance_mm": inner_clearance_mm,
                    "local_horizontal_root_cosine": horizontal_root_cosine,
                    "local_maximum_root_z_mm": maximum_root_z_mm,
                    "local_root_height_difference_mm": root_height_difference_mm,
                    "pair_local_objective": pair_objective,
                }
            )
        best = selected_combinations[0]
        print(
            f"pair={pair_order} gws={pair['gws_score']:.4f} "
            f"left_obj={best[1][0]:.3f} right_obj={best[2][0]:.3f} "
            f"left_links={best[1][6]['contact_link_count']} "
            f"right_links={best[2][6]['contact_link_count']} "
            f"inner_clearance_mm={best[4]:.2f} "
            f"variants={len(selected_combinations)}",
            flush=True,
        )
        if (
            args.stop_after_candidates > 0
            and len(metadata) >= args.stop_after_candidates
        ):
            print(
                f"early_stop_after_candidates={len(metadata)}",
                flush=True,
            )
            break

    if not left_candidates:
        raise RuntimeError(
            "No region pair passed the hand-workspace IK/reachability filter"
        )
    selected_indices = quality_constrained_fps(
        left_candidates, right_candidates, metadata, args
    )
    left_candidates = [left_candidates[index] for index in selected_indices]
    right_candidates = [right_candidates[index] for index in selected_indices]
    metadata = [metadata[index] for index in selected_indices]
    print(
        f"quality_fps_selected={len(metadata)} "
        f"parents={len({row['region_pair_order'] for row in metadata})}",
        flush=True,
    )
    derived = dict(entry)
    derived["precomputed_bimanual_candidates"] = {
        "left_q": torch.stack(left_candidates),
        "right_q": torch.stack(right_candidates),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.audit_json.exists():
        raise RuntimeError("Refusing to overwrite candidate or audit output")
    torch.save([derived], args.output)
    audit = {
        "method": method_name(args),
        "official_code_available": False,
        "paper": "arXiv:2604.06589v1",
        "object_name": args.object,
        "mesh_path": str(mesh_path),
        "mesh_faces": int(len(mesh.faces)),
        "object_extents_mm": extents_mm.tolist(),
        "surface_points": args.surface_points,
        "anchors_requested": args.anchors,
        "regions_retained": len(regions),
        "region_points": args.region_points,
        "source_indices": [int(index) for index in source_indices],
        "pair_variants": args.pair_variants,
        "stop_after_candidates": args.stop_after_candidates,
        "gws_contacts_per_region": args.gws_contacts,
        "region_radius_mm": region_radius * 1000.0,
        "enhanced_bidex_v2": args.enhanced_bidex_v2,
        "enhanced_bidex_v3": args.enhanced_bidex_v3,
        "v3_parent_candidate_cap": args.v3_parent_candidate_cap,
        "v3_finger_perturbations": args.v3_finger_perturbations,
        "v3_quality_keep_fraction": args.v3_quality_keep_fraction,
        "v3_final_candidates": args.v3_final_candidates,
        "maximum_horizontal_root_cosine": args.maximum_horizontal_root_cosine,
        "maximum_root_z_mm": args.maximum_root_z_mm,
        "maximum_root_height_difference_mm": (
            args.maximum_root_height_difference_mm
        ),
        "selected_parent_count": len(
            {row["region_pair_order"] for row in metadata}
        ),
        "ik_filter_scope": (
            "free_root_Allegro_hand_workspace_proxy; no arm model in TRO-Grasp"
        ),
        "pairs": [
            {
                **pair,
                "left_region": serializable_region(
                    regions[pair["left_region_index"]]
                ),
                "right_region": serializable_region(
                    regions[pair["right_region_index"]]
                ),
                "local_optimizations": [
                    row
                    for row in metadata
                    if row["region_pair_order"] == index
                ],
            }
            for index, pair in enumerate(pairs)
        ],
        "unique_side_region_optimizations": len(variant_cache),
        "uncached_side_region_optimizations": 2 * len(pairs),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(args.output.resolve()),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"saved={args.output.resolve()}", flush=True)
    print(f"audit={args.audit_json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
