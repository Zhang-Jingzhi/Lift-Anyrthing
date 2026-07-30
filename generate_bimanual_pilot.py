"""Generate an auditable pilot dataset for two identical Allegro-hand actors.

The generator uses single-hand poses only as geometric initializations.  The
second pose is moved to the opposite object side and searched over wrist-roll
angles.  Candidates are filtered by joint limits, exact mesh penetration,
contact proximity, inter-hand clearance, and a six-direction Isaac rollout.

This is intentionally labelled a pilot dataset: the repository provides an
Allegro left-hand URDF but no matching Allegro right-hand URDF.
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from run_bimanual_opposed_search import oppose_pose
from run_bimanual_pairing_baseline import hand_clearances, run_isaac
from utils.controller import controller
from utils.hand_model import create_hand_model


DIRECTION_NAMES = ("+X", "+Y", "+Z", "-X", "-Y", "-Z")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_source_pairs(q_batch, count, seed):
    """Select a mixture of diverse and random unique source-pose pairs."""
    pairs = []
    for left in range(len(q_batch)):
        for right in range(left + 1, len(q_batch)):
            internal_distance = float(
                torch.norm(q_batch[left, 6:] - q_batch[right, 6:])
            )
            left_direction = q_batch[left, :3]
            right_direction = q_batch[right, :3]
            cosine = float(
                torch.dot(left_direction, right_direction)
                / (
                    left_direction.norm().clamp_min(1e-8)
                    * right_direction.norm().clamp_min(1e-8)
                )
            )
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "internal_distance": internal_distance,
                    "source_root_cosine": cosine,
                }
            )
    count = min(count, len(pairs))
    diverse_count = count // 2
    diverse = sorted(
        pairs,
        key=lambda item: item["internal_distance"],
        reverse=True,
    )[:diverse_count]
    diverse_keys = {
        (item["left_index"], item["right_index"]) for item in diverse
    }
    remaining = [
        item
        for item in pairs
        if (item["left_index"], item["right_index"]) not in diverse_keys
    ]
    random.Random(seed).shuffle(remaining)
    return diverse + remaining[: count - diverse_count]


def transformed_points(hand, q):
    links, _ = hand.get_transformed_links_pc(q)
    return torch.cat(list(links.values()), dim=0).detach().cpu().numpy()


def exact_pose_metrics(hand, mesh, query, q, contact_threshold):
    points = transformed_points(hand, q)
    _, distances, _ = query.on_surface(points)
    inside = mesh.contains(points)
    penetration = float(distances[inside].max()) if inside.any() else 0.0
    return {
        "penetration_depth_mm": penetration * 1000.0,
        "min_surface_distance_mm": float(distances.min()) * 1000.0,
        "contact_point_count": int(
            np.count_nonzero(distances <= contact_threshold)
        ),
    }


def transformed_points_batch(hand, q_batch):
    hand.update_status(q_batch)
    transforms = torch.stack(
        [
            hand.frame_status[link_name].get_matrix()
            for link_name in hand.links_pc
        ],
        dim=1,
    )
    return hand.get_transformed_links_pc_from_se3(transforms)


def realized_pose_metrics_batch(
    hand,
    mesh,
    query,
    left_q,
    right_q,
    contact_threshold,
    batch_size,
):
    results = []
    for start in range(0, len(left_q), batch_size):
        end = min(start + batch_size, len(left_q))
        count = end - start
        combined_q = torch.cat(
            [left_q[start:end], right_q[start:end]],
            dim=0,
        )
        points = transformed_points_batch(hand, combined_q)
        flat_points = points.detach().cpu().numpy().reshape(-1, 3)
        _, distances, _ = query.on_surface(flat_points)
        inside = mesh.contains(flat_points)
        distances = distances.reshape(2 * count, -1)
        inside = inside.reshape(2 * count, -1)
        penetration = np.where(inside, distances, 0.0).max(axis=1)
        minimum = distances.min(axis=1)
        contact_count = np.count_nonzero(
            distances <= contact_threshold,
            axis=1,
        )
        clearance = (
            torch.cdist(points[:count], points[count:])
            .amin(dim=(1, 2))
            .detach()
            .cpu()
            .numpy()
        )
        for offset in range(count):
            results.append(
                {
                    "left": {
                        "penetration_depth_mm": float(
                            penetration[offset] * 1000
                        ),
                        "min_surface_distance_mm": float(
                            minimum[offset] * 1000
                        ),
                        "contact_point_count": int(
                            contact_count[offset]
                        ),
                    },
                    "right": {
                        "penetration_depth_mm": float(
                            penetration[count + offset] * 1000
                        ),
                        "min_surface_distance_mm": float(
                            minimum[count + offset] * 1000
                        ),
                        "contact_point_count": int(
                            contact_count[count + offset]
                        ),
                    },
                    "clearance_mm": float(clearance[offset] * 1000),
                }
            )
    return results


def chunked_realized_geometry(
    args,
    object_name,
    left_q,
    right_q,
    object_dir,
):
    results = []
    for start in range(0, len(left_q), args.realized_batch_size):
        end = min(start + args.realized_batch_size, len(left_q))
        chunk_dir = (
            object_dir
            / "realized_geometry_chunks"
            / f"{start:05d}_{end:05d}"
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)
        left_path = chunk_dir / "left_q.pt"
        right_path = chunk_dir / "right_q.pt"
        output_path = chunk_dir / "realized_geometry_result.pt"
        torch.save(left_q[start:end], left_path)
        torch.save(right_q[start:end], right_path)
        if not output_path.is_file() or args.force:
            command = [
                sys.executable,
                str(
                    args.repo
                    / "validation/bimanual_realized_geometry_main.py"
                ),
                "--repo",
                str(args.repo),
                "--object-name",
                object_name,
                "--left-q-file",
                str(left_path),
                "--right-q-file",
                str(right_path),
                "--output-file",
                str(output_path),
                "--contact-mm",
                str(args.contact_mm),
            ]
            with (chunk_dir / "geometry.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=args.repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        results.extend(
            torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )
        )
    return results


def pose_pair_metrics(
    hand,
    mesh,
    query,
    q,
    contact_threshold,
):
    outer, inner = controller("allegro", q.unsqueeze(0))
    return {
        "outer": exact_pose_metrics(
            hand,
            mesh,
            query,
            outer[0],
            contact_threshold,
        ),
        "inner": exact_pose_metrics(
            hand,
            mesh,
            query,
            inner[0],
            contact_threshold,
        ),
    }


def joint_limit_pass(hand, q):
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=q.dtype)
    upper = torch.as_tensor(upper, dtype=q.dtype)
    tolerance = 1e-5
    return bool(
        torch.all(q >= lower - tolerance)
        and torch.all(q <= upper + tolerance)
    )


def clamp_to_joint_limits(hand, q_batch):
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=q_batch.dtype)
    upper = torch.as_tensor(upper, dtype=q_batch.dtype)
    return torch.maximum(torch.minimum(q_batch, upper), lower)


def command_final_metrics(command_q, final_q):
    command_rotation = Rotation.from_euler(
        "XYZ",
        command_q[3:6].detach().cpu().numpy(),
    )
    final_rotation = Rotation.from_euler(
        "XYZ",
        final_q[3:6].detach().cpu().numpy(),
    )
    return {
        "translation_mm": float(
            torch.norm(final_q[:3] - command_q[:3]) * 1000
        ),
        "rotation_degrees": float(
            (command_rotation.inv() * final_rotation).magnitude()
            * 180.0
            / np.pi
        ),
        "joint_l2": float(
            torch.norm(final_q[6:] - command_q[6:])
        ),
    }


def chunked_isaac(
    args,
    object_name,
    left_q,
    right_q,
    object_dir,
):
    parts = []
    for start in range(0, len(left_q), args.isaac_batch_size):
        end = min(start + args.isaac_batch_size, len(left_q))
        chunk_dir = object_dir / "isaac_chunks" / f"{start:05d}_{end:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        run_args = SimpleNamespace(
            repo=args.repo,
            isaac_python=args.isaac_python,
            gpu=args.gpu,
            force=args.force,
        )
        parts.append(
            run_isaac(
                run_args,
                object_name,
                left_q[start:end],
                right_q[start:end],
                chunk_dir,
            )
        )
    keys = set.intersection(*(set(part) for part in parts))
    return {
        key: torch.cat([part[key] for part in parts], dim=0)
        for key in keys
    }


def rejection_reasons(row):
    reasons = []
    for key, label in (
        ("joint_limit_pass", "joint_limit"),
        ("outer_penetration_pass", "outer_penetration"),
        ("inter_hand_clearance_pass", "inter_hand_collision"),
        ("dual_contact_pass", "missing_contact"),
    ):
        if not row[key]:
            reasons.append(label)
    if row["isaac_evaluated"] and not row["isaac_success"]:
        reasons.append("isaac_instability")
    if (
        row["isaac_evaluated"]
        and not row["realized_joint_limit_pass"]
    ):
        reasons.append("realized_joint_limit")
    if (
        row["isaac_evaluated"]
        and not row["realized_object_penetration_pass"]
    ):
        reasons.append("realized_object_penetration")
    if (
        row["isaac_evaluated"]
        and not row["realized_hand_clearance_pass"]
    ):
        reasons.append("realized_hand_collision")
    return reasons


def process_object(args, entry, hand, rolls, object_index):
    object_name = entry["object_name"]
    dataset, name = object_name.split("+")
    mesh_path = (
        args.repo
        / "data/data_urdf/object"
        / dataset
        / name
        / f"{name}.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    query = trimesh.proximity.ProximityQuery(mesh)
    q_batch = clamp_to_joint_limits(
        hand,
        entry["predict_q"].detach().cpu(),
    )
    pairs = select_source_pairs(
        q_batch,
        args.pairs_per_object,
        args.seed + object_index,
    )

    left_candidates = []
    right_candidates = []
    candidate_meta = []
    for pair_index, pair in enumerate(pairs):
        for roll_degrees in rolls:
            left_candidates.append(
                q_batch[pair["left_index"]].clone()
            )
            right_candidates.append(
                oppose_pose(
                    q_batch[pair["right_index"]],
                    float(roll_degrees),
                )
            )
            candidate_meta.append(
                {
                    **pair,
                    "source_pair_index": pair_index,
                    "opposition_roll_degrees": float(roll_degrees),
                }
            )
    left_q = torch.stack(left_candidates)
    right_q = torch.stack(right_candidates)
    left_outer_q, left_target_q = controller("allegro", left_q)
    right_outer_q, right_target_q = controller("allegro", right_q)

    left_cache = {}
    right_cache = {}
    print(
        f"[{object_name}] exact geometric audit for "
        f"{len(left_q)} candidates",
        flush=True,
    )
    for candidate_index, meta in enumerate(
        tqdm(candidate_meta, desc=f"geometry {object_name}")
    ):
        left_key = meta["left_index"]
        right_key = (
            meta["right_index"],
            meta["opposition_roll_degrees"],
        )
        if left_key not in left_cache:
            left_cache[left_key] = pose_pair_metrics(
                hand,
                mesh,
                query,
                left_q[candidate_index],
                args.contact_mm / 1000.0,
            )
        if right_key not in right_cache:
            right_cache[right_key] = pose_pair_metrics(
                hand,
                mesh,
                query,
                right_q[candidate_index],
                args.contact_mm / 1000.0,
            )

    clearances = hand_clearances(hand, left_q, right_q)
    rows = []
    geometry_indices = []
    for index, meta in enumerate(candidate_meta):
        left_metrics = left_cache[meta["left_index"]]
        right_metrics = right_cache[
            (
                meta["right_index"],
                meta["opposition_roll_degrees"],
            )
        ]
        outer_clearance, inner_clearance = clearances[index]
        joint_pass = joint_limit_pass(
            hand,
            left_q[index],
        ) and joint_limit_pass(hand, right_q[index])
        outer_penetration_pass = (
            left_metrics["outer"]["penetration_depth_mm"]
            <= args.penetration_mm
            and right_metrics["outer"]["penetration_depth_mm"]
            <= args.penetration_mm
        )
        inner_penetration_pass = (
            left_metrics["inner"]["penetration_depth_mm"]
            <= args.penetration_mm
            and right_metrics["inner"]["penetration_depth_mm"]
            <= args.penetration_mm
        )
        clearance_pass = (
            outer_clearance > args.clearance_mm
            and inner_clearance > args.clearance_mm
        )
        dual_contact_pass = (
            left_metrics["inner"]["min_surface_distance_mm"]
            <= args.contact_mm
            and right_metrics["inner"]["min_surface_distance_mm"]
            <= args.contact_mm
        )
        geometry_pass = (
            joint_pass
            and outer_penetration_pass
            and clearance_pass
            and dual_contact_pass
        )
        if geometry_pass:
            geometry_indices.append(index)
        rows.append(
            {
                "candidate_index": index,
                "object_name": object_name,
                **meta,
                "joint_limit_pass": joint_pass,
                "left_outer_penetration_mm": left_metrics["outer"][
                    "penetration_depth_mm"
                ],
                "right_outer_penetration_mm": right_metrics["outer"][
                    "penetration_depth_mm"
                ],
                "outer_penetration_pass": outer_penetration_pass,
                "left_inner_penetration_mm": left_metrics["inner"][
                    "penetration_depth_mm"
                ],
                "right_inner_penetration_mm": right_metrics["inner"][
                    "penetration_depth_mm"
                ],
                "inner_penetration_pass": inner_penetration_pass,
                "left_contact_distance_mm": left_metrics["inner"][
                    "min_surface_distance_mm"
                ],
                "right_contact_distance_mm": right_metrics["inner"][
                    "min_surface_distance_mm"
                ],
                "left_contact_point_count": left_metrics["inner"][
                    "contact_point_count"
                ],
                "right_contact_point_count": right_metrics["inner"][
                    "contact_point_count"
                ],
                "dual_contact_pass": dual_contact_pass,
                "outer_hand_clearance_mm": outer_clearance,
                "inner_hand_clearance_mm": inner_clearance,
                "inter_hand_clearance_pass": clearance_pass,
                "geometry_pass": geometry_pass,
                "isaac_evaluated": False,
                "isaac_success": False,
                "settle_displacement_mm": "",
                "final_displacement_mm": "",
                "max_direction_displacement_mm": "",
                "left_command_to_final_l2": "",
                "right_command_to_final_l2": "",
                "left_command_to_final_translation_mm": "",
                "right_command_to_final_translation_mm": "",
                "left_command_to_final_rotation_degrees": "",
                "right_command_to_final_rotation_degrees": "",
                "left_command_to_final_joint_l2": "",
                "right_command_to_final_joint_l2": "",
                "realized_joint_limit_pass": False,
                "left_realized_penetration_mm": "",
                "right_realized_penetration_mm": "",
                "left_realized_surface_distance_mm": "",
                "right_realized_surface_distance_mm": "",
                "realized_hand_clearance_mm": "",
                "realized_object_penetration_pass": False,
                "realized_hand_clearance_pass": False,
                "realized_pose_pass": False,
                **{
                    f"displacement_{direction}_mm": ""
                    for direction in DIRECTION_NAMES
                },
                "strict_success": False,
                "rejection_reasons": "",
            }
        )

    print(
        f"[{object_name}] geometry pass "
        f"{len(geometry_indices)}/{len(rows)}",
        flush=True,
    )
    realized_q = {}
    if geometry_indices:
        geometry_indices_tensor = torch.as_tensor(geometry_indices)
        object_dir = args.output_dir / object_name.replace("+", "__")
        object_dir.mkdir(parents=True, exist_ok=True)
        isaac = chunked_isaac(
            args,
            object_name,
            left_q[geometry_indices_tensor],
            right_q[geometry_indices_tensor],
            object_dir,
        )
        realized_metrics = chunked_realized_geometry(
            args,
            object_name,
            isaac["left_q_final"],
            isaac["right_q_final"],
            object_dir,
        )
        for isaac_index, candidate_index in enumerate(geometry_indices):
            row = rows[candidate_index]
            row["isaac_evaluated"] = True
            row["isaac_success"] = bool(isaac["success"][isaac_index])
            row["settle_displacement_mm"] = float(
                isaac["settle_displacement"][isaac_index] * 1000
            )
            row["final_displacement_mm"] = float(
                isaac["disturbance_displacement"][isaac_index] * 1000
            )
            row["max_direction_displacement_mm"] = float(
                isaac["max_direction_displacement"][isaac_index] * 1000
            )
            left_q_final = isaac["left_q_final"][isaac_index]
            right_q_final = isaac["right_q_final"][isaac_index]
            left_delta = command_final_metrics(
                left_target_q[candidate_index],
                left_q_final,
            )
            right_delta = command_final_metrics(
                right_target_q[candidate_index],
                right_q_final,
            )
            row["left_command_to_final_l2"] = float(
                torch.norm(
                    left_q_final - left_target_q[candidate_index]
                )
            )
            row["right_command_to_final_l2"] = float(
                torch.norm(
                    right_q_final - right_target_q[candidate_index]
                )
            )
            for side, delta in (
                ("left", left_delta),
                ("right", right_delta),
            ):
                row[
                    f"{side}_command_to_final_translation_mm"
                ] = delta["translation_mm"]
                row[
                    f"{side}_command_to_final_rotation_degrees"
                ] = delta["rotation_degrees"]
                row[
                    f"{side}_command_to_final_joint_l2"
                ] = delta["joint_l2"]

            left_realized = realized_metrics[isaac_index]["left"]
            right_realized = realized_metrics[isaac_index]["right"]
            realized_clearance = realized_metrics[isaac_index][
                "clearance_mm"
            ]
            row["realized_joint_limit_pass"] = (
                joint_limit_pass(hand, left_q_final)
                and joint_limit_pass(hand, right_q_final)
            )
            row["left_realized_penetration_mm"] = left_realized[
                "penetration_depth_mm"
            ]
            row["right_realized_penetration_mm"] = right_realized[
                "penetration_depth_mm"
            ]
            row[
                "left_realized_surface_distance_mm"
            ] = left_realized["min_surface_distance_mm"]
            row[
                "right_realized_surface_distance_mm"
            ] = right_realized["min_surface_distance_mm"]
            row["realized_hand_clearance_mm"] = realized_clearance
            row["realized_object_penetration_pass"] = (
                left_realized["penetration_depth_mm"]
                <= args.penetration_mm
                and right_realized["penetration_depth_mm"]
                <= args.penetration_mm
            )
            row["realized_hand_clearance_pass"] = (
                realized_clearance > args.clearance_mm
            )
            row["realized_pose_pass"] = (
                row["realized_joint_limit_pass"]
                and row["realized_object_penetration_pass"]
                and row["realized_hand_clearance_pass"]
            )
            realized_q[candidate_index] = (
                left_q_final,
                right_q_final,
            )
            direction_values = isaac["direction_displacements"][
                isaac_index
            ]
            for direction_index, direction in enumerate(DIRECTION_NAMES):
                row[f"displacement_{direction}_mm"] = float(
                    direction_values[direction_index] * 1000
                )
            row["strict_success"] = (
                row["isaac_success"] and row["realized_pose_pass"]
            )

    retained = []
    for index, row in enumerate(rows):
        reasons = rejection_reasons(row)
        row["rejection_reasons"] = ";".join(reasons)
        if row["strict_success"]:
            left_q_final, right_q_final = realized_q[index]
            retained.append(
                {
                    "object_name": object_name,
                    "left_q": left_q_final,
                    "right_q": right_q_final,
                    "left_q_seed": left_q[index],
                    "right_q_seed": right_q[index],
                    "left_q_outer": left_outer_q[index],
                    "right_q_outer": right_outer_q[index],
                    "left_q_command": left_target_q[index],
                    "right_q_command": right_target_q[index],
                    "candidate_index": index,
                    "source_pair_index": row["source_pair_index"],
                    "left_source_index": row["left_index"],
                    "right_source_index": row["right_index"],
                    "opposition_roll_degrees": row[
                        "opposition_roll_degrees"
                    ],
                    "metrics": {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "object_name",
                            "rejection_reasons",
                        }
                    },
                }
            )
    return rows, retained


def object_summary(object_name, rows):
    def rate(key):
        return float(np.mean([bool(row[key]) for row in rows]))

    reasons = Counter()
    for row in rows:
        reasons.update(
            reason
            for reason in row["rejection_reasons"].split(";")
            if reason
        )
    evaluated = [row for row in rows if row["isaac_evaluated"]]
    return {
        "object_name": object_name,
        "num_candidates": len(rows),
        "joint_limit_pass_rate": rate("joint_limit_pass"),
        "outer_penetration_pass_rate": rate(
            "outer_penetration_pass"
        ),
        "inner_penetration_pass_rate": rate(
            "inner_penetration_pass"
        ),
        "inter_hand_clearance_pass_rate": rate(
            "inter_hand_clearance_pass"
        ),
        "dual_contact_pass_rate": rate("dual_contact_pass"),
        "geometry_pass_rate": rate("geometry_pass"),
        "isaac_success_rate": rate("isaac_success"),
        "realized_pose_pass_rate_evaluated": float(
            np.mean(
                [bool(row["realized_pose_pass"]) for row in evaluated]
            )
        )
        if evaluated
        else 0.0,
        "strict_success_rate": rate("strict_success"),
        "num_retained": sum(row["strict_success"] for row in rows),
        "top_rejection_reason": (
            reasons.most_common(1)[0][0] if reasons else ""
        ),
    }


def write_report(path, summaries, total_retained, config):
    lines = [
        "# Bimanual pilot dataset generation report",
        "",
        "> Scope: two identical Allegro left-hand actors. This is not a",
        "> physical left/right Allegro dataset because the repository has no",
        "> Allegro right-hand URDF.",
        "",
        "## Protocol",
        "",
        f"- Source pairs per object: {config['pairs_per_object']}.",
        f"- Opposed wrist-roll choices: {config['roll_count']}.",
        f"- Exact penetration threshold: {config['penetration_mm']} mm.",
        f"- Sampled inter-hand clearance: {config['clearance_mm']} mm.",
        f"- Contact-distance threshold: {config['contact_mm']} mm.",
        f"- Isaac batch size: {config['isaac_batch_size']}.",
        "- Isolated realized-pose mesh-audit subprocess batch size: "
        f"{config['realized_batch_size']}.",
        "- Isaac gravity: disabled, matching the legacy TRO-Grasp protocol.",
        "- Exact penetration of the controller target is recorded as a",
        "  diagnostic but is not a pre-simulation rejection criterion;",
        "  contact dynamics stop the fingers before that unconstrained",
        "  target is reached.",
        "- Physics success requires settle displacement <= 50 mm and the",
        "  maximum displacement observed after any of the six directions",
        "  (+X,+Y,+Z,-X,-Y,-Z) <= 20 mm.",
        "- Retained training poses are the realized left/right joint states",
        "  after Isaac contact dynamics in the settled object frame.",
        "- Before saving, the realized poses are checked again for joint",
        f"  limits, <= {config['penetration_mm']} mm hand-object penetration,",
        f"  and > {config['clearance_mm']} mm inter-hand clearance.",
        "  Controller seeds, open poses, and target poses are",
        "  retained separately for auditability.",
        "",
        "## Funnel by object",
        "",
        "| Object | Candidates | Geometry | Isaac | Realized pose | "
        "Strict | Retained | "
        "Top rejection |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['object_name']}` | {row['num_candidates']} | "
            f"{row['geometry_pass_rate']:.1%} | "
            f"{row['isaac_success_rate']:.1%} | "
            f"{row['realized_pose_pass_rate_evaluated']:.1%} | "
            f"{row['strict_success_rate']:.1%} | "
            f"{row['num_retained']} | "
            f"{row['top_rejection_reason']} |"
        )
    total_candidates = sum(row["num_candidates"] for row in summaries)
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Candidates: {total_candidates}.",
            f"- Strictly retained: {total_retained}.",
            f"- Overall strict retention: "
            f"{total_retained / max(total_candidates, 1):.1%}.",
            "",
            "## Limitations before formal use",
            "",
            "- Add or construct a verified Allegro right-hand URDF if the",
            "  target hardware is a true left/right pair.",
            "- Add gravity-on robustness checks.",
            "- Add single-hand removal ablations to quantify whether both",
            "  hands are necessary.",
            "- Manually inspect representative and worst-case samples.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--source-vis",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/single_hand_baseline/"
            "allegro_unconditioned/vis.pt"
        ),
    )
    parser.add_argument(
        "--object-split",
        type=Path,
        default=Path(
            "graph_exp/bimanual_large_object/object_selection/"
            "bimanual_object_split.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/pilot_v4_realized_strict"
        ),
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        required=True,
        help="Python executable from the Isaac Gym environment.",
    )
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--pairs-per-object", type=int, default=80)
    parser.add_argument("--roll-count", type=int, default=8)
    parser.add_argument("--isaac-batch-size", type=int, default=24)
    parser.add_argument("--realized-batch-size", type=int, default=24)
    parser.add_argument("--penetration-mm", type=float, default=5.0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument("--contact-mm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for key in ("source_vis", "object_split", "output_dir"):
        value = getattr(args, key)
        if not value.is_absolute():
            setattr(args, key, (args.repo / value).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.object_split.read_text())
    object_names = args.objects or split["development"]
    entries = torch.load(args.source_vis, map_location="cpu")
    lookup = {entry["object_name"]: entry for entry in entries}
    missing = sorted(set(object_names) - set(lookup))
    if missing:
        raise KeyError(f"Objects missing from source vis: {missing}")

    rolls = np.linspace(
        0.0,
        180.0,
        args.roll_count,
        endpoint=False,
    )
    hand = create_hand_model("allegro", torch.device("cpu"))
    all_rows = []
    all_retained = []
    summaries = []

    config = {
        "dataset_version": "bimanual_pilot_v4_realized_strict",
        "robot_setup": "two_identical_allegro_left_actors",
        "train_objects": object_names,
        "held_out_objects": split["held_out_test"],
        "pairs_per_object": args.pairs_per_object,
        "roll_count": args.roll_count,
        "isaac_batch_size": args.isaac_batch_size,
        "realized_batch_size": args.realized_batch_size,
        "penetration_mm": args.penetration_mm,
        "clearance_mm": args.clearance_mm,
        "contact_mm": args.contact_mm,
        "seed": args.seed,
        "gravity_enabled": False,
        "disturbance_directions": list(DIRECTION_NAMES),
        "settle_threshold_mm": 50.0,
        "disturbance_threshold_mm": 20.0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    for object_index, object_name in enumerate(object_names):
        checkpoint_path = (
            args.output_dir
            / "object_checkpoints"
            / f"{object_name.replace('+', '__')}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and checkpoint_path.is_file():
            saved = torch.load(checkpoint_path, map_location="cpu")
            rows = saved["rows"]
            retained = saved["retained"]
            print(f"[resume] {object_name}", flush=True)
        else:
            rows, retained = process_object(
                args,
                lookup[object_name],
                hand,
                rolls,
                object_index,
            )
            torch.save(
                {"rows": rows, "retained": retained},
                checkpoint_path,
            )
        all_rows.extend(rows)
        all_retained.extend(retained)
        summaries.append(object_summary(object_name, rows))
        write_csv(args.output_dir / "sample_results.partial.csv", all_rows)
        torch.save(
            all_retained,
            args.output_dir / "filtered_pairs.partial.pt",
        )

    write_csv(args.output_dir / "sample_results.csv", all_rows)
    write_csv(args.output_dir / "object_summary.csv", summaries)
    torch.save(all_retained, args.output_dir / "filtered_pairs.pt")
    torch.save(
        {
            "version": config["dataset_version"],
            "manifest": config,
            "samples": all_retained,
        },
        args.output_dir / "bimanual_dataset.pt",
    )
    write_report(
        args.output_dir / "REPORT.md",
        summaries,
        len(all_retained),
        config,
    )
    print(
        f"Completed: retained {len(all_retained)}/"
        f"{len(all_rows)} candidates",
        flush=True,
    )


if __name__ == "__main__":
    main()
