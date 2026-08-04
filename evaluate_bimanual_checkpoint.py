#!/usr/bin/env python3
"""Generate two-hand grasps and apply geometry-first strict evaluation."""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

from generate_bimanual_pilot import (
    DIRECTION_NAMES,
    clamp_to_joint_limits,
    joint_limit_pass,
    realized_pose_metrics_batch,
)
from model.tro_graph import RobotGraph
from run_bimanual_pairing_baseline import run_isaac
from run_bimanual_pairing_baseline import hand_clearances
from utils.controller import controller
from utils.hand_model import create_hand_model
from utils.optimization import process_transform
from utils.pyroki_ik import PyrokiRetarget


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_links(links):
    return {
        f"{side}/{name}": points
        for side in ("left", "right")
        for name, points in links.items()
    }


def make_batch(repo, object_name, count, num_points, hand):
    dataset, name = object_name.split("+")
    mesh_path = (
        repo / "data/data_urdf/object" / dataset / name / f"{name}.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    points = torch.as_tensor(
        mesh.sample(count * num_points).reshape(count, num_points, 3),
        dtype=torch.float32,
    )
    links = duplicate_links(hand.links_pc)
    return mesh, {
        "robot_name": "allegro_bimanual",
        "object_name": object_name,
        "object_pc": points,
        "robot_links_pc": [links for _ in range(count)],
    }


def retarget_pair(hand, clean_pose, count):
    target_links = list(hand.links_pc)
    sides = []
    for side in ("left", "right"):
        side_pose = {
            key.split("/", 1)[1]: value
            for key, value in clean_pose.items()
            if key.startswith(f"{side}/")
        }
        positions = process_transform(hand.pk_chain, side_pose)
        sides.append(torch.stack([positions[name] for name in target_links], 1))

    initial = torch.stack(
        [hand.get_initial_q() for _ in range(2 * count)]
    )
    targets = torch.cat(sides, dim=0)
    metadata = json.loads(
        Path("data/data_urdf/robot/urdf_assets_meta.json").read_text()
    )
    solver = PyrokiRetarget(metadata["urdf_path"]["allegro"], target_links)
    solve = jax.jit(solver.solve_retarget)
    predicted = solve(
        initial_q=jnp.asarray(initial.numpy()),
        target_pos=jnp.asarray(targets.detach().cpu().numpy()),
    )
    jax.block_until_ready(predicted)
    predicted = torch.from_numpy(np.asarray(predicted)).float()
    return predicted[:count], predicted[count:]


def geometry_gate(
    hand,
    mesh,
    left_q,
    right_q,
    penetration_mm,
    contact_mm,
    clearance_mm,
    left_projection_rad,
    right_projection_rad,
    max_ik_projection_rad,
    geometry_batch_size,
):
    query = trimesh.proximity.ProximityQuery(mesh)
    left_outer, left_target = controller("allegro", left_q)
    right_outer, right_target = controller("allegro", right_q)
    outer_metrics = realized_pose_metrics_batch(
        hand,
        mesh,
        query,
        left_outer,
        right_outer,
        contact_mm / 1000,
        batch_size=min(geometry_batch_size, len(left_q)),
    )
    target_metrics = realized_pose_metrics_batch(
        hand,
        mesh,
        query,
        left_target,
        right_target,
        contact_mm / 1000,
        batch_size=min(geometry_batch_size, len(left_q)),
    )
    rows = []
    for index in range(len(left_q)):
        left_outer_metrics = outer_metrics[index]["left"]
        right_outer_metrics = outer_metrics[index]["right"]
        left_target_metrics = target_metrics[index]["left"]
        right_target_metrics = target_metrics[index]["right"]
        outer_clearance = outer_metrics[index]["clearance_mm"]
        target_clearance = target_metrics[index]["clearance_mm"]
        joint_pass = joint_limit_pass(
            hand, left_q[index]
        ) and joint_limit_pass(hand, right_q[index])
        projection_pass = (
            left_projection_rad[index] <= max_ik_projection_rad
            and right_projection_rad[index] <= max_ik_projection_rad
        )
        outer_penetration_pass = (
            left_outer_metrics["penetration_depth_mm"] <= penetration_mm
            and right_outer_metrics["penetration_depth_mm"]
            <= penetration_mm
        )
        target_penetration_pass = (
            left_target_metrics["penetration_depth_mm"] <= penetration_mm
            and right_target_metrics["penetration_depth_mm"]
            <= penetration_mm
        )
        clearance_pass = (
            outer_clearance > clearance_mm
            and target_clearance > clearance_mm
        )
        dual_contact_pass = (
            left_target_metrics["min_surface_distance_mm"] <= contact_mm
            and right_target_metrics["min_surface_distance_mm"] <= contact_mm
        )
        geometry_pass = (
            joint_pass
            and projection_pass
            and outer_penetration_pass
            and target_penetration_pass
            and clearance_pass
            and dual_contact_pass
        )
        rows.append(
            {
                "sample_index": index,
                "joint_limit_pass": joint_pass,
                "ik_projection_pass": projection_pass,
                "left_ik_projection_rad": left_projection_rad[index],
                "right_ik_projection_rad": right_projection_rad[index],
                "outer_penetration_pass": outer_penetration_pass,
                "target_penetration_pass": target_penetration_pass,
                "dual_contact_pass": dual_contact_pass,
                "inter_hand_clearance_pass": clearance_pass,
                "geometry_pass": geometry_pass,
                "left_target_penetration_mm": left_target_metrics[
                    "penetration_depth_mm"
                ],
                "right_target_penetration_mm": right_target_metrics[
                    "penetration_depth_mm"
                ],
                "left_contact_distance_mm": left_target_metrics[
                    "min_surface_distance_mm"
                ],
                "right_contact_distance_mm": right_target_metrics[
                    "min_surface_distance_mm"
                ],
                "target_hand_clearance_mm": target_clearance,
                "isaac_evaluated": False,
                "isaac_success": False,
                "realized_pose_pass": False,
                "strict_success": False,
            }
        )
    return rows, left_outer, left_target, right_outer, right_target


def refine_radial_geometry(
    hand,
    mesh,
    left_q,
    right_q,
    penetration_mm,
    contact_mm,
    clearance_mm,
    geometry_batch_size,
    shift_min_mm=-30.0,
    shift_max_mm=45.0,
    shift_step_mm=5.0,
):
    """Search small radial base translations with exact mesh metrics."""
    count = len(left_q)
    shifts_mm = torch.arange(
        shift_min_mm,
        shift_max_mm + 0.5 * shift_step_mm,
        shift_step_mm,
        dtype=left_q.dtype,
    )
    shifts = shifts_mm / 1000.0
    num_shifts = len(shifts)
    center = torch.as_tensor(mesh.centroid, dtype=left_q.dtype)

    def candidates(q):
        direction = q[:, :3] - center
        direction = direction / direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        expanded = q[:, None, :].expand(
            -1,
            num_shifts,
            -1,
        ).clone()
        expanded[:, :, :3] += (
            shifts[None, :, None] * direction[:, None, :]
        )
        return expanded.reshape(count * num_shifts, -1)

    left_candidates = candidates(left_q)
    right_candidates = candidates(right_q)
    left_outer, left_target = controller("allegro", left_candidates)
    right_outer, right_target = controller("allegro", right_candidates)
    query = trimesh.proximity.ProximityQuery(mesh)
    outer_metrics = realized_pose_metrics_batch(
        hand,
        mesh,
        query,
        left_outer,
        right_outer,
        contact_mm / 1000.0,
        batch_size=min(geometry_batch_size, count * num_shifts),
    )
    target_metrics = realized_pose_metrics_batch(
        hand,
        mesh,
        query,
        left_target,
        right_target,
        contact_mm / 1000.0,
        batch_size=min(geometry_batch_size, count * num_shifts),
    )

    def score(side, flat_index, shift_index):
        outer = outer_metrics[flat_index][side]
        target = target_metrics[flat_index][side]
        penetration_excess = (
            max(
                outer["penetration_depth_mm"] - penetration_mm,
                0.0,
            )
            + max(
                target["penetration_depth_mm"] - penetration_mm,
                0.0,
            )
        )
        contact_excess = max(
            target["min_surface_distance_mm"] - contact_mm,
            0.0,
        )
        return (
            1000.0 * penetration_excess
            + 100.0 * contact_excess
            + 0.05 * abs(float(shifts_mm[shift_index]))
        )

    selected_left = []
    selected_right = []
    selected_left_shift = []
    selected_right_shift = []
    for sample_index in range(count):
        start = sample_index * num_shifts
        left_ranked = sorted(
            range(num_shifts),
            key=lambda index: score(
                "left",
                start + index,
                index,
            ),
        )
        right_ranked = sorted(
            range(num_shifts),
            key=lambda index: score(
                "right",
                start + index,
                index,
            ),
        )
        best = None
        for left_index in left_ranked[:4]:
            for right_index in right_ranked[:4]:
                candidate_left = left_candidates[
                    start + left_index
                ].unsqueeze(0)
                candidate_right = right_candidates[
                    start + right_index
                ].unsqueeze(0)
                outer_clearance, target_clearance = hand_clearances(
                    hand,
                    candidate_left,
                    candidate_right,
                )[0]
                clearance_excess = (
                    max(clearance_mm - outer_clearance, 0.0)
                    + max(clearance_mm - target_clearance, 0.0)
                )
                combined_score = (
                    score("left", start + left_index, left_index)
                    + score(
                        "right",
                        start + right_index,
                        right_index,
                    )
                    + 10000.0 * clearance_excess
                )
                if best is None or combined_score < best[0]:
                    best = (
                        combined_score,
                        left_index,
                        right_index,
                    )
        _, left_index, right_index = best
        selected_left.append(left_candidates[start + left_index])
        selected_right.append(right_candidates[start + right_index])
        selected_left_shift.append(float(shifts_mm[left_index]))
        selected_right_shift.append(float(shifts_mm[right_index]))
    return (
        torch.stack(selected_left),
        torch.stack(selected_right),
        selected_left_shift,
        selected_right_shift,
    )


def evaluate_object(args, model, hand, object_name):
    object_dir = args.output_dir / object_name.replace("+", "__")
    object_dir.mkdir(parents=True, exist_ok=True)
    mesh, batch = make_batch(
        args.repo,
        object_name,
        args.samples_per_object,
        args.num_points,
        hand,
    )
    batch["object_pc"] = batch["object_pc"].to(args.device)
    with torch.no_grad():
        clean_pose = model.inference(batch)[0]
    left_root_key = next(
        key for key in clean_pose if key.startswith("left/")
    )
    right_root_key = next(
        key for key in clean_pose if key.startswith("right/")
    )
    raw_root_separation_mm = (
        clean_pose[left_root_key][:, :3, 3]
        - clean_pose[right_root_key][:, :3, 3]
    ).norm(dim=1).mul(1000.0).detach().cpu()
    left_q_raw, right_q_raw = retarget_pair(
        hand, clean_pose, args.samples_per_object
    )
    if (
        not torch.isfinite(left_q_raw).all()
        or not torch.isfinite(right_q_raw).all()
    ):
        raise FloatingPointError(f"Non-finite IK output for {object_name}")
    left_q = clamp_to_joint_limits(hand, left_q_raw)
    right_q = clamp_to_joint_limits(hand, right_q_raw)
    left_q_before_refinement = left_q.clone()
    right_q_before_refinement = right_q.clone()
    left_projection = (left_q - left_q_raw).abs().amax(dim=1)
    right_projection = (right_q - right_q_raw).abs().amax(dim=1)
    left_shift_mm = [0.0] * len(left_q)
    right_shift_mm = [0.0] * len(right_q)
    if args.refine_geometry:
        (
            left_q,
            right_q,
            left_shift_mm,
            right_shift_mm,
        ) = refine_radial_geometry(
            hand,
            mesh,
            left_q,
            right_q,
            args.penetration_mm,
            args.contact_mm,
            args.clearance_mm,
            args.geometry_batch_size,
        )

    rows, left_outer, left_target, right_outer, right_target = geometry_gate(
        hand,
        mesh,
        left_q,
        right_q,
        args.penetration_mm,
        args.contact_mm,
        args.clearance_mm,
        left_projection.tolist(),
        right_projection.tolist(),
        args.max_ik_projection_rad,
        args.geometry_batch_size,
    )
    ik_root_separation_mm = (
        left_q[:, :3] - right_q[:, :3]
    ).norm(dim=1).mul(1000.0)
    for index, row in enumerate(rows):
        row["raw_root_separation_mm"] = float(
            raw_root_separation_mm[index]
        )
        row["ik_root_separation_mm"] = float(
            ik_root_separation_mm[index]
        )
        row["left_radial_shift_mm"] = left_shift_mm[index]
        row["right_radial_shift_mm"] = right_shift_mm[index]
        row["object_name"] = object_name
    geometry_indices = [
        index for index, row in enumerate(rows) if row["geometry_pass"]
    ]
    print(
        f"{object_name}: geometry {len(geometry_indices)}/{len(rows)}",
        flush=True,
    )

    torch.save(
        {
            "object_name": object_name,
            "left_q_seed": left_q,
            "right_q_seed": right_q,
            "left_q_before_refinement": left_q_before_refinement,
            "right_q_before_refinement": right_q_before_refinement,
            "left_q_ik_raw": left_q_raw,
            "right_q_ik_raw": right_q_raw,
            "left_q_outer": left_outer,
            "right_q_outer": right_outer,
            "left_q_command": left_target,
            "right_q_command": right_target,
            "geometry_indices": geometry_indices,
        },
        object_dir / "generated_q.pt",
    )
    if not geometry_indices or args.geometry_only:
        return rows

    selected = torch.as_tensor(geometry_indices)
    isaac_args = SimpleNamespace(
        repo=args.repo,
        isaac_python=args.isaac_python,
        gpu=args.gpu,
        force=args.force,
    )
    isaac = run_isaac(
        isaac_args,
        object_name,
        left_q[selected],
        right_q[selected],
        object_dir,
    )
    realized = realized_pose_metrics_batch(
        hand,
        mesh,
        trimesh.proximity.ProximityQuery(mesh),
        isaac["left_q_final"],
        isaac["right_q_final"],
        args.contact_mm / 1000,
        batch_size=min(8, len(geometry_indices)),
    )
    for local_index, sample_index in enumerate(geometry_indices):
        row = rows[sample_index]
        metrics = realized[local_index]
        realized_pass = (
            joint_limit_pass(hand, isaac["left_q_final"][local_index])
            and joint_limit_pass(hand, isaac["right_q_final"][local_index])
            and metrics["left"]["penetration_depth_mm"]
            <= args.penetration_mm
            and metrics["right"]["penetration_depth_mm"]
            <= args.penetration_mm
            and metrics["clearance_mm"] > args.clearance_mm
        )
        row.update(
            {
                "isaac_evaluated": True,
                "isaac_success": bool(isaac["success"][local_index]),
                "settle_displacement_mm": float(
                    isaac["settle_displacement"][local_index] * 1000
                ),
                "max_direction_displacement_mm": float(
                    isaac["max_direction_displacement"][local_index] * 1000
                ),
                "left_realized_penetration_mm": metrics["left"][
                    "penetration_depth_mm"
                ],
                "right_realized_penetration_mm": metrics["right"][
                    "penetration_depth_mm"
                ],
                "realized_hand_clearance_mm": metrics["clearance_mm"],
                "realized_pose_pass": realized_pass,
                "strict_success": bool(isaac["success"][local_index])
                and realized_pass,
            }
        )
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            row[f"displacement_{direction}_mm"] = float(
                isaac["direction_displacements"][
                    local_index, direction_index
                ]
                * 1000
            )
    return rows


def summarize(rows):
    summary = []
    for object_name in sorted({row["object_name"] for row in rows}):
        selected = [
            row for row in rows if row["object_name"] == object_name
        ]
        summary.append(
            {
                "object_name": object_name,
                "samples": len(selected),
                "joint_limit_rate": float(
                    np.mean([row["joint_limit_pass"] for row in selected])
                ),
                "ik_projection_rate": float(
                    np.mean([row["ik_projection_pass"] for row in selected])
                ),
                "penetration_rate": float(
                    np.mean(
                        [
                            row["outer_penetration_pass"]
                            and row["target_penetration_pass"]
                            for row in selected
                        ]
                    )
                ),
                "dual_contact_rate": float(
                    np.mean([row["dual_contact_pass"] for row in selected])
                ),
                "clearance_rate": float(
                    np.mean(
                        [row["inter_hand_clearance_pass"] for row in selected]
                    )
                ),
                "geometry_rate": float(
                    np.mean([row["geometry_pass"] for row in selected])
                ),
                "isaac_rate_all": float(
                    np.mean([row["isaac_success"] for row in selected])
                ),
                "strict_rate_all": float(
                    np.mean([row["strict_success"] for row in selected])
                ),
            }
        )
    return summary


def write_report(path, summary, geometry_only):
    lines = [
        "# Bimanual checkpoint strict evaluation",
        "",
        "Evaluation order: joint limits and exact mesh geometry first; only",
        "geometry-valid candidates are submitted to six-direction Isaac",
        "disturbance rollout.",
        "",
        "| Object | N | Joint | IK projection | Penetration | Dual contact | Clearance | Geometry | Isaac/all | Strict/all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['object_name']}` | {row['samples']} | "
            f"{row['joint_limit_rate']:.1%} | "
            f"{row['ik_projection_rate']:.1%} | "
            f"{row['penetration_rate']:.1%} | "
            f"{row['dual_contact_rate']:.1%} | "
            f"{row['clearance_rate']:.1%} | "
            f"{row['geometry_rate']:.1%} | "
            f"{row['isaac_rate_all']:.1%} | "
            f"{row['strict_rate_all']:.1%} |"
        )
    if geometry_only:
        lines.extend(
            [
                "",
                "This run used `--geometry-only`; Isaac and strict rates are",
                "placeholders and must not be interpreted as rollout results.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-object", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--penetration-mm", type=float, default=5.0)
    parser.add_argument("--contact-mm", type=float, default=5.0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument(
        "--geometry-batch-size",
        type=int,
        default=1,
        help="Bound exact-mesh geometry memory use on complex objects.",
    )
    parser.add_argument(
        "--max-ik-projection-rad",
        type=float,
        default=0.05,
        help="Reject IK solutions requiring a larger joint-limit projection.",
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(os.environ.get("ISAAC_PYTHON", "python")),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--disable-separation-projection",
        action="store_true",
    )
    parser.add_argument("--refine-geometry", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.repo = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = OmegaConf.load(args.config)
    config.model.mode = "test"
    config.model.inference_config = {
        "inference_mode": "unconditioned",
        "bimanual_separation_projection": (
            not args.disable_separation_projection
        ),
        "separation_projection_start_t": 100,
        "root_direction_prior": [0.024, -0.272, 0.889],
    }
    config.model.diffusion_config.ddim_steps = args.ddim_steps
    model = RobotGraph(**config.model).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    hand = create_hand_model("allegro", torch.device("cpu"))

    all_rows = []
    for object_name in args.objects:
        all_rows.extend(evaluate_object(args, model, hand, object_name))
    summary = summarize(all_rows)
    write_csv(args.output_dir / "sample_results.csv", all_rows)
    write_csv(args.output_dir / "object_summary.csv", summary)
    write_report(args.output_dir / "REPORT.md", summary, args.geometry_only)
    print((args.output_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
