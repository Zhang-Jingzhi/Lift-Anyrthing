"""Pair two independent Allegro grasps and evaluate them in bimanual Isaac Gym."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from utils.controller import controller
from utils.hand_model import create_hand_model


def opposite_root_matching(q_batch):
    roots = q_batch[:, :3]
    roots = roots / roots.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosine = roots @ roots.T
    remaining = set(range(len(q_batch)))
    pairs = []
    while len(remaining) >= 2:
        indices = sorted(remaining)
        best = min(
            (
                (float(cosine[left, right]), left, right)
                for offset, left in enumerate(indices)
                for right in indices[offset + 1 :]
            ),
            key=lambda item: item[0],
        )
        score, left, right = best
        remaining.remove(left)
        remaining.remove(right)
        pairs.append((left, right, score))
    return pairs


def transformed_points(hand, q):
    links, _ = hand.get_transformed_links_pc(q)
    return torch.cat(list(links.values()), dim=-2)


def hand_clearances(left_hand, left_q, right_q, right_hand=None):
    right_hand = right_hand or left_hand
    left_outer, left_inner = controller(
        left_hand.robot_name,
        left_q,
        hand=left_hand,
    )
    right_outer, right_inner = controller(
        right_hand.robot_name,
        right_q,
        hand=right_hand,
    )
    rows = []
    for index in range(len(left_q)):
        outer_distance = torch.cdist(
            transformed_points(left_hand, left_outer[index]),
            transformed_points(right_hand, right_outer[index]),
        ).min()
        inner_distance = torch.cdist(
            transformed_points(left_hand, left_inner[index]),
            transformed_points(right_hand, right_inner[index]),
        ).min()
        rows.append(
            (
                float(outer_distance * 1000),
                float(inner_distance * 1000),
            )
        )
    return rows


def run_isaac(args, object_name, left_q, right_q, object_dir):
    left_path = object_dir / "left_q.pt"
    right_path = object_dir / "right_q.pt"
    result_path = object_dir / "isaac_result.pt"
    torch.save(left_q, left_path)
    torch.save(right_q, right_path)
    if result_path.is_file() and not args.force:
        return torch.load(result_path, map_location="cpu")

    command = [
        str(args.isaac_python),
        str(args.repo / "validation/bimanual_isaac_main.py"),
        "--object-name",
        object_name,
        "--left-q-file",
        str(left_path),
        "--right-q-file",
        str(right_path),
        "--output-file",
        str(result_path),
        "--gpu",
        str(args.gpu),
    ]
    left_robot_name = getattr(args, "left_robot_name", None)
    right_robot_name = getattr(args, "right_robot_name", None)
    if left_robot_name or right_robot_name:
        if not left_robot_name or not right_robot_name:
            raise ValueError("Both left_robot_name and right_robot_name are required")
        command.extend(
            [
                "--left-robot-name",
                left_robot_name,
                "--right-robot-name",
                right_robot_name,
            ]
        )
    else:
        command.extend(
            ["--robot-name", getattr(args, "robot_name", "allegro")]
        )
    command.extend(
        [
            "--gravity",
            str(getattr(args, "gravity", 0.0)),
            "--gravity-settle-step",
            str(getattr(args, "gravity_settle_step", 100)),
            "--active-hands",
            getattr(args, "active_hands", "both"),
            "--lift-height",
            str(getattr(args, "lift_height", 0.0)),
            "--lift-step",
            str(getattr(args, "lift_step", 100)),
            "--min-lift-height",
            str(getattr(args, "min_lift_height", 0.03)),
            "--robot-friction",
            str(getattr(args, "robot_friction", 3.0)),
            "--object-friction",
            str(getattr(args, "object_friction", 3.0)),
            "--contact-offset",
            str(getattr(args, "contact_offset", 0.01)),
            "--max-gravity-displacement",
            str(getattr(args, "max_gravity_displacement", 0.02)),
            "--max-direction-displacement",
            str(getattr(args, "max_direction_displacement", 0.02)),
            "--object-density",
            str(getattr(args, "object_density", 500.0)),
        ]
    )
    finger_effort_limit = getattr(args, "finger_effort_limit", None)
    if finger_effort_limit is not None:
        command.extend(
            ["--finger-effort-limit", str(finger_effort_limit)]
        )
    if getattr(args, "staged_gravity", False):
        command.append("--staged-gravity")
    if getattr(args, "independent_directions", False):
        command.append("--independent-directions")
    if getattr(args, "gravity_only", False):
        command.append("--gravity-only")
    if getattr(args, "support_during_closure", False):
        command.append("--support-during-closure")
    if getattr(args, "fixture_during_closure", False):
        command.append("--fixture-during-closure")
    if getattr(args, "capture_contacts", False):
        command.append("--capture-contacts")
    if getattr(args, "object_vhacd", False):
        command.extend(
            [
                "--object-vhacd",
                "--object-vhacd-resolution",
                str(getattr(args, "object_vhacd_resolution", 300000)),
                "--object-vhacd-max-convex-hulls",
                str(
                    getattr(
                        args, "object_vhacd_max_convex_hulls", 64
                    )
                ),
                "--object-vhacd-max-vertices",
                str(getattr(args, "object_vhacd_max_vertices", 64)),
            ]
        )
    if getattr(args, "object_multicollision", False):
        command.append("--object-multicollision")
    if getattr(args, "object_vhacd_high_v1", False):
        command.append("--object-vhacd-high-v1")
    if getattr(args, "object_vhacd_visual_high_v2", False):
        command.append("--object-vhacd-visual-high-v2")
    environment = os.environ.copy()
    environment["PATH"] = (
        str(args.isaac_python.parent)
        + ":"
        + environment.get("PATH", "")
    )
    environment["LD_LIBRARY_PATH"] = (
        str(args.isaac_python.parent.parent / "lib")
        + ":"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    nvidia_icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.is_file():
        environment["VK_ICD_FILENAMES"] = str(nvidia_icd)
    with (object_dir / "isaac.log").open("w") as log:
        print("COMMAND:", shlex.join(command), file=log, flush=True)
        subprocess.run(
            command,
            cwd=args.repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return torch.load(result_path, map_location="cpu")


def aggregate(sample_rows):
    rows = []
    object_names = sorted({row["object_name"] for row in sample_rows})
    for object_name in object_names:
        samples = [
            row for row in sample_rows if row["object_name"] == object_name
        ]
        rows.append(
            {
                "object_name": object_name,
                "num_pairs": len(samples),
                "isaac_success_rate": np.mean(
                    [row["isaac_success"] for row in samples]
                ),
                "both_object_penetration_pass_rate": np.mean(
                    [row["both_object_penetration_pass"] for row in samples]
                ),
                "hand_hand_clearance_pass_rate": np.mean(
                    [row["hand_hand_clearance_pass"] for row in samples]
                ),
                "strict_joint_success_rate": np.mean(
                    [row["strict_joint_success"] for row in samples]
                ),
                "mean_settle_displacement_mm": np.mean(
                    [row["settle_displacement_mm"] for row in samples]
                ),
                "mean_disturbance_displacement_mm": np.mean(
                    [row["disturbance_displacement_mm"] for row in samples]
                ),
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    lines = [
        "# Two-independent-grasp bimanual baseline",
        "",
        "Two independently generated Allegro grasps are greedily paired by",
        "opposite root-position direction. There is no joint optimization.",
        "This deliberately simple method is the baseline suggested by the advisor.",
        "",
        "`Strict joint` requires: bimanual Isaac success, both hand-object",
        "penetration checks pass at 5 mm, and sampled hand-hand clearance exceeds",
        "2 mm in both the open and controller-target poses.",
        "",
        "| Object | Pairs | Isaac | Object mesh pass | Hand clearance | Strict joint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['object_name']}` | {row['num_pairs']} | "
            f"{row['isaac_success_rate']:.1%} | "
            f"{row['both_object_penetration_pass_rate']:.1%} | "
            f"{row['hand_hand_clearance_pass_rate']:.1%} | "
            f"{row['strict_joint_success_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Isaac success: "
            f"{np.mean([row['isaac_success_rate'] for row in rows]):.1%}.",
            f"- Strict joint success: "
            f"{np.mean([row['strict_joint_success_rate'] for row in rows]):.1%}.",
            "",
            "This baseline uses two copies of the same Allegro embodiment. It is",
            "a simulation proof of concept, not yet a mirrored left/right hardware",
            "configuration.",
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
            "graph_exp/bimanual_large_object/two_single_grasp_baseline"
        ),
    )
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(os.environ.get("ISAAC_PYTHON", "python")),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--clearance-mm", type=float, default=2.0)
    parser.add_argument(
        "--objects",
        nargs="+",
        help=(
            "Optional object subset for a smoke test. Names use the "
            "dataset+object form, for example ycb+pitcher_base."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    for name in ("source_vis", "object_split", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, (args.repo / value).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.object_split.read_text())
    object_names = (
        args.objects
        if args.objects
        else split["development"] + split["held_out_test"]
    )
    entries = torch.load(args.source_vis, map_location="cpu")
    entry_lookup = {entry["object_name"]: entry for entry in entries}
    hand = create_hand_model("allegro", torch.device("cpu"))

    sample_rows = []
    for object_name in object_names:
        print(f"Pairing {object_name}", flush=True)
        entry = entry_lookup[object_name]
        pairs = opposite_root_matching(entry["predict_q"])
        left_indices = [pair[0] for pair in pairs]
        right_indices = [pair[1] for pair in pairs]
        left_q = entry["predict_q"][left_indices]
        right_q = entry["predict_q"][right_indices]
        clearances = hand_clearances(hand, left_q, right_q)

        object_dir = args.output_dir / object_name.replace("+", "__")
        object_dir.mkdir(parents=True, exist_ok=True)
        isaac = run_isaac(args, object_name, left_q, right_q, object_dir)
        for pair_index, (left_index, right_index, cosine) in enumerate(pairs):
            outer_clearance, inner_clearance = clearances[pair_index]
            clearance_pass = (
                outer_clearance > args.clearance_mm
                and inner_clearance > args.clearance_mm
            )
            object_penetration_pass = bool(
                entry["penetration_success"][left_index]
                and entry["penetration_success"][right_index]
            )
            isaac_success = bool(isaac["success"][pair_index])
            sample_rows.append(
                {
                    "object_name": object_name,
                    "pair_index": pair_index,
                    "left_source_index": left_index,
                    "right_source_index": right_index,
                    "root_direction_cosine": cosine,
                    "left_source_rollout_success": bool(
                        entry["success"][left_index]
                    ),
                    "right_source_rollout_success": bool(
                        entry["success"][right_index]
                    ),
                    "both_object_penetration_pass": object_penetration_pass,
                    "outer_hand_clearance_mm": outer_clearance,
                    "inner_hand_clearance_mm": inner_clearance,
                    "hand_hand_clearance_pass": clearance_pass,
                    "settle_displacement_mm": float(
                        isaac["settle_displacement"][pair_index] * 1000
                    ),
                    "disturbance_displacement_mm": float(
                        isaac["disturbance_displacement"][pair_index] * 1000
                    ),
                    "isaac_success": isaac_success,
                    "strict_joint_success": isaac_success
                    and object_penetration_pass
                    and clearance_pass,
                }
            )

    summary_rows = aggregate(sample_rows)
    write_csv(args.output_dir / "sample_results.csv", sample_rows)
    write_csv(args.output_dir / "object_summary.csv", summary_rows)
    write_report(args.output_dir / "REPORT.md", summary_rows)
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
