#!/usr/bin/env python3
"""Train in chunks and automatically trigger geometry-first evaluation."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


DEVELOPMENT_OBJECTS = (
    "ycb+bleach_cleanser",
    "ycb+wood_block",
    "contactdb+piggy_bank",
    "ycb+power_drill",
    "contactdb+cube_large",
    "contactdb+cylinder_large",
)
HELD_OUT_OBJECTS = (
    "ycb+pitcher_base",
    "ycb+cracker_box",
    "ycb+toy_airplane",
)


def run(command, log_path, repo, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\nCOMMAND: " + " ".join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(
            list(map(str, command)),
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def true_count(csv_path, key):
    if not csv_path.is_file():
        return 0
    with csv_path.open(newline="") as file:
        return sum(
            row[key].strip().lower() == "true"
            for row in csv.DictReader(file)
        )


def root_separation_summary(q_path):
    generated = torch.load(
        q_path,
        map_location="cpu",
        weights_only=False,
    )
    separation = (
        generated["left_q_seed"][:, :3]
        - generated["right_q_seed"][:, :3]
    ).norm(dim=1) * 1000
    return {
        "mean_mm": float(separation.mean()),
        "min_mm": float(separation.min()),
        "max_mm": float(separation.max()),
    }


def evaluate(
    args,
    checkpoint,
    objects,
    output_dir,
    log_path,
    environment,
    geometry_only,
):
    command = [
        sys.executable,
        args.repo / "evaluate_bimanual_checkpoint.py",
        "--config",
        args.config,
        "--checkpoint",
        checkpoint,
        "--objects",
        *objects,
        "--output-dir",
        output_dir,
        "--samples-per-object",
        str(args.samples_per_object),
        "--ddim-steps",
        str(args.ddim_steps),
        "--force",
    ]
    if geometry_only:
        command.append("--geometry-only")
    if args.refine_geometry:
        command.append("--refine-geometry")
    run(command, log_path, args.repo, environment)


def csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_combined_report(path, summary, geometry_only):
    lines = [
        "# Bimanual checkpoint strict evaluation",
        "",
        "Each object was evaluated in an isolated process so exact-mesh "
        "refinement memory is released between objects.",
        "",
        "| Object | N | Joint | IK projection | Penetration | "
        "Dual contact | Clearance | Geometry | Isaac/all | Strict/all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        rates = [
            float(row[key]) * 100
            for key in (
                "joint_limit_rate",
                "ik_projection_rate",
                "penetration_rate",
                "dual_contact_rate",
                "clearance_rate",
                "geometry_rate",
                "isaac_rate_all",
                "strict_rate_all",
            )
        ]
        lines.append(
            f"| `{row['object_name']}` | {row['samples']} | "
            + " | ".join(f"{rate:.1f}%" for rate in rates)
            + " |"
        )
    if geometry_only:
        lines.extend(
            [
                "",
                "This run used `--geometry-only`; Isaac and strict rates "
                "are placeholders.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def evaluate_isolated(
    args,
    checkpoint,
    objects,
    output_dir,
    log_path,
    environment,
    geometry_only,
):
    """Evaluate geometry and Isaac in separate per-object processes."""
    parts_dir = output_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    summary_rows = []
    with log_path.open("a") as aggregate_log:
        aggregate_log.write(
            "\nStarting isolated per-object evaluation.\n"
        )
    for object_name in objects:
        slug = object_name.replace("+", "__")
        part_dir = parts_dir / slug
        part_samples = part_dir / "sample_results.csv"
        geometry_rows = csv_rows(part_samples)
        if len(geometry_rows) != args.samples_per_object:
            evaluate(
                args,
                checkpoint,
                (object_name,),
                part_dir,
                log_path.with_name(
                    f"{log_path.stem}__{slug}__geometry.log"
                ),
                environment,
                geometry_only=True,
            )
            geometry_rows = csv_rows(part_samples)
        if len(geometry_rows) != args.samples_per_object:
            raise RuntimeError(
                f"Incomplete geometry evaluation for {object_name}: "
                f"{len(geometry_rows)}/{args.samples_per_object}"
            )
        part_summary = csv_rows(part_dir / "object_summary.csv")
        if len(part_summary) != 1:
            raise RuntimeError(
                f"Missing object summary for {object_name}"
            )
        completed_rows = geometry_rows
        if not geometry_only:
            geometry_count = sum(
                row["geometry_pass"].strip().lower() == "true"
                for row in geometry_rows
            )
            rollout_csv = part_dir / "refined_rollout_results.csv"
            rollout_rows = csv_rows(rollout_csv)
            if geometry_count and (
                len(rollout_rows) != args.samples_per_object
            ):
                rollout_command = [
                    sys.executable,
                    args.repo / "run_refined_bimanual_rollout.py",
                    "--evaluation-dir",
                    part_dir,
                    "--object-name",
                    object_name,
                    "--gpu",
                    "0",
                ]
                run(
                    rollout_command,
                    log_path.with_name(
                        f"{log_path.stem}__{slug}__isaac.log"
                    ),
                    args.repo,
                    environment,
                )
                rollout_rows = csv_rows(rollout_csv)
            if geometry_count:
                if len(rollout_rows) != args.samples_per_object:
                    raise RuntimeError(
                        f"Incomplete Isaac evaluation for {object_name}: "
                        f"{len(rollout_rows)}/{args.samples_per_object}"
                    )
                completed_rows = rollout_rows
            part_summary[0]["isaac_rate_all"] = str(
                sum(
                    row["isaac_success"].strip().lower() == "true"
                    for row in completed_rows
                )
                / args.samples_per_object
            )
            part_summary[0]["strict_rate_all"] = str(
                sum(
                    row["strict_success"].strip().lower() == "true"
                    for row in completed_rows
                )
                / args.samples_per_object
            )
            write_csv_rows(
                part_dir / "object_summary.csv",
                part_summary,
            )
        sample_rows.extend(completed_rows)
        summary_rows.extend(part_summary)
        with log_path.open("a") as aggregate_log:
            aggregate_log.write(
                f"completed {object_name}: "
                f"geometry={part_summary[0]['geometry_rate']}, "
                f"strict={part_summary[0]['strict_rate_all']}\n"
            )
    write_csv_rows(output_dir / "sample_results.csv", sample_rows)
    write_csv_rows(output_dir / "object_summary.csv", summary_rows)
    write_combined_report(
        output_dir / "REPORT.md",
        summary_rows,
        geometry_only,
    )


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train_bimanual_smoke_rtx3070.yaml"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, default=30)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--samples-per-object", type=int, default=10)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=603,
    )
    parser.add_argument("--refine-geometry", action="store_true")
    parser.add_argument(
        "--evaluation-only-epoch",
        type=int,
        help="Skip training/screening and resume isolated development evaluation.",
    )
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.run_dir = (
        args.run_dir
        if args.run_dir.is_absolute()
        else (args.repo / args.run_dir)
    ).resolve()
    args.config = (
        args.config
        if args.config.is_absolute()
        else (args.repo / args.config)
    ).resolve()
    supervisor_dir = args.run_dir / "supervisor"
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    state_path = supervisor_dir / "state.json"

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["XLA_PLATFORMS"] = "cpu"
    if args.evaluation_only_epoch is not None:
        target = args.evaluation_only_epoch
        numbered_checkpoint = args.run_dir / "ckpt" / f"{target}.pth"
        if not numbered_checkpoint.is_file():
            raise FileNotFoundError(numbered_checkpoint)
        if state_path.is_file():
            state = json.loads(state_path.read_text())
        else:
            state = {
                "status": "running",
                "start_epoch": target,
                "max_epoch": target,
                "interval": args.interval,
                "completed": [],
            }
        state["status"] = "evaluating"
        record = next(
            (
                item
                for item in state["completed"]
                if item.get("epoch") == target
            ),
            None,
        )
        if record is None:
            record = {"epoch": target}
            state["completed"].append(record)
        write_state(state_path, state)
        development_dir = (
            supervisor_dir / f"development_epoch_{target:03d}"
        )
        evaluate_isolated(
            args,
            numbered_checkpoint,
            DEVELOPMENT_OBJECTS,
            development_dir,
            supervisor_dir / f"development_epoch_{target:03d}.log",
            environment,
            geometry_only=False,
        )
        development_strict = true_count(
            development_dir / "sample_results.csv",
            "strict_success",
        )
        record["development_strict_success"] = development_strict
        if development_strict:
            held_out_dir = (
                supervisor_dir / f"held_out_epoch_{target:03d}"
            )
            evaluate_isolated(
                args,
                numbered_checkpoint,
                HELD_OUT_OBJECTS,
                held_out_dir,
                supervisor_dir / f"held_out_epoch_{target:03d}.log",
                environment,
                geometry_only=False,
            )
            record["held_out_strict_success"] = true_count(
                held_out_dir / "sample_results.csv",
                "strict_success",
            )
            state["status"] = "completed_with_development_success"
            state["selected_epoch"] = target
        else:
            state["status"] = (
                "completed_evaluation_without_development_success"
            )
        write_state(state_path, state)
        print(
            f"evaluation epoch={target} development_strict="
            f"{development_strict}",
            flush=True,
        )
        return

    state = {
        "status": "running",
        "start_epoch": args.start_epoch,
        "max_epoch": args.max_epoch,
        "interval": args.interval,
        "completed": [],
    }
    write_state(state_path, state)

    current = args.start_epoch
    while current < args.max_epoch:
        target = min(current + args.interval, args.max_epoch)
        checkpoint = args.run_dir / "ckpt/latest.pth"
        train_log = supervisor_dir / f"train_{current + 1:03d}_{target:03d}.log"
        train_command = [
                sys.executable,
                args.repo / "train.py",
                "--config",
                args.config,
                "--epochs",
                str(target),
                "--max-batches-per-epoch",
                str(args.max_batches_per_epoch),
                "--save-dir",
                args.run_dir,
            ]
        if checkpoint.is_file():
            train_command.extend(["--resume-from", checkpoint])
        elif current:
            raise FileNotFoundError(
                f"Missing resume checkpoint at epoch {current}: "
                f"{checkpoint}"
            )
        run(
            train_command,
            train_log,
            args.repo,
            environment,
        )
        numbered_checkpoint = args.run_dir / "ckpt" / f"{target}.pth"
        screen_dir = supervisor_dir / f"screen_epoch_{target:03d}"
        evaluate(
            args,
            numbered_checkpoint,
            ("ycb+power_drill",),
            screen_dir,
            supervisor_dir / f"screen_epoch_{target:03d}.log",
            environment,
            geometry_only=True,
        )
        geometry_count = true_count(
            screen_dir / "sample_results.csv",
            "geometry_pass",
        )
        separation = root_separation_summary(
            screen_dir / "ycb__power_drill/generated_q.pt"
        )
        record = {
            "epoch": target,
            "screen_geometry_pass": geometry_count,
            "root_separation": separation,
        }
        state["completed"].append(record)
        state["last_epoch"] = target
        write_state(state_path, state)
        print(
            f"epoch={target} geometry={geometry_count}/"
            f"{args.samples_per_object} root_mean="
            f"{separation['mean_mm']:.2f}mm",
            flush=True,
        )

        if geometry_count:
            development_dir = supervisor_dir / f"development_epoch_{target:03d}"
            evaluate_isolated(
                args,
                numbered_checkpoint,
                DEVELOPMENT_OBJECTS,
                development_dir,
                supervisor_dir / f"development_epoch_{target:03d}.log",
                environment,
                geometry_only=False,
            )
            development_strict = true_count(
                development_dir / "sample_results.csv",
                "strict_success",
            )
            record["development_strict_success"] = development_strict
            write_state(state_path, state)
            if development_strict:
                held_out_dir = supervisor_dir / f"held_out_epoch_{target:03d}"
                evaluate_isolated(
                    args,
                    numbered_checkpoint,
                    HELD_OUT_OBJECTS,
                    held_out_dir,
                    supervisor_dir / f"held_out_epoch_{target:03d}.log",
                    environment,
                    geometry_only=False,
                )
                record["held_out_strict_success"] = true_count(
                    held_out_dir / "sample_results.csv",
                    "strict_success",
                )
                state["status"] = "completed_with_development_success"
                state["selected_epoch"] = target
                write_state(state_path, state)
                print(
                    f"selected epoch {target}; development strict="
                    f"{development_strict}, held-out strict="
                    f"{record['held_out_strict_success']}",
                    flush=True,
                )
                return
        current = target

    state["status"] = "completed_max_epoch_without_development_success"
    write_state(state_path, state)
    print("Reached maximum epoch without strict development success", flush=True)


if __name__ == "__main__":
    main()
