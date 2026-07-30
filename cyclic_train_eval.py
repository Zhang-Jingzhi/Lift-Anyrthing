import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
SUCCESS_PATTERN = re.compile(rf"Total success rate: {NUMBER}")
DIVERSITY_PATTERN = re.compile(rf"Total diversity: {NUMBER}")
TIME_PATTERN = re.compile(rf"Grasp generation time: {NUMBER}")
PENETRATION_PASS_PATTERN = re.compile(rf"Penetration pass rate: {NUMBER}")
PENETRATION_METHOD_PATTERN = re.compile(r"Penetration method: ([A-Za-z0-9_-]+)")
PENETRATION_DEPTH_PATTERN = re.compile(
    rf"Penetration depth \(mm\): mean={NUMBER}, "
    rf"p95={NUMBER}, max={NUMBER}"
)


def log(message, status_path=None):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if status_path is not None:
        status_path.write_text(line + "\n")


def screen_is_running(name):
    result = subprocess.run(
        ["screen", "-ls"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f".{name}" in result.stdout


def stop_screen(name):
    if screen_is_running(name):
        subprocess.run(["screen", "-S", name, "-X", "quit"], check=True)
    deadline = time.time() + 60
    while screen_is_running(name):
        if time.time() >= deadline:
            raise TimeoutError(f"screen session did not stop: {name}")
        time.sleep(1)


def checkpoint_epoch(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return int(checkpoint["epoch"])
    except (EOFError, OSError, RuntimeError, KeyError):
        return None


def wait_for_checkpoint(path, epoch, train_screen, status_path, poll_seconds):
    last_curve_update = 0
    while True:
        if path.is_file():
            found_epoch = checkpoint_epoch(path)
            if found_epoch == epoch:
                return

        if not screen_is_running(train_screen):
            raise RuntimeError(
                f"training stopped before checkpoint {epoch} was ready"
            )

        now = time.time()
        if now - last_curve_update >= 60:
            log(
                f"training toward epoch {epoch}; waiting for {path.name}",
                status_path,
            )
            last_curve_update = now
        time.sleep(poll_seconds)


def launch_training(args, resume_checkpoint, stop_epoch):
    command = [
        "screen",
        "-dmS",
        args.train_screen,
        "-L",
        "-Logfile",
        str(args.train_log),
        "env",
        "WANDB_MODE=offline",
        "WANDB_SILENT=true",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        "train.py",
        "--config",
        str(args.train_config),
        "--resume-from",
        str(resume_checkpoint),
        "--stop-after-epoch",
        str(stop_epoch),
    ]
    subprocess.run(command, cwd=args.repo, check=True)
    deadline = time.time() + 30
    while not screen_is_running(args.train_screen):
        if time.time() >= deadline:
            raise RuntimeError("training screen failed to start")
        time.sleep(1)
    launch_monitor(args)


def launch_monitor(args):
    stop_screen(args.monitor_screen)
    monitor_dir = args.output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "screen",
        "-dmS",
        args.monitor_screen,
        "-L",
        "-Logfile",
        str(monitor_dir / "monitor.screen.log"),
        "env",
        "MPLBACKEND=Agg",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        "monitor_training.py",
        "--log",
        str(args.train_log),
        "--output-dir",
        str(monitor_dir),
        "--latest-checkpoint",
        str(args.checkpoint_dir / "latest.pth"),
        "--screen-name",
        args.train_screen,
        "--watch-seconds",
        "60",
    ]
    subprocess.run(command, cwd=args.repo, check=True)


def parse_result(path):
    text = path.read_text()

    def one(pattern):
        match = pattern.search(text)
        return float(match.group(1)) if match else ""

    depth = PENETRATION_DEPTH_PATTERN.search(text)
    method = PENETRATION_METHOD_PATTERN.search(text)
    return {
        "success_rate": one(SUCCESS_PATTERN),
        "diversity": one(DIVERSITY_PATTERN),
        "generation_time_s": one(TIME_PATTERN),
        "penetration_pass_rate": one(PENETRATION_PASS_PATTERN),
        "penetration_mean_mm": float(depth.group(1)) if depth else "",
        "penetration_p95_mm": float(depth.group(2)) if depth else "",
        "penetration_max_mm": float(depth.group(3)) if depth else "",
        "penetration_method": method.group(1) if method else "",
    }


def append_summary(path, row):
    fieldnames = [
        "epoch",
        "evaluation",
        "success_rate",
        "diversity",
        "generation_time_s",
        "penetration_pass_rate",
        "penetration_mean_mm",
        "penetration_p95_mm",
        "penetration_max_mm",
        "penetration_method",
        "result_path",
    ]
    rows = []
    if path.is_file():
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
    row_key = (str(row["epoch"]), row["evaluation"])
    rows = [
        existing
        for existing in rows
        if (existing["epoch"], existing["evaluation"]) != row_key
    ]
    rows.append({key: row.get(key, "") for key in fieldnames})
    rows.sort(key=lambda value: (int(value["epoch"]), value["evaluation"]))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(args, epoch, name, base_config):
    save_dir = args.output_dir / "evaluations" / f"epoch_{epoch:03d}" / name
    result_path = save_dir / "res.txt"
    if result_path.is_file():
        log(f"reusing completed evaluation: {result_path}")
        return result_path

    save_dir.mkdir(parents=True, exist_ok=True)
    eval_log = save_dir / "evaluation.log"
    command = [
        sys.executable,
        "evaluate_checkpoint.py",
        "--base-config",
        str(base_config),
        "--checkpoint",
        str(args.checkpoint_dir / f"{epoch}.pth"),
        "--save-dir",
        str(save_dir),
        "--batch-size",
        str(args.eval_batch_size),
        "--split-batch-size",
        str(args.eval_split_batch_size),
        "--seed",
        str(args.seed),
        "--objects",
        *args.objects,
    ]
    for attempt in range(1, args.eval_retries + 1):
        with eval_log.open("a") as file:
            file.write(f"\n=== evaluation attempt {attempt} ===\n")
            result = subprocess.run(
                command,
                cwd=args.repo,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0 and result_path.is_file():
            return result_path
        log(
            f"epoch {epoch} {name} attempt {attempt}/"
            f"{args.eval_retries} failed with exit code {result.returncode}"
        )
        if attempt < args.eval_retries:
            time.sleep(10)
    raise RuntimeError(
        f"evaluation failed after {args.eval_retries} attempts: {name}"
    )


def refresh_curve(args):
    command = [
        sys.executable,
        "monitor_training.py",
        "--log",
        str(args.train_log),
        "--output-dir",
        str(args.output_dir / "monitor"),
        "--latest-checkpoint",
        str(args.checkpoint_dir / "latest.pth"),
        "--screen-name",
        args.train_screen,
    ]
    subprocess.run(
        command,
        cwd=args.repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("config/train_multi_hand_rtx3070.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/reproduction/train-multi-hand-rtx3070"
        ),
    )
    parser.add_argument("--train-screen", default="tro_train_300")
    parser.add_argument("--monitor-screen", default="tro_train_monitor")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--initial-epoch", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--eval-split-batch-size", type=int, default=4)
    parser.add_argument("--eval-retries", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--objects",
        nargs="+",
        default=[
            "contactdb+apple",
            "contactdb+camera",
            "contactdb+water_bottle",
        ],
    )
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.train_config = (args.repo / args.train_config).resolve()
    args.output_dir = (args.repo / args.output_dir).resolve()
    args.checkpoint_dir = args.output_dir / "ckpt"
    args.train_log = args.output_dir.with_suffix(".screen.log")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "cycle_status.txt"
    summary_path = args.output_dir / "evaluation_summary.csv"

    evaluations = [
        (
            "shadowhand_unconditioned",
            args.repo / "config/test_palm_unconditioned_rtx3070.yaml",
        ),
        (
            "allegro_conditioned",
            args.repo / "config/test_palm_conditioned_rtx3070.yaml",
        ),
    ]

    epoch = args.initial_epoch
    while epoch <= args.max_epoch:
        checkpoint = args.checkpoint_dir / f"{epoch}.pth"
        log(f"waiting for epoch {epoch} checkpoint", status_path)
        wait_for_checkpoint(
            checkpoint,
            epoch,
            args.train_screen,
            status_path,
            args.poll_seconds,
        )

        # The initial training process predates clean stop support, so terminate
        # it immediately after its checkpoint becomes readable. Later runs stop
        # naturally at each requested epoch.
        stop_screen(args.train_screen)
        log(f"training stopped at epoch {epoch}; starting evaluation", status_path)

        for name, base_config in evaluations:
            result_path = run_evaluation(args, epoch, name, base_config)
            row = {
                "epoch": epoch,
                "evaluation": name,
                **parse_result(result_path),
                "result_path": str(result_path),
            }
            append_summary(summary_path, row)
            log(
                f"epoch {epoch} {name}: "
                f"success={row['success_rate']}, "
                f"penetration_pass={row['penetration_pass_rate']}",
                status_path,
            )

        refresh_curve(args)
        if epoch >= args.max_epoch:
            log("completed all training/evaluation cycles", status_path)
            return

        next_epoch = min(epoch + args.interval, args.max_epoch)
        launch_training(args, checkpoint, next_epoch)
        log(
            f"evaluation complete; resumed training toward epoch {next_epoch}",
            status_path,
        )
        epoch = next_epoch


if __name__ == "__main__":
    main()
