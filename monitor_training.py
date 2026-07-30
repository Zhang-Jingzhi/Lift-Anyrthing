import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BATCH_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+) "
    r"batch (?P<batch>\d+)/(?P<batches>\d+) "
    r"loss=(?P<loss>[-+.\deE]+)"
)
EPOCH_PATTERN = re.compile(
    r"Epoch (?P<epoch>\d+)/(?P<epochs>\d+) "
    r"completed: avg_loss=(?P<loss>[-+.\deE]+)"
)


def read_current_run(log_path):
    lines = log_path.read_text(errors="replace").splitlines()

    # Training is intentionally restarted after each evaluation. Keep one latest
    # entry per epoch/batch so the curve remains continuous across resumptions.
    batches_by_key = {}
    epochs_by_key = {}
    for line in lines:
        batch_match = BATCH_PATTERN.search(line)
        if batch_match:
            row = {key: int(value) for key, value in batch_match.groupdict().items()
                   if key != "loss"}
            row["loss"] = float(batch_match.group("loss"))
            row["global_step"] = (
                (row["epoch"] - 1) * row["batches"] + row["batch"]
            )
            batches_by_key[(row["epoch"], row["batch"])] = row

        epoch_match = EPOCH_PATTERN.search(line)
        if epoch_match:
            row = {
                "epoch": int(epoch_match.group("epoch")),
                "epochs": int(epoch_match.group("epochs")),
                "loss": float(epoch_match.group("loss")),
            }
            epochs_by_key[row["epoch"]] = row
    batches = [
        batches_by_key[key]
        for key in sorted(batches_by_key)
    ]
    epochs = [
        epochs_by_key[key]
        for key in sorted(epochs_by_key)
    ]
    return batches, epochs


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values, window):
    if len(values) < window:
        return np.asarray(values)
    kernel = np.ones(window, dtype=np.float64) / window
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, np.convolve(values, kernel, mode="valid")])


def convergence_status(epoch_rows, min_epochs, patience, relative_min_delta):
    if not epoch_rows:
        return False, "Waiting for the first completed epoch.", None, None

    best_loss = float("inf")
    best_epoch = None
    last_improvement_epoch = None
    for row in epoch_rows:
        threshold = best_loss * (1.0 - relative_min_delta)
        if row["loss"] < threshold:
            best_loss = row["loss"]
            best_epoch = row["epoch"]
            last_improvement_epoch = row["epoch"]

    current_epoch = epoch_rows[-1]["epoch"]
    stale_epochs = current_epoch - last_improvement_epoch
    should_stop = current_epoch >= min_epochs and stale_epochs >= patience
    message = (
        f"epoch={current_epoch}, best_loss={best_loss:.6f} at epoch "
        f"{best_epoch}, epochs_without_meaningful_improvement={stale_epochs}/"
        f"{patience}, relative_min_delta={relative_min_delta:.4%}"
    )
    return should_stop, message, best_epoch, best_loss


def screen_is_running(name):
    result = subprocess.run(
        ["screen", "-ls"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f".{name}" in result.stdout


def update(args):
    batches, epochs = read_current_run(args.log)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output_dir / "training_loss.csv",
        ["epoch", "epochs", "batch", "batches", "global_step", "loss"],
        batches,
    )
    write_csv(
        args.output_dir / "epoch_loss.csv",
        ["epoch", "epochs", "loss"],
        epochs,
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    if batches:
        steps = np.asarray([row["global_step"] for row in batches])
        losses = np.asarray([row["loss"] for row in batches])
        axes[0].plot(steps, losses, alpha=0.35, marker=".", label="logged batch loss")
        axes[0].plot(
            steps,
            rolling_mean(losses, args.rolling_window),
            linewidth=2,
            label=f"rolling mean ({args.rolling_window} logged points)",
        )
        axes[0].set_yscale("log")
        axes[0].legend()
    axes[0].set_title("TRO-Grasp multi-hand training loss")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].grid(alpha=0.25)

    if epochs:
        axes[1].plot(
            [row["epoch"] for row in epochs],
            [row["loss"] for row in epochs],
            marker="o",
        )
    axes[1].set_title("Epoch average loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("average loss")
    axes[1].grid(alpha=0.25)
    figure.savefig(args.output_dir / "training_curve.png", dpi=160)
    plt.close(figure)

    should_stop, convergence, best_epoch, best_loss = convergence_status(
        epochs,
        args.min_epochs,
        args.patience,
        args.relative_min_delta,
    )
    running = screen_is_running(args.screen_name)
    latest_batch = batches[-1] if batches else None
    summary_lines = [
        f"screen_running={running}",
        f"logged_batch_points={len(batches)}",
        f"completed_epochs={len(epochs)}",
        f"convergence={convergence}",
    ]
    if latest_batch:
        summary_lines.append(
            "latest="
            f"epoch {latest_batch['epoch']}/{latest_batch['epochs']} "
            f"batch {latest_batch['batch']}/{latest_batch['batches']} "
            f"loss {latest_batch['loss']:.6f}"
        )
    (args.output_dir / "status.txt").write_text("\n".join(summary_lines) + "\n")

    if should_stop and args.auto_stop and running:
        if not args.latest_checkpoint.is_file():
            raise RuntimeError(
                f"Refusing early stop: checkpoint is missing: "
                f"{args.latest_checkpoint}"
            )
        subprocess.run(
            ["screen", "-S", args.screen_name, "-X", "quit"],
            check=True,
        )
        stop_message = (
            f"Early-stopped {args.screen_name}: {convergence}; "
            f"checkpoint={args.latest_checkpoint}\n"
        )
        (args.output_dir / "early_stop.txt").write_text(stop_message)
        print(stop_message, end="", flush=True)
        return False

    print(
        f"Updated curve: batches={len(batches)}, epochs={len(epochs)}, "
        f"running={running}; {convergence}",
        flush=True,
    )
    return running


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-name", default="tro_train_300")
    parser.add_argument("--latest-checkpoint", type=Path, required=True)
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--relative-min-delta", type=float, default=0.001)
    parser.add_argument("--auto-stop", action="store_true")
    args = parser.parse_args()

    while True:
        running = update(args)
        if args.watch_seconds <= 0 or not running:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
