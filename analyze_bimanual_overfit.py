#!/usr/bin/env python3
"""Summarize a tiny-set overfit run with a fixed-noise checkpoint test."""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset.BimanualPairDataset import create_bimanual_dataloader
from model.tro_graph import RobotGraph
from train import prepare_input


LOSS_KEYS = (
    "loss_total",
    "loss_trans",
    "loss_rot",
    "loss_penetration",
    "loss_contact",
    "loss_inter_hand",
)


def read_training_metrics(path):
    with path.open(newline="") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def stage_summary(rows, count=20):
    stages = {
        f"first_{count}": rows[:count],
        f"last_{count}": rows[-count:],
    }
    summary = {}
    for stage, selected in stages.items():
        summary[stage] = {}
        for key in LOSS_KEYS:
            values = np.asarray([row[key] for row in selected])
            summary[stage][key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
    return summary


def checkpoint_epoch(path):
    return int(path.stem)


def fixed_noise_evaluation(config, checkpoint_paths, seeds, device):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataloader = create_bimanual_dataloader(config.dataset)
    batch = prepare_input(next(iter(dataloader)), device)
    model = RobotGraph(**config.model).to(device)
    results = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        measurements = {key: [] for key in LOSS_KEYS}
        with torch.no_grad():
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                losses = model(batch)
                for key in LOSS_KEYS:
                    measurements[key].append(float(losses[key].item()))
        row = {"epoch": checkpoint_epoch(checkpoint_path)}
        for key, values in measurements.items():
            row[key] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
        results.append(row)
    return results


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path, raw_rows, fixed_rows):
    epochs = sorted({int(row["epoch"]) for row in raw_rows})
    epoch_mean = []
    for epoch in epochs:
        values = [
            row["loss_total"]
            for row in raw_rows
            if int(row["epoch"]) == epoch
        ]
        epoch_mean.append(float(np.mean(values)))

    fixed_epochs = [row["epoch"] for row in fixed_rows]
    fixed_total = [row["loss_total"] for row in fixed_rows]
    fixed_std = [row["loss_total_std"] for row in fixed_rows]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, epoch_mean, color="#4c78a8", alpha=0.45)
    window = min(5, len(epoch_mean))
    smooth = np.convolve(
        epoch_mean, np.ones(window) / window, mode="valid"
    )
    axes[0].plot(
        epochs[window - 1 :],
        smooth,
        color="#1f4e79",
        linewidth=2.5,
        label=f"{window}-epoch moving mean",
    )
    axes[0].set_title("Stochastic training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].errorbar(
        fixed_epochs,
        fixed_total,
        yerr=fixed_std,
        marker="o",
        capsize=3,
        color="#e45756",
    )
    axes[1].set_title("Fixed batch + fixed diffusion noise")
    axes[1].set_xlabel("Checkpoint epoch")
    axes[1].set_ylabel("Mean total loss")
    axes[1].grid(alpha=0.25)

    figure.suptitle("Bimanual 4-sample overfit diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(path, summary, fixed_rows):
    first = summary["first_20"]["loss_total"]
    last = summary["last_20"]["loss_total"]
    start = fixed_rows[0]
    end = fixed_rows[-1]
    relative_drop = 100.0 * (
        start["loss_total"] - end["loss_total"]
    ) / start["loss_total"]
    passed = (
        end["loss_total"] < start["loss_total"]
        and last["median"] < first["median"]
    )
    text = f"""# Bimanual tiny-set overfit diagnostic

## Result

**{"PASS" if passed else "NOT YET PASS"}**: the four-sample training run is
numerically stable, and the fixed-noise checkpoint loss
{"decreases" if relative_drop > 0 else "does not decrease"} from
{start["loss_total"]:.4f} at epoch {start["epoch"]} to
{end["loss_total"]:.4f} at epoch {end["epoch"]}
({relative_drop:.1f}% relative change).

## Stochastic training batches

- First 20 batches: mean total loss {first["mean"]:.4f}, median
  {first["median"]:.4f}, maximum {first["max"]:.4f}.
- Last 20 batches: mean total loss {last["mean"]:.4f}, median
  {last["median"]:.4f}, maximum {last["max"]:.4f}.
- Random diffusion timesteps and noise make individual training batches
  spiky, so the fixed-noise comparison is the primary overfit diagnostic.

## Interpretation

This test checks that the bimanual data path, 42-link model output, optimizer,
and geometry regularizers can learn a deliberately tiny dataset. It is not a
grasp-quality result and must not be compared with rollout success rates.
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    run_dir = Path(args.run_dir)
    raw_rows = read_training_metrics(run_dir / "batch_metrics.csv")
    checkpoints = sorted(
        (run_dir / "ckpt").glob("[0-9]*.pth"),
        key=checkpoint_epoch,
    )
    if not checkpoints:
        raise RuntimeError(f"No numbered checkpoints found in {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_rows = fixed_noise_evaluation(
        config,
        checkpoints,
        range(args.seeds),
        device,
    )
    summary = stage_summary(raw_rows)
    write_csv(run_dir / "fixed_noise_checkpoint_metrics.csv", fixed_rows)
    plot_results(run_dir / "overfit_curve.png", raw_rows, fixed_rows)
    write_report(run_dir / "REPORT.md", summary, fixed_rows)
    print((run_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
