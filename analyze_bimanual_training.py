#!/usr/bin/env python3
"""Create a compact convergence report for the bimanual pilot run."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from analyze_bimanual_overfit import (
    fixed_noise_evaluation,
    read_training_metrics,
    write_csv,
)
import torch


def epoch_rows(raw_rows):
    result = []
    for epoch in sorted({int(row["epoch"]) for row in raw_rows}):
        selected = [
            row for row in raw_rows if int(row["epoch"]) == epoch
        ]
        result.append(
            {
                "epoch": epoch,
                **{
                    f"{key}_mean": float(
                        np.mean([row[key] for row in selected])
                    )
                    for key in (
                        "loss_total",
                        "loss_trans",
                        "loss_rot",
                        "loss_penetration",
                        "loss_contact",
                        "loss_inter_hand",
                    )
                },
                "loss_total_median": float(
                    np.median([row["loss_total"] for row in selected])
                ),
                "loss_total_max": float(
                    np.max([row["loss_total"] for row in selected])
                ),
            }
        )
    return result


def plot(path, epochs, fixed):
    x = [row["epoch"] for row in epochs]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(
        x,
        [row["loss_total_mean"] for row in epochs],
        marker="o",
        label="epoch mean",
    )
    axes[0].plot(
        x,
        [row["loss_total_median"] for row in epochs],
        marker="s",
        label="epoch median",
    )
    axes[0].set_title("Stochastic training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    fixed_x = [row["epoch"] for row in fixed]
    axes[1].errorbar(
        fixed_x,
        [row["loss_total"] for row in fixed],
        yerr=[row["loss_total_std"] for row in fixed],
        marker="o",
        capsize=3,
        label="total",
    )
    axes[1].plot(
        fixed_x,
        [row["loss_trans"] for row in fixed],
        marker=".",
        label="translation",
    )
    axes[1].plot(
        fixed_x,
        [row["loss_rot"] for row in fixed],
        marker=".",
        label="rotation",
    )
    axes[1].set_title("Fixed batch + fixed diffusion noise")
    axes[1].set_xlabel("Checkpoint epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Bimanual six-object pilot convergence")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def report(path, epochs, fixed):
    first = epochs[0]
    last = epochs[-1]
    fixed_first = fixed[0]
    fixed_last = fixed[-1]
    finite = all(
        np.isfinite(value)
        for row in epochs
        for key, value in row.items()
        if key != "epoch"
    )
    lines = [
        "# Bimanual six-object pilot training",
        "",
        f"- Completed epochs: {len(epochs)}.",
        f"- Numerically finite: **{'yes' if finite else 'no'}**.",
        (
            f"- Stochastic epoch mean: {first['loss_total_mean']:.4f} "
            f"-> {last['loss_total_mean']:.4f}."
        ),
        (
            f"- Stochastic epoch median: {first['loss_total_median']:.4f} "
            f"-> {last['loss_total_median']:.4f}."
        ),
        (
            f"- Fixed-noise checkpoint loss: "
            f"{fixed_first['loss_total']:.4f} at epoch "
            f"{fixed_first['epoch']} -> {fixed_last['loss_total']:.4f} "
            f"at epoch {fixed_last['epoch']}."
        ),
        "",
        "Random diffusion timesteps produce expected batch-level spikes. The",
        "fixed-noise comparison is used to distinguish those spikes from a",
        "true numerical instability. Grasp quality is reported separately by",
        "the geometry-first Isaac evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = OmegaConf.load(args.config)
    raw = read_training_metrics(run_dir / "batch_metrics.csv")
    epochs = epoch_rows(raw)
    checkpoints = sorted(
        (run_dir / "ckpt").glob("[0-9]*.pth"),
        key=lambda path: int(path.stem),
    )
    fixed = fixed_noise_evaluation(
        config,
        checkpoints,
        range(args.seeds),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    write_csv(run_dir / "epoch_metrics.csv", epochs)
    write_csv(run_dir / "fixed_noise_checkpoint_metrics.csv", fixed)
    plot(run_dir / "training_curve.png", epochs, fixed)
    report(run_dir / "REPORT.md", epochs, fixed)
    print((run_dir / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
