#!/usr/bin/env python3
"""Compose six rendered three-view examples into one sheet per method."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.selection_json.read_text(encoding="utf-8"))
    for method, rows in report["methods"].items():
        figure, axes = plt.subplots(2, 3, figsize=(24, 7.8))
        for axis, row in zip(axes.flat, rows):
            image_path = (
                args.render_root
                / method
                / f"{row['number']:02d}_{row['category']}_three_views.png"
            )
            axis.imshow(plt.imread(image_path))
            axis.set_axis_off()
            axis.set_title(
                f"{row['number']}. {row['label']} | links "
                f"{row['left_links']}/{row['right_links']} | "
                f"lift {row['lift_mm']:.1f} mm | "
                f"6-dir {row['six_direction_mm']:.2f} mm",
                fontsize=10,
                fontweight="bold",
            )
        figure.suptitle(
            f"{method.upper()} — six strict 3/3-rollout toy-airplane examples",
            fontsize=18,
            fontweight="bold",
        )
        figure.subplots_adjust(
            left=0.01, right=0.99, bottom=0.02, top=0.91, wspace=0.02, hspace=0.13
        )
        output = args.render_root / f"{method}_six_examples_overview.png"
        if output.exists():
            raise FileExistsError(output)
        figure.savefig(output, dpi=160, facecolor="white")
        plt.close(figure)
        print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
