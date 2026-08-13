#!/usr/bin/env python3
"""Compose four object renders into one comparison image per method."""

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
        figure, axes = plt.subplots(2, 2, figsize=(18, 8.6))
        for axis, row in zip(axes.flat, rows):
            image_path = (
                args.render_root
                / method
                / f"{row['number']:02d}_{row['slug']}_three_views.png"
            )
            axis.imshow(plt.imread(image_path))
            axis.set_axis_off()
            axis.set_title(
                f"{row['number']}. {row['object_name']} | links "
                f"{row['left_links']}/{row['right_links']} | "
                f"lift {row['lift_mm']:.1f} mm | "
                f"6-dir {row['six_direction_mm']:.2f} mm",
                fontsize=11,
                fontweight="bold",
            )
        figure.suptitle(
            f"{method.upper()} — four common irregular objects, strict 3/3 rollouts",
            fontsize=19,
            fontweight="bold",
        )
        figure.subplots_adjust(
            left=0.01, right=0.99, bottom=0.02, top=0.91, wspace=0.02, hspace=0.12
        )
        output = args.render_root / f"{method}_four_objects_comparison.png"
        if output.exists():
            raise FileExistsError(output)
        figure.savefig(output, dpi=170, facecolor="white")
        plt.close(figure)
        print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
