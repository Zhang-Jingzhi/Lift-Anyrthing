#!/usr/bin/env python3
"""Select auditable, varied examples for baseline/BiDex visualization."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "migration_4090"))

from select_bulk_repeat_candidates import pose_key, rank_key


def load_unique(root):
    unique = {}
    for path in sorted(root.glob("seed_*/repeat_verified/verified_dataset.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for index, sample in enumerate(payload["samples"]):
            key = pose_key(sample)
            row = {
                "sample": sample,
                "source": path,
                "source_index": index,
                "manifest": payload.get("manifest", {}),
            }
            previous = unique.get(key)
            if previous is None or rank_key(sample) < rank_key(previous["sample"]):
                unique[key] = row
    return unique


def metric(row, name):
    return float(row["sample"]["metrics"][name])


def choose_nearest_unused(rows, ordered_indices, used):
    return next(index for index in ordered_indices if index not in used)


def select_rows(rows):
    used = set()
    selected = []

    specs = [
        (
            "most_stable",
            "Most stable",
            sorted(range(len(rows)), key=lambda i: metric(rows[i], "repeat_max_6dir_mm")),
        ),
        (
            "typical",
            "Typical stability",
            sorted(
                range(len(rows)),
                key=lambda i: abs(
                    metric(rows[i], "repeat_max_6dir_mm")
                    - float(
                        np.median(
                            [metric(row, "repeat_max_6dir_mm") for row in rows]
                        )
                    )
                ),
            ),
        ),
        (
            "most_contact",
            "Most bilateral contact",
            sorted(
                range(len(rows)),
                key=lambda i: (
                    -min(
                        metric(rows[i], "repeat_min_left_contact_links"),
                        metric(rows[i], "repeat_min_right_contact_links"),
                    ),
                    -(
                        metric(rows[i], "repeat_min_left_contact_links")
                        + metric(rows[i], "repeat_min_right_contact_links")
                    ),
                ),
            ),
        ),
        (
            "highest_lift",
            "Highest minimum lift",
            sorted(
                range(len(rows)),
                key=lambda i: -metric(rows[i], "repeat_min_lift_mm"),
            ),
        ),
        (
            "widest_clearance",
            "Widest hand clearance",
            sorted(
                range(len(rows)),
                key=lambda i: -metric(rows[i], "repeat_min_hand_clearance_mm"),
            ),
        ),
    ]
    for category, label, order in specs:
        index = choose_nearest_unused(rows, order, used)
        used.add(index)
        selected.append((category, label, rows[index]))

    # Pick the remaining pose farthest from all five selections in a
    # dimension-normalized paired-pose space. This is a QC example, not a
    # replacement for the final dataset's quality ranking.
    features = np.stack(
        [
            torch.cat(
                [row["sample"]["left_q_seed"], row["sample"]["right_q_seed"]]
            ).numpy()
            for row in rows
        ]
    )
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - features.mean(axis=0)) / scale
    selected_indices = list(used)
    distances = np.linalg.norm(
        normalized[:, None, :] - normalized[selected_indices][None, :, :], axis=2
    ).min(axis=1)
    distances[list(used)] = -np.inf
    diverse_index = int(np.argmax(distances))
    selected.append(("diverse_pose", "Most different pose", rows[diverse_index]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--bidex-root", type=Path, required=True)
    parser.add_argument("--bidex-final", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    dataset_dir = args.output_dir / "selection_datasets"
    dataset_dir.mkdir()

    baseline = load_unique(args.baseline_root)
    bidex_all = load_unique(args.bidex_root)
    final_payload = torch.load(args.bidex_final, map_location="cpu", weights_only=False)
    final_keys = {pose_key(sample) for sample in final_payload["samples"]}
    bidex = {key: row for key, row in bidex_all.items() if key in final_keys}
    if len(baseline) < 6 or len(bidex) < 6:
        raise RuntimeError(
            f"Need at least six samples per method: baseline={len(baseline)}, "
            f"bidex={len(bidex)}"
        )

    report = {"schema": "tro_grasp_method_visual_examples_v1", "methods": {}}
    for method, lookup in (("baseline", baseline), ("bidex", bidex)):
        rows = list(lookup.values())
        selections = select_rows(rows)
        method_rows = []
        for number, (category, label, row) in enumerate(selections, start=1):
            output = dataset_dir / f"{method}_{number:02d}_{category}.pt"
            manifest = dict(row["manifest"])
            manifest.update(
                {
                    "visual_selection_method": method,
                    "visual_selection_category": category,
                    "visual_selection_label": label,
                    "visual_selection_source": str(row["source"].resolve()),
                    "visual_selection_source_index": row["source_index"],
                }
            )
            torch.save(
                {
                    "version": "method_visual_examples_v1",
                    "manifest": manifest,
                    "samples": [row["sample"]],
                },
                output,
            )
            metrics = row["sample"]["metrics"]
            method_rows.append(
                {
                    "number": number,
                    "category": category,
                    "label": label,
                    "dataset": str(output.resolve()),
                    "source": str(row["source"].resolve()),
                    "source_index": row["source_index"],
                    "left_links": int(metrics["repeat_min_left_contact_links"]),
                    "right_links": int(metrics["repeat_min_right_contact_links"]),
                    "penetration_mm": float(
                        max(
                            metrics["repeat_max_left_penetration_mm"],
                            metrics["repeat_max_right_penetration_mm"],
                        )
                    ),
                    "clearance_mm": float(metrics["repeat_min_hand_clearance_mm"]),
                    "lift_mm": float(metrics["repeat_min_lift_mm"]),
                    "gravity_mm": float(metrics["repeat_max_gravity_mm"]),
                    "six_direction_mm": float(metrics["repeat_max_6dir_mm"]),
                }
            )
        report["methods"][method] = method_rows
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
