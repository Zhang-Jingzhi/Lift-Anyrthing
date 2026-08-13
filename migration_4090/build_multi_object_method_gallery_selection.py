#!/usr/bin/env python3
"""Build one strict representative per common object and synthesis method."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "migration_4090"))

from select_bulk_repeat_candidates import pose_key, rank_key


OBJECTS = (
    ("ycb+bleach_cleanser", "bleach_cleanser"),
    ("ycb+pitcher_base", "pitcher_base"),
    ("ycb+toy_airplane", "toy_airplane"),
    ("ycb+cracker_box", "cracker_box"),
)


def load_rows(paths, expected_object):
    unique = {}
    for path in sorted(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for index, sample in enumerate(payload.get("samples", [])):
            if sample.get("object_name") != expected_object:
                continue
            metrics = sample.get("metrics", {})
            if not (
                metrics.get("strict_success")
                and metrics.get("repeat_count") == 3
                and metrics.get("repeat_success_rate") == 1.0
                and metrics.get("bimanual_required")
                and metrics.get("repeat_bilateral_physical_contact_all_phases")
            ):
                continue
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
    return list(unique.values())


def sources(repo, visual_root, method, slug):
    if method == "baseline" and slug == "cracker_box":
        return [visual_root / "validated/baseline_cracker/verified_dataset.pt"]
    if method == "bidex" and slug == "pitcher_base":
        return [visual_root / "validated/bidex_pitcher/verified_dataset.pt"]
    if method == "bidex" and slug == "cracker_box":
        return [visual_root / "validated/bidex_cracker/verified_dataset.pt"]
    root = (
        repo
        / "graph_exp/bimanual_data"
        / f"bulk_irregular_{method}_v1"
        / f"ycb_{slug}"
    )
    return sorted(root.glob("seed_*/repeat_verified/verified_dataset.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    visual_root = args.visual_root.resolve()
    selection_dir = visual_root / "gallery_selection"
    if selection_dir.exists():
        raise FileExistsError(selection_dir)
    selection_dir.mkdir()

    report = {"schema": "tro_grasp_multi_object_method_gallery_v1", "methods": {}}
    for method in ("baseline", "bidex"):
        method_rows = []
        for number, (object_name, slug) in enumerate(OBJECTS, start=1):
            candidates = load_rows(
                sources(repo, visual_root, method, slug), object_name
            )
            if not candidates:
                raise RuntimeError(f"No strict sample for {method} {object_name}")
            row = min(candidates, key=lambda item: rank_key(item["sample"]))
            output = selection_dir / f"{method}_{number:02d}_{slug}.pt"
            manifest = dict(row["manifest"])
            manifest.update(
                {
                    "visual_selection_method": method,
                    "visual_selection_label": slug,
                    "visual_selection_source": str(row["source"].resolve()),
                    "visual_selection_source_index": row["source_index"],
                }
            )
            torch.save(
                {
                    "version": "multi_object_method_gallery_v1",
                    "manifest": manifest,
                    "samples": [row["sample"]],
                },
                output,
            )
            metrics = row["sample"]["metrics"]
            method_rows.append(
                {
                    "number": number,
                    "object_name": object_name,
                    "slug": slug,
                    "dataset": str(output.resolve()),
                    "source": str(row["source"].resolve()),
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
    (visual_root / "gallery_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
