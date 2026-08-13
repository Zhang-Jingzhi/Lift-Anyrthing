#!/usr/bin/env python3
"""Aggregate the final 12-way strict XHand smoke results across audit roots."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "migration_4090/results"
REPORT_DIR = ROOT / "migration_4090/reports"
OBJECTS = ("sphere", "cube", "cracker", "bleach", "pitcher", "drill")
METHODS = ("baseline", "bidex_v3")


def find_summaries():
    found = {}
    for path in RESULTS.glob("xhand_compact_6x100_*/physics/*/*/seed*/smoke_success.json"):
        parts = path.parts
        physics_index = parts.index("physics")
        key = (parts[physics_index + 1], parts[physics_index + 2])
        found.setdefault(key, []).append(path)
    return {key: sorted(paths)[-1] for key, paths in found.items()}


def load_sample(dataset, index):
    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    return payload["samples"][index]


def main():
    summaries = find_summaries()
    rows = []
    for method in METHODS:
        for obj in OBJECTS:
            key = (obj, method)
            path = summaries.get(key)
            if path is None:
                rows.append({"object": obj, "method": method, "strict_success": False})
                continue
            summary = json.loads(path.read_text())
            strict_path = Path(summary["repeat_reports"][0])
            strict = json.loads(strict_path.read_text())
            both = strict["runs"]["both"]
            sample = load_sample(summary["dataset"], summary["sample_index"])
            transform = sample.get("object_transform", {})
            physical = sample.get("physical_parameters", {})
            rows.append({
                "object": obj,
                "method": method,
                "strict_success": True,
                "size_index": transform.get("size_index"),
                "object_name": sample.get("object_name"),
                "mesh_path": sample.get("object_mesh_path"),
                "dataset": summary["dataset"],
                "sample_index": summary["sample_index"],
                "summary_path": str(path),
                "repeat_reports": summary["repeat_reports"],
                "repeat_success_rate": summary["repeat_success_rate"],
                "left_only_physical_pass": summary["left_only_physical_pass"],
                "right_only_physical_pass": summary["right_only_physical_pass"],
                "left_closure_contact_count": both["closure"]["left_contact_count"],
                "right_closure_contact_count": both["closure"]["right_contact_count"],
                "left_lift_contact_count": both["lifted"]["left_contact_count"],
                "right_lift_contact_count": both["lifted"]["right_contact_count"],
                "left_settled_contact_count": both["settled"]["left_contact_count"],
                "right_settled_contact_count": both["settled"]["right_contact_count"],
                "left_contact_links": both["settled"]["left_contact_links"],
                "right_contact_links": both["settled"]["right_contact_links"],
                "left_max_penetration_mm": max(
                    both[phase]["left_max_penetration_mm"]
                    for phase in ("closure", "lifted", "settled")
                ),
                "right_max_penetration_mm": max(
                    both[phase]["right_max_penetration_mm"]
                    for phase in ("closure", "lifted", "settled")
                ),
                "lift_height_mm": both["lift_object_displacement_m"] * 1000.0,
                "gravity_displacement_mm": both["gravity_displacement_m"] * 1000.0,
                "six_direction_displacements_mm": [
                    value * 1000.0 for value in both["six_direction_displacements_m"]
                ],
                "six_direction_max_mm": max(both["six_direction_displacements_m"]) * 1000.0,
                "target_mass_kg": physical.get("target_mass_kg"),
                "actual_mass_kg": both.get("object_mass_kg"),
                "density_kg_m3": both.get("object_density_kg_m3"),
                "friction": both.get("object_friction"),
                "gravity_m_s2": both.get("gravity_m_s2"),
                "effort_limits": both.get("effort_limits"),
                "controller": both.get("controller"),
                "isaac_gpu": both.get("isaac_gpu"),
            })

    payload = {
        "schema": "xhand_compact_final_12way_strict_smoke_v1",
        "strict_success_count": sum(row["strict_success"] for row in rows),
        "target_count": 12,
        "rows": rows,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "xhand_compact_12way_smoke_final_20260810.json"
    md_path = REPORT_DIR / "xhand_compact_12way_smoke_final_20260810.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Tianji + dual-XHand final 12-way strict smoke report",
        "",
        f"Strict success: {payload['strict_success_count']}/12",
        "",
        "| Method | Object | Size | Contacts C(L/R) | Lift mm | Gravity mm | Six-dir max mm | Repeat | Ablations |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row["strict_success"]:
            lines.append(f"| {row['method']} | {row['object']} | - | - | - | - | - | failed | - |")
            continue
        lines.append(
            f"| {row['method']} | {row['object']} | {row['size_index']} | "
            f"{row['left_closure_contact_count']}/{row['right_closure_contact_count']} | "
            f"{row['lift_height_mm']:.3f} | {row['gravity_displacement_mm']:.3f} | "
            f"{row['six_direction_max_mm']:.3f} | {row['repeat_success_rate']:.0%} | "
            f"L={row['left_only_physical_pass']}, R={row['right_only_physical_pass']} |"
        )
    lines += ["", f"Machine-readable report: `{json_path}`", ""]
    md_path.write_text("\n".join(lines))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), **payload}, indent=2))


if __name__ == "__main__":
    main()
