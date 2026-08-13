#!/usr/bin/env python3
"""Re-screen legacy formal samples with actual PhysX q and mesh penetration."""

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from xhand_visual_mesh_gate import (
    evaluate_visual_mesh_penetration,
    materialize_actual_closure_state,
)


OBJECTS = ("sphere", "cube", "cracker", "bleach", "pitcher", "drill")
METHODS = ("baseline", "bidex_v3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("graph_exp/bimanual_data/xhand_compact_formal_1200_v1"),
    )
    parser.add_argument("--max-depth-mm", type=float, default=2.0)
    parser.add_argument("--max-inside-fraction", type=float, default=0.01)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    migration = args.root / f"legacy_target_q_v1_{stamp}"
    migration.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "xhand_formal_actual_pose_migration_v2",
        "root": str(args.root.resolve()),
        "legacy_archive": str(migration.resolve()),
        "max_depth_mm": args.max_depth_mm,
        "max_inside_fraction": args.max_inside_fraction,
        "rows": [],
    }

    for method in METHODS:
        for obj in OBJECTS:
            accepted = args.root / method / obj / "accepted"
            if not accepted.is_dir():
                continue
            manifests = sorted(accepted.glob("sample_*.json"))
            if not manifests:
                continue
            legacy = migration / method / obj / "accepted"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            accepted.rename(legacy)
            new_accepted = args.root / method / obj / "accepted"
            new_accepted.mkdir(parents=True, exist_ok=False)
            passed = failed = 0
            for old_manifest_path in manifests:
                old_manifest = json.loads((legacy / old_manifest_path.name).read_text())
                source_dataset = Path(old_manifest["source_dataset"])
                source_index = int(old_manifest["source_sample_index"])
                report_paths = [Path(p) for p in old_manifest["strict_repeat_reports"]]
                sample_path = legacy / Path(old_manifest["sample_path"]).name
                sample = torch.load(sample_path, map_location="cpu", weights_only=False)
                strict = json.loads(report_paths[0].read_text())
                actual = materialize_actual_closure_state(sample, strict)
                gate = evaluate_visual_mesh_penetration(
                    actual,
                    max_depth_mm=args.max_depth_mm,
                    max_inside_fraction=args.max_inside_fraction,
                )
                candidate_dir = source_dataset.parent / f"candidate_{source_index:03d}"
                old_marker = candidate_dir / "accepted.done"
                if old_marker.exists():
                    old_marker.rename(candidate_dir / "accepted_target_q_legacy.done")
                row = {
                    "object": obj,
                    "method": method,
                    "legacy_manifest": str((legacy / old_manifest_path.name).resolve()),
                    "source_dataset": str(source_dataset.resolve()),
                    "source_sample_index": source_index,
                    "visual_mesh_gate": gate,
                }
                if gate["pass"]:
                    ordinal = passed
                    stem = f"sample_{ordinal:04d}"
                    out_sample = new_accepted / f"{stem}.pt"
                    out_manifest = new_accepted / f"{stem}.json"
                    actual["formal_validation"] = {
                        "schema": "xhand_compact_formal_validation_v2_actual_pose",
                        "strict_repeat_count": 3,
                        "strict_success_rate": 1.0,
                        "report_paths": [str(p.resolve()) for p in report_paths],
                        "left_only_physical_pass": False,
                        "right_only_physical_pass": False,
                        "lift_height_m": strict["runs"]["both"]["lift_object_displacement_m"],
                        "gravity_displacement_m": strict["runs"]["both"]["gravity_displacement_m"],
                        "six_direction_displacements_m": strict["runs"]["both"]["six_direction_displacements_m"],
                        "penetration_pass": bool(strict["runs"]["both"]["penetration_pass"]),
                        "visual_mesh_gate": gate,
                    }
                    actual["isaac_gym_physical_validated"] = True
                    actual["isaaclab_physical_validated"] = False
                    manifest = copy.deepcopy(old_manifest)
                    manifest.update({
                        "schema": "xhand_compact_formal_success_v2_actual_pose",
                        "sample_path": str(out_sample.resolve()),
                        "visual_mesh_gate": gate,
                        "max_commanded_actual_dof_error_rad": strict["runs"]["both"]["max_dof_position_error_after_closure"],
                    })
                    torch.save(actual, out_sample)
                    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
                    (candidate_dir / "accepted.done").touch()
                    row["status"] = "accepted_actual_pose"
                    row["new_manifest"] = str(out_manifest.resolve())
                    passed += 1
                else:
                    rejection = candidate_dir / "visual_mesh_rejection.json"
                    rejection.write_text(json.dumps({**row, "reason": "visual_mesh_penetration"}, indent=2) + "\n")
                    (candidate_dir / "mesh_rejected.done").touch()
                    row["status"] = "rejected_visual_mesh"
                    failed += 1
                report["rows"].append(row)
            report.setdefault("combos", []).append({
                "object": obj,
                "method": method,
                "legacy_count": len(manifests),
                "accepted_actual_pose": passed,
                "rejected_visual_mesh": failed,
            })

    out = migration / "migration_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out.resolve()), "combos": report.get("combos", [])}, indent=2))


if __name__ == "__main__":
    main()
