#!/usr/bin/env python3
"""Persist one independently Isaac-validated formal XHand grasp."""

import argparse
import json
import os
from pathlib import Path

import torch

from xhand_visual_mesh_gate import (
    evaluate_visual_mesh_penetration,
    materialize_actual_closure_state,
)


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--method", choices=("baseline", "bidex_v3"), required=True)
    parser.add_argument("--size-index", type=int, required=True)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--rejection-output", type=Path, default=None)
    args = parser.parse_args()

    reports = [load(path) for path in args.reports]
    if not all(report.get("physical_pass") is True for report in reports):
        raise RuntimeError("refusing to save a candidate without 3/3 strict passes")
    for report in reports:
        runs = report.get("runs", {})
        if not (
            runs.get("both", {}).get("physical_pass") is True
            and runs.get("left", {}).get("physical_pass") is False
            and runs.get("right", {}).get("physical_pass") is False
        ):
            raise RuntimeError("strict report does not satisfy both-hand ablation gate")

    args.accepted_root.mkdir(parents=True, exist_ok=True)
    source_identity = (str(args.dataset.resolve()), args.sample_index)
    existing = sorted(args.accepted_root.glob("sample_*.json"))
    for path in existing:
        payload = load(path)
        identity = (payload.get("source_dataset"), payload.get("source_sample_index"))
        if identity == source_identity:
            print(json.dumps({"status": "already_saved", "manifest": str(path)}))
            return

    payload = torch.load(args.dataset, map_location="cpu", weights_only=False)
    sample = payload["samples"][args.sample_index]
    both = reports[0]["runs"]["both"]
    sample = materialize_actual_closure_state(sample, reports[0])
    visual_mesh_gate = evaluate_visual_mesh_penetration(sample)
    if not visual_mesh_gate["pass"]:
        rejection = {
            "schema": "xhand_compact_formal_rejection_v1",
            "reason": "visual_mesh_penetration",
            "object": args.object,
            "method": args.method,
            "size_index": args.size_index,
            "source_dataset": str(args.dataset.resolve()),
            "source_sample_index": args.sample_index,
            "strict_repeat_reports": [str(path.resolve()) for path in args.reports],
            "visual_mesh_gate": visual_mesh_gate,
        }
        if args.rejection_output is not None:
            args.rejection_output.parent.mkdir(parents=True, exist_ok=True)
            args.rejection_output.write_text(json.dumps(rejection, indent=2) + "\n")
        print(json.dumps(rejection, indent=2))
        raise SystemExit(4)
    sample["method"] = args.method
    sample["isaac_gym_physical_validated"] = True
    sample["isaaclab_physical_validated"] = False
    sample["formal_validation"] = {
        "schema": "xhand_compact_formal_validation_v1",
        "strict_repeat_count": 3,
        "strict_success_rate": 1.0,
        "report_paths": [str(path.resolve()) for path in args.reports],
        "left_only_physical_pass": False,
        "right_only_physical_pass": False,
        "lift_height_m": both["lift_object_displacement_m"],
        "gravity_displacement_m": both["gravity_displacement_m"],
        "six_direction_displacements_m": both["six_direction_displacements_m"],
        "penetration_pass": bool(both["penetration_pass"]),
        "visual_mesh_gate": visual_mesh_gate,
        "left_closure_contact_count": both["closure"]["left_contact_count"],
        "right_closure_contact_count": both["closure"]["right_contact_count"],
        "left_lift_contact_count": both["lifted"]["left_contact_count"],
        "right_lift_contact_count": both["lifted"]["right_contact_count"],
    }

    ordinal = len(existing)
    stem = f"sample_{ordinal:04d}"
    sample_path = args.accepted_root / f"{stem}.pt"
    manifest_path = args.accepted_root / f"{stem}.json"
    if sample_path.exists() or manifest_path.exists():
        raise FileExistsError(f"non-contiguous accepted output at {stem}")

    manifest = {
        "schema": "xhand_compact_formal_success_v1",
        "object": args.object,
        "method": args.method,
        "size_index": args.size_index,
        "source_dataset": source_identity[0],
        "source_sample_index": args.sample_index,
        "sample_path": str(sample_path.resolve()),
        "strict_repeat_reports": [str(path.resolve()) for path in args.reports],
        "strict_success_rate": 1.0,
        "left_only_physical_pass": False,
        "right_only_physical_pass": False,
        "lift_height_mm": both["lift_object_displacement_m"] * 1000.0,
        "gravity_displacement_mm": both["gravity_displacement_m"] * 1000.0,
        "six_direction_max_mm": max(both["six_direction_displacements_m"]) * 1000.0,
        "left_closure_contact_count": both["closure"]["left_contact_count"],
        "right_closure_contact_count": both["closure"]["right_contact_count"],
        "left_lift_contact_count": both["lifted"]["left_contact_count"],
        "right_lift_contact_count": both["lifted"]["right_contact_count"],
        "penetration_pass": bool(both["penetration_pass"]),
        "visual_mesh_gate": visual_mesh_gate,
        "max_commanded_actual_dof_error_rad": float(
            both["max_dof_position_error_after_closure"]
        ),
    }

    tmp_sample = sample_path.with_suffix(".pt.tmp")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    torch.save(sample, tmp_sample)
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp_sample, sample_path)
    os.replace(tmp_manifest, manifest_path)
    print(json.dumps({"status": "saved", "manifest": str(manifest_path), "sample": str(sample_path)}))


if __name__ == "__main__":
    main()
