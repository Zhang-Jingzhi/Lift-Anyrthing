#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "migration_4090/results/xhand_fullbody_grasps_v1_batch_v2"
OUT = ROOT / "migration_4090/results/xhand_fullbody_grasps_v1_final"
SCHEDULE = ROOT / "migration_4090/results/large_random_6x8_v1/shared_schedule_6x100.json"
OUT.mkdir(parents=True, exist_ok=True)

chunks = [(0, 75), (75, 150), (150, 225), (225, 300), (300, 375), (375, 450), (450, 525), (525, 600)]
sources = {
    "baseline": [SRC / f"chunk_{i:02d}/baseline__{a:04d}_{b:04d}.pt" for i, (a, b) in enumerate(chunks)],
    "bidex_v3": [SRC / f"chunk_{i:02d}/bidex_v3__{a:04d}_{b:04d}.pt" for i, (a, b) in enumerate(chunks)],
}
sources["bidex_v3"][4] = SRC / "chunk_04_bidex_repair/bidex_v3__0300_0375.pt"

schedule = json.loads(SCHEDULE.read_text())["entries"]
report = {"schema": "xhand_fullbody_grasp_pose_final_audit_v1", "methods": {}, "errors": []}

for method, paths in sources.items():
    samples = []
    for chunk_index, path in enumerate(paths):
        if not path.is_file():
            report["errors"].append(f"missing {path}")
            continue
        payload = torch.load(path, weights_only=False)
        chunk_samples = payload["samples"]
        start = chunks[chunk_index][0]
        for local_index, sample in enumerate(chunk_samples):
            # Older chunk files used a local index; normalize it to the
            # shared 0..599 schedule while preserving all pose data.
            sample["schedule_index"] = start + local_index
        samples.extend(chunk_samples)
    indices = [int(s.get("schedule_index", -1)) for s in samples]
    if len(samples) != 600:
        report["errors"].append(f"{method}: count={len(samples)}")
    if sorted(indices) != list(range(600)):
        report["errors"].append(f"{method}: schedule index set invalid")
    metrics = []
    max_ik = 0.0
    fixed_max = 0.0
    for sample in samples:
        q = np.asarray(sample["full_body_q"], dtype=np.float64)
        names = sample["joint_names"]
        max_ik = max(max_ik, float(max(sample["ik_position_error_m"])))
        fixed_vals = [abs(float(q[names.index(n)])) for n in ("waist_extend2", "waist_yaw", "waist_pitch")]
        fixed_max = max(fixed_max, max(fixed_vals))
        m = sample["metrics"]
        metrics.append(m)
        if len(q) != 51 or not np.isfinite(q).all():
            report["errors"].append(f"{method} index {sample.get('schedule_index')}: invalid q")
        if max(sample["ik_position_error_m"]) > 0.002 or fixed_vals and max(fixed_vals) > 1e-6:
            report["errors"].append(f"{method} index {sample.get('schedule_index')}: IK/fixed-waist gate")
        if not (m["left_contact_points"] > 0 and m["right_contact_points"] > 0 and m["left_penetration_mm"] <= 2.0 and m["right_penetration_mm"] <= 2.0 and m["hand_clearance_mm"] > 2.0):
            report["errors"].append(f"{method} index {sample.get('schedule_index')}: geometry gate")
    report["methods"][method] = {
        "count": len(samples), "max_ik_error_m": max_ik, "max_fixed_waist_abs_rad": fixed_max,
        "min_left_contact_points": min((m["left_contact_points"] for m in metrics), default=0),
        "min_right_contact_points": min((m["right_contact_points"] for m in metrics), default=0),
        "max_left_penetration_mm": max((m["left_penetration_mm"] for m in metrics), default=0.0),
        "max_right_penetration_mm": max((m["right_penetration_mm"] for m in metrics), default=0.0),
        "min_hand_clearance_mm": min((m["hand_clearance_mm"] for m in metrics), default=0.0),
        "physical_validation": "pending_isaaclab",
    }
    torch.save({"schema": "xhand_fullbody_grasp_pose_v1", "method": method, "samples": sorted(samples, key=lambda s: s["schedule_index"])}, OUT / f"{method}.pt")

report["total"] = sum(v["count"] for v in report["methods"].values())
report["expected_total"] = 1200
report["pass"] = not report["errors"] and report["total"] == 1200
(OUT / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if report["errors"]:
    raise SystemExit(1)
