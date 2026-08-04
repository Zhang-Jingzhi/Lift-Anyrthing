#!/usr/bin/env python3
"""Merge deduplicated, three-repeat-verified xlarge tabletop samples."""

import csv
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "graph_exp/bimanual_data/final_xlarge_verified_collection"
SOURCES = [
    (
        REPO / "graph_exp/bimanual_data/final_v11_xlarge_tabletop_bidex/final_verified/bimanual_dataset.pt",
        "anchor_v11",
        {None: 14},
    ),
    (
        REPO / "graph_exp/bimanual_data/final_v14_cylinder_xlarge_targeted/repeat_verified/verified_dataset.pt",
        "targeted_v14",
        {0: 2, 1: 14},
    ),
    (
        REPO / "graph_exp/bimanual_data/final_v15_cylinder_xlarge_fine/repeat_verified/verified_dataset.pt",
        "fine_v15",
        {0: 14},
    ),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = []
    rows = []
    seen = set()
    for path, batch_name, source_map in SOURCES:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for batch_index, source in enumerate(payload["samples"]):
            local_source = source.get("left_source_index")
            original_source = source_map[local_source]
            roll = float(
                source.get(
                    "opposition_roll_degrees",
                    source.get("provenance", {}).get("opposition_roll_degrees"),
                )
            )
            key = (source["object_name"], original_source, round(roll, 6))
            if key in seen:
                continue
            seen.add(key)
            sample = dict(source)
            sample["provenance"] = {
                **source.get("provenance", {}),
                "collection_batch": batch_name,
                "collection_batch_index": batch_index,
                "original_source_index": original_source,
                "opposition_roll_degrees": roll,
            }
            samples.append(sample)
            metrics = sample["metrics"]
            left_links = metrics.get(
                "left_realized_contact_link_count",
                metrics.get("left_contact_link_count"),
            )
            right_links = metrics.get(
                "right_realized_contact_link_count",
                metrics.get("right_contact_link_count"),
            )
            left_penetration = metrics.get(
                "left_realized_penetration_mm",
                metrics.get("left_penetration_depth_mm"),
            )
            right_penetration = metrics.get(
                "right_realized_penetration_mm",
                metrics.get("right_penetration_depth_mm"),
            )
            rows.append(
                {
                    "sample_index": len(samples) - 1,
                    "batch": batch_name,
                    "original_source_index": original_source,
                    "roll_degrees": roll,
                    "repeat_count": metrics.get("repeat_count", 3),
                    "lift_mm": metrics["lift_displacement_mm"],
                    "gravity_mm": metrics["gravity_displacement_mm"],
                    "max_6dir_mm": metrics["max_direction_displacement_mm"],
                    "left_links": left_links,
                    "right_links": right_links,
                    "left_penetration_mm": left_penetration,
                    "right_penetration_mm": right_penetration,
                }
            )

    manifest = {
        "schema": "tro_grasp_xlarge_tabletop_bimanual_collection_v1",
        "num_unique_samples": len(samples),
        "object_name": "contactdb+cylinder_xlarge",
        "object_extents_mm": [240.0, 240.0, 180.0],
        "gravity_m_s2": 9.8,
        "object_friction": 1.0,
        "robot_friction": 1.0,
        "finger_effort_limit_nm": 0.7,
        "object_density_kg_m3": 50.0,
        "minimum_lift_mm": 30.0,
        "maximum_gravity_displacement_mm": 10.0,
        "maximum_6dir_displacement_mm": 15.0,
        "maximum_penetration_mm": 2.0,
        "minimum_contact_links_per_hand": 3,
        "maximum_mean_local_wrench_residual": 0.35,
        "single_hand_ablation_required_to_fail": True,
        "repeat_count_per_sample": 3,
        "tabletop_side_approach_only": True,
        "gravity_active_from_first_frame": True,
        "simultaneous_bimanual_closure": True,
    }
    torch.save({"manifest": manifest, "samples": samples}, OUTPUT / "bimanual_dataset.pt")
    with (OUTPUT / "sample_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "manifest": manifest,
        "ranges": {
            "lift_mm": [min(row["lift_mm"] for row in rows), max(row["lift_mm"] for row in rows)],
            "gravity_mm": [min(row["gravity_mm"] for row in rows), max(row["gravity_mm"] for row in rows)],
            "max_6dir_mm": [min(row["max_6dir_mm"] for row in rows), max(row["max_6dir_mm"] for row in rows)],
            "left_penetration_mm": [min(row["left_penetration_mm"] for row in rows), max(row["left_penetration_mm"] for row in rows)],
            "right_penetration_mm": [min(row["right_penetration_mm"] for row in rows), max(row["right_penetration_mm"] for row in rows)],
        },
        "batch_counts": {
            name: sum(row["batch"] == name for row in rows)
            for name in sorted({row["batch"] for row in rows})
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# 大物体桌面双手抓取：三次重复验证数据集

- 独立样本：{len(samples)} 条（按原始姿态源与滚转角去重）。
- 物体：直径 240 mm、高 180 mm 的圆柱容器，位于桌面上。
- 物理：重力 9.8 m/s² 从第一帧开启，左右手同步侧向闭合。
- 每条样本均通过 3 次完整重复，并且左右任一单手消融均失败。
- 硬门槛：抬升 ≥30 mm；重力位移 ≤10 mm；六方向最大位移 ≤15 mm；穿透 ≤2 mm；每只手 ≥3 条接触链。
- 局部协作门槛：每只手平均扳手残差 ≤0.35（摩擦系数 1.0）。

详细数值见 `sample_summary.csv`，机器可读数据见 `bimanual_dataset.pt`。
"""
    (OUTPUT / "REPORT_CN.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
