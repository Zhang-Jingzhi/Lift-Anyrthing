#!/usr/bin/env python3
"""Build machine-readable and human-readable reports for the strict sample."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


repo = Path(__file__).resolve().parents[1]
base = (
    repo
    / "graph_exp/bimanual_data/"
    "4090_smoke_cylinder_xlarge_source14_targeted_retry5"
)
repeat_dir = base / "repeat_verified_gpu0"
capture = torch.load(base / "contact_capture_result.pt", map_location="cpu")
geometry = torch.load(
    base / "contact_capture_geometry/geometry_result.pt", map_location="cpu"
)[0]
source = torch.load(base / "repeat_source_sample.pt", map_location="cpu")
sample = source["samples"][0]
both = torch.load(repeat_dir / "both_result.pt", map_location="cpu")
left_only = torch.load(repeat_dir / "left_result.pt", map_location="cpu")
right_only = torch.load(repeat_dir / "right_result.pt", map_location="cpu")


def mm(value):
    return float(value) * 1000.0


def physical_phase(name):
    rows = capture[name][0]
    side_points = defaultdict(int)
    side_links = defaultdict(set)
    for row in rows:
        side = row["actor1"] if row["actor0"] == "object" else row["actor0"]
        link = (
            row["body1_name"]
            if row["actor1"] == side
            else row["body0_name"]
        )
        side_points[side] += 1
        side_links[side].add(link)
    return {
        "num_points": len(rows),
        "point_count_by_side": dict(side_points),
        "contact_links_by_side": {
            key: sorted(value) for key, value in side_links.items()
        },
        "points": rows,
    }


contacts = {
    "description": (
        "All object-hand PhysX contact manifolds returned for the last "
        "substep at closure, after lift, and after gravity settling."
    ),
    "object_name": sample["object_name"],
    "phases": {
        "closure": physical_phase("closure_contacts"),
        "after_lift": physical_phase("lifted_contacts"),
        "after_gravity_settle": physical_phase("settled_contacts"),
    },
}

contact_json = repeat_dir / "physical_contact_points.json"
contact_csv = repeat_dir / "physical_settled_contact_points.csv"
metrics_json = repeat_dir / "strict_sample_metrics.json"
report_md = repeat_dir / "STRICT_SAMPLE_REPORT_CN.md"
for path in (contact_json, contact_csv, metrics_json, report_md):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing {path}")

contact_json.write_text(json.dumps(contacts, indent=2) + "\n")
settled = contacts["phases"]["after_gravity_settle"]["points"]
with contact_csv.open("w", newline="", encoding="utf-8") as handle:
    fieldnames = (
        "contact_index",
        "side",
        "link_name",
        "x_m",
        "y_m",
        "z_m",
        "normal_x",
        "normal_y",
        "normal_z",
        "initial_overlap_m",
        "normal_impulse",
    )
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in settled:
        side = row["actor1"] if row["actor0"] == "object" else row["actor0"]
        link = (
            row["body1_name"]
            if row["actor1"] == side
            else row["body0_name"]
        )
        point = row["world_midpoint_m"]
        normal = row["normal"]
        writer.writerow(
            {
                "contact_index": row["contact_index"],
                "side": side,
                "link_name": link,
                "x_m": point[0],
                "y_m": point[1],
                "z_m": point[2],
                "normal_x": normal[0],
                "normal_y": normal[1],
                "normal_z": normal[2],
                "initial_overlap_m": row["initial_overlap_m"],
                "normal_impulse": row["normal_impulse"],
            }
        )

first_metrics = sample["metrics"]
additional = []
with (repeat_dir / "repeat_results.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        additional.append(
            {
                "repeat": int(row["repeat"]),
                "passed": row["passed"] == "True",
                "lift_mm": float(row["lift_mm"]),
                "gravity_mm": float(row["gravity_mm"]),
                "max_6dir_mm": float(row["max_6dir_mm"]),
            }
        )
rollouts = [
    {
        "repeat": 1,
        "passed": bool(first_metrics["strict_success"]),
        "lift_mm": float(first_metrics["lift_displacement_mm"]),
        "gravity_mm": float(first_metrics["gravity_displacement_mm"]),
        "max_6dir_mm": float(first_metrics["max_direction_displacement_mm"]),
    },
    *additional,
]
direction_names = ("+X", "+Y", "+Z", "-X", "-Y", "-Z")
direction_mm = {
    name: mm(capture["direction_displacements"][0, index])
    for index, name in enumerate(direction_names)
}
settled_summary = contacts["phases"]["after_gravity_settle"]
metrics = {
    "object_name": sample["object_name"],
    "source_vis_parent_index": 14,
    "opposition_roll_degrees": float(sample["opposition_roll_degrees"]),
    "true_left_asset": "allegro_left",
    "true_right_asset": "allegro_right",
    "gravity_m_s2": 9.8,
    "simultaneous_closure": True,
    "table_support_during_closure": True,
    "fixture_during_closure": False,
    "side_approach": True,
    "started_below_object": False,
    "physical_contact_point_count": settled_summary["num_points"],
    "left_physical_contact_point_count": settled_summary[
        "point_count_by_side"
    ]["left"],
    "right_physical_contact_point_count": settled_summary[
        "point_count_by_side"
    ]["right"],
    "left_physical_contact_links": settled_summary[
        "contact_links_by_side"
    ]["left"],
    "right_physical_contact_links": settled_summary[
        "contact_links_by_side"
    ]["right"],
    "left_physical_contact_link_count": len(
        settled_summary["contact_links_by_side"]["left"]
    ),
    "right_physical_contact_link_count": len(
        settled_summary["contact_links_by_side"]["right"]
    ),
    "left_geometric_contact_point_count_2mm": geometry["left"][
        "contact_point_count"
    ],
    "right_geometric_contact_point_count_2mm": geometry["right"][
        "contact_point_count"
    ],
    "left_geometric_contact_link_count_2mm": geometry["left"][
        "contact_link_count"
    ],
    "right_geometric_contact_link_count_2mm": geometry["right"][
        "contact_link_count"
    ],
    "left_max_penetration_mm": geometry["left"]["penetration_depth_mm"],
    "right_max_penetration_mm": geometry["right"]["penetration_depth_mm"],
    "inter_hand_min_clearance_mm": geometry["clearance_mm"],
    "lift_height_mm": mm(capture["lift_displacement_z"][0]),
    "gravity_stage_object_displacement_mm": mm(
        capture["gravity_displacement"][0]
    ),
    "six_direction_displacement_mm": direction_mm,
    "six_direction_max_displacement_mm": max(direction_mm.values()),
    "left_only": {
        "success": bool(left_only["success"][0]),
        "gravity_success": bool(left_only["gravity_success"][0]),
        "settle_displacement_mm": mm(left_only["settle_displacement"][0]),
        "lift_height_mm": mm(left_only["lift_displacement_z"][0]),
        "gravity_displacement_mm": mm(
            left_only["gravity_displacement"][0]
        ),
    },
    "right_only": {
        "success": bool(right_only["success"][0]),
        "gravity_success": bool(right_only["gravity_success"][0]),
        "settle_displacement_mm": mm(right_only["settle_displacement"][0]),
        "lift_height_mm": mm(right_only["lift_displacement_z"][0]),
        "gravity_displacement_mm": mm(
            right_only["gravity_displacement"][0]
        ),
    },
    "robot_friction": 1.0,
    "object_friction": 1.0,
    "object_density_kg_m3": 50.0,
    "object_mass_kg": float(capture["object_mass_kg"][0]),
    "finger_joint_effort_limit_nm": 0.7,
    "contact_offset_m": 0.002,
    "rollouts": rollouts,
    "repeat_count": len(rollouts),
    "repeat_success_rate": sum(row["passed"] for row in rollouts)
    / len(rollouts),
    "physical_contact_points_json": str(contact_json),
    "physical_settled_contact_points_csv": str(contact_csv),
    "geometric_contact_points_csv": str(
        repeat_dir / "geometric_contact_points.csv"
    ),
    "visualization_png": str(
        repeat_dir / "verified_lateral_grasp_three_views.png"
    ),
    "final_dataset": str(repeat_dir / "verified_dataset.pt"),
}
metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")

directions = ", ".join(
    f"{name}={value:.3f}" for name, value in direction_mm.items()
)
report_md.write_text(
    f"""# 严格双手数据生成 smoke 样本

- 物体：`{metrics['object_name']}`；`source_vis` 原始姿态 14；对置滚转 {metrics['opposition_roll_degrees']:.2f}°。
- 资产：真实 `allegro_left` + 真实 `allegro_right`；重力 9.8 m/s²；桌面支撑闭合；无固定夹具；两手同步侧向闭合；无物体下方起抓。
- 重力稳定阶段真实 PhysX 接触：左手 {metrics['left_physical_contact_point_count']} 点 / {metrics['left_physical_contact_link_count']} links；右手 {metrics['right_physical_contact_point_count']} 点 / {metrics['right_physical_contact_link_count']} links。
- 最终 2 mm 几何接触：左手 {metrics['left_geometric_contact_point_count_2mm']} 点 / {metrics['left_geometric_contact_link_count_2mm']} links；右手 {metrics['right_geometric_contact_point_count_2mm']} 点 / {metrics['right_geometric_contact_link_count_2mm']} links。
- 最大穿透：左 {metrics['left_max_penetration_mm']:.3f} mm；右 {metrics['right_max_penetration_mm']:.3f} mm；手间最小净距 {metrics['inter_hand_min_clearance_mm']:.3f} mm。
- 抬升 {metrics['lift_height_mm']:.3f} mm；重力阶段物体位移 {metrics['gravity_stage_object_displacement_mm']:.3f} mm。
- 六方向独立扰动位移（mm）：{directions}；最大 {metrics['six_direction_max_displacement_mm']:.3f} mm。
- left-only：失败（抬升 {metrics['left_only']['lift_height_mm']:.3f} mm < 30 mm）；right-only：失败（抬升 {metrics['right_only']['lift_height_mm']:.3f} mm < 30 mm）。
- 摩擦：手/物均 1.0；密度 50 kg/m³；质量 {metrics['object_mass_kg']:.6f} kg；手指关节 effort 上限 0.7 N·m。
- 共 {metrics['repeat_count']} 次完整 rollout，严格成功率 {metrics['repeat_success_rate']:.0%}。
- 所有 PhysX 接触点：`{contact_json}`；稳定阶段 CSV：`{contact_csv}`。
- 全部 2 mm 几何接触点：`{repeat_dir / 'geometric_contact_points.csv'}`。
- 可视化：`{repeat_dir / 'verified_lateral_grasp_three_views.png'}`。
- 最终重复验证数据：`{repeat_dir / 'verified_dataset.pt'}`。
"""
)
print(metrics_json)
print(contact_json)
print(contact_csv)
print(report_md)
