#!/usr/bin/env python3
"""Build an auditable comparison of the two irregular-object smoke samples."""

import csv
import json
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "graph_exp/bimanual_data/4090_irregular_bleach_baseline_source4_candidates06_strict_minlinks1_v20"
BIDEX = REPO / "graph_exp/bimanual_data/4090_irregular_bleach_bidexgrasp_standoff_strict_density260_minlinks2_v9"
OUTPUT_JSON = REPO / "migration_4090/results/irregular_bleach_method_comparison_v1.json"
OUTPUT_MD = REPO / "migration_4090/IRREGULAR_METHOD_COMPARISON.md"


def load_method(name, root, sample_index, repeat_dir, render_dir, contact_dir):
    payload = torch.load(root / "bimanual_dataset.pt", map_location="cpu", weights_only=False)
    sample = payload["samples"][sample_index]
    metrics = sample["metrics"]
    with (root / repeat_dir / "repeat_results.csv").open(newline="") as handle:
        repeats = list(csv.DictReader(handle))
    contact_summary = json.loads(
        (root / contact_dir / "all_physical_contacts.json").read_text()
    )
    directions = {
        direction: float(metrics[f"displacement_{direction}_mm"])
        for direction in ("+X", "+Y", "+Z", "-X", "-Y", "-Z")
    }
    return {
        "method": name,
        "object_name": sample["object_name"],
        "left_contact_links": int(metrics["left_realized_contact_link_count"]),
        "right_contact_links": int(metrics["right_realized_contact_link_count"]),
        "left_contact_distance_mm": float(metrics["left_realized_surface_distance_mm"]),
        "right_contact_distance_mm": float(metrics["right_realized_surface_distance_mm"]),
        "left_penetration_mm": float(metrics["left_realized_penetration_mm"]),
        "right_penetration_mm": float(metrics["right_realized_penetration_mm"]),
        "hand_clearance_mm": float(metrics["realized_hand_clearance_mm"]),
        "lift_mm": float(metrics["lift_displacement_mm"]),
        "gravity_displacement_mm": float(metrics["gravity_displacement_mm"]),
        "direction_displacements_mm": directions,
        "max_direction_displacement_mm": float(metrics["max_direction_displacement_mm"]),
        "left_only_success": bool(metrics["left_only_gravity_success"]),
        "right_only_success": bool(metrics["right_only_gravity_success"]),
        "left_only_lift_mm": float(metrics["left_only_lift_displacement_mm"]),
        "right_only_lift_mm": float(metrics["right_only_lift_displacement_mm"]),
        "friction": float(payload["manifest"]["object_friction"]),
        "density_kg_m3": float(payload["manifest"]["object_density_kg_m3"]),
        "object_mass_kg": float(metrics["object_mass_kg"]),
        "finger_effort_limit_nm": float(payload["manifest"]["finger_effort_limit_nm"]),
        "repeat_count": 1 + len(repeats),
        "repeat_success_count": 1 + sum(row["passed"] == "True" for row in repeats),
        "repeat_success_rate": (1 + sum(row["passed"] == "True" for row in repeats)) / (1 + len(repeats)),
        "repeat_rows": repeats,
        "physical_contact_records": contact_summary["total_records"],
        "active_impulse_records": contact_summary["active_impulse_records"],
        "settled_physical_contacts": contact_summary["by_stage"]["settled"],
        "all_physical_contacts_csv": str((root / contact_dir / "all_physical_contacts.csv").resolve()),
        "all_physical_contacts_json": str((root / contact_dir / "all_physical_contacts.json").resolve()),
        "all_geometric_contacts_csv": str((REPO / render_dir / "contacts.csv").resolve()),
        "visualization": str((REPO / render_dir / "final_three_views.png").resolve()),
        "strict_dataset": str((root / "bimanual_dataset.pt").resolve()),
        "repeat_verified_dataset": str((root / repeat_dir / "verified_dataset.pt").resolve()),
    }


def main():
    baseline = load_method(
        "opposed_radial_baseline", BASELINE, 1, "repeat_verified_v21",
        "migration_4090/renders/baseline_bleach_source4_v20", "contact_capture_v22",
    )
    bidex = load_method(
        "bidex_region_gws_local", BIDEX, 0, "repeat_verified_v11",
        "migration_4090/renders/bidexgrasp_bleach_standoff_v9", "contact_capture_v12",
    )
    result = {
        "comparison_scope": "one strict ycb+bleach_cleanser seed per method",
        "acceptance": {
            "gravity_m_s2": 9.8,
            "contact_distance_mm": 2.0,
            "penetration_mm": 2.0,
            "hand_clearance_mm": 2.0,
            "minimum_lift_mm": 30.0,
            "maximum_six_direction_displacement_mm": 15.0,
            "single_hand_ablations_must_fail": True,
        },
        "baseline": baseline,
        "bidex": bidex,
        "provisional_verdict": (
            "BiDex-style has better contact-link redundancy (2/2 versus 1/1). "
            "Baseline has larger lift and smaller six-direction displacement, "
            "but its object is much lighter, so those raw dynamics are not a "
            "controlled head-to-head comparison. Bulk retention/repeat statistics "
            "are required for the final method ranking."
        ),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 不规则物体双手数据合成方法对比（smoke）",
        "",
        "两种方法均使用真实 Allegro 左/右手、桌面、`g=9.8`、同步闭合、lift、六向独立扰动和左右单手消融。",
        "",
        "| 指标 | 传统对置/径向 | BiDex 风格区域+局部优化 |",
        "|---|---:|---:|",
        f"| 物体 | `{baseline['object_name']}` | `{bidex['object_name']}` |",
        f"| 左/右接触 link | {baseline['left_contact_links']}/{baseline['right_contact_links']} | {bidex['left_contact_links']}/{bidex['right_contact_links']} |",
        f"| 左/右最大穿透 mm | {baseline['left_penetration_mm']:.3f}/{baseline['right_penetration_mm']:.3f} | {bidex['left_penetration_mm']:.3f}/{bidex['right_penetration_mm']:.3f} |",
        f"| 手间最小净距 mm | {baseline['hand_clearance_mm']:.3f} | {bidex['hand_clearance_mm']:.3f} |",
        f"| lift mm | {baseline['lift_mm']:.3f} | {bidex['lift_mm']:.3f} |",
        f"| 重力阶段位移 mm | {baseline['gravity_displacement_mm']:.3f} | {bidex['gravity_displacement_mm']:.3f} |",
        f"| 六向最大位移 mm | {baseline['max_direction_displacement_mm']:.3f} | {bidex['max_direction_displacement_mm']:.3f} |",
        f"| left/right-only 成功 | {baseline['left_only_success']}/{baseline['right_only_success']} | {bidex['left_only_success']}/{bidex['right_only_success']} |",
        f"| 密度 kg/m³ / 质量 kg | {baseline['density_kg_m3']:.0f} / {baseline['object_mass_kg']:.3f} | {bidex['density_kg_m3']:.0f} / {bidex['object_mass_kg']:.3f} |",
        f"| 摩擦 / effort Nm | {baseline['friction']:.1f} / {baseline['finger_effort_limit_nm']:.1f} | {bidex['friction']:.1f} / {bidex['finger_effort_limit_nm']:.1f} |",
        f"| 重复 rollout | {baseline['repeat_success_count']}/{baseline['repeat_count']} | {bidex['repeat_success_count']}/{bidex['repeat_count']} |",
        "",
        "## 初步判断",
        "",
        "BiDex 风格样本的接触冗余更好（每手 2 个 link，而传统方法每手 1 个）。传统样本 lift 更高、六向位移更低，但其物体质量显著更小，因此不能把这两个原始数值直接解释为方法更优。最终结论以 600 条/方法的严格保留率、接触冗余、重复成功率和同质量分层统计为准。",
        "",
        "## 文件",
        "",
        f"- 传统方法全部物理接触点：`{baseline['all_physical_contacts_csv']}`",
        f"- BiDex 全部物理接触点：`{bidex['all_physical_contacts_csv']}`",
        f"- 传统方法三视图：`{baseline['visualization']}`",
        f"- BiDex 三视图：`{bidex['visualization']}`",
        f"- 完整机器可读对比：`{OUTPUT_JSON.resolve()}`",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    print(OUTPUT_MD.resolve())
    print(OUTPUT_JSON.resolve())


if __name__ == "__main__":
    main()
