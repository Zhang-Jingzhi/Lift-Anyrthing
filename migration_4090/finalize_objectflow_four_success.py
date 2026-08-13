#!/usr/bin/env python3
"""Finalize the four ObjectFlow XHand strict-success samples.

This script never edits source candidates or validation reports. It copies the
selected artifacts into one delivery directory, validates every strict gate,
and creates a presentation-ready four-object summary image.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "migration_4090/objectflow_four_success_v1/final"
URDF = Path(
    "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/"
    "xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"
)

ITEMS = {
    "episode_000002": {
        "candidate": 4,
        "pose": ROOT / "migration_4090/objectflow_four_success_v1/materialized/episode_000002/4/sample.pt",
        "strict": ROOT / "migration_4090/objectflow_four_success_v1/strict/episode_000002_candidate4_full.json",
        "raw": ROOT / "migration_4090/objectflow_four_success_v1/reports/episode_000002_raw_mesh.json",
        "log": ROOT / "migration_4090/objectflow_four_success_v1/logs/strict_episode_000002_candidate4_full.log",
    },
    "episode_000003": {
        "candidate": 5,
        "pose": ROOT / "migration_4090/objectflow_four_success_v1/materialized/episode_000003/5/sample.pt",
        "strict": ROOT / "migration_4090/objectflow_four_success_v1/strict/episode_000003.json",
        "raw": ROOT / "migration_4090/objectflow_four_success_v1/reports/episode_000003_raw_mesh.json",
        "log": ROOT / "migration_4090/objectflow_four_success_v1/logs/strict_episode_000003.log",
    },
    "episode_000233": {
        "candidate": 5,
        "pose": ROOT / "migration_4090/objectflow_four_success_v1/materialized/episode_000233/5/sample.pt",
        "strict": ROOT / "migration_4090/objectflow_four_success_v1/strict/episode_000233.json",
        "raw": ROOT / "migration_4090/objectflow_four_success_v1/reports/episode_000233_raw_mesh.json",
        "log": ROOT / "migration_4090/objectflow_four_success_v1/logs/strict_episode_000233.log",
    },
    "episode_000234": {
        "candidate": "posture_ik",
        "pose": ROOT / "migration_4090/objectflow_pose_preview/palm_inward_smoke_v1/isaac_postureik.pt",
        "strict": ROOT / "migration_4090/objectflow_pose_preview/palm_inward_smoke_v1/isaac_run_v3/report_postureik.json",
        "raw": ROOT / "migration_4090/objectflow_four_success_v1/reports/episode_000234_raw_mesh.json",
        "log": ROOT / "migration_4090/logs/palm_inward_smoke_v1/postureik.log",
    },
}

SIMULATION = {
    "backend": "Isaac Gym Preview 4 GPU PhysX",
    "gpu": 0,
    "gravity_m_s2": 9.8,
    "dt_s": 0.002,
    "substeps": 4,
    "physx_position_iterations": 16,
    "physx_velocity_iterations": 4,
    "object_collision": "VHACD",
    "vhacd_resolution": 1_000_000,
    "vhacd_max_convex_hulls": 128,
    "vhacd_max_vertices_per_hull": 64,
    "object_density_kg_m3": 50.0,
    "friction": 2.0,
    "hand_stiffness": 50.0,
    "hand_damping": 2.0,
    "hand_effort": "native URDF limits (0.4 or 1.1 per hand joint)",
    "closure_steps": 600,
    "closure_settle_steps": 300,
    "lift_height_command_mm": 50.0,
    "lift_steps": 100,
    "gravity_steps": 500,
    "disturbance_steps_per_direction": 100,
    "robot_root_alignment_offset_xyz_m": [0.0, 0.0, 0.0],
    "robot_collision_filter": 1,
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def assert_strict(report: dict) -> None:
    assert report["physical_pass"] is True
    both = report["runs"]["both"]
    assert both["physical_pass"] is True
    assert report["runs"]["left"]["physical_pass"] is False
    assert report["runs"]["right"]["physical_pass"] is False
    for phase in ("closure", "lifted"):
        assert both[phase]["left_contact_count"] > 0
        assert both[phase]["right_contact_count"] > 0
    assert both["lift_object_displacement_m"] >= 0.03
    assert both["gravity_displacement_m"] <= 0.01
    assert max(both["six_direction_displacements_m"]) <= 0.015
    assert both["penetration_pass"] is True
    assert both["max_allowed_penetration_mm"] == 2.0


def copy_new(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite {dst}")
    shutil.copy2(src, dst)


def main() -> None:
    rows = []
    for episode, item in ITEMS.items():
        for key in ("pose", "strict", "raw", "log"):
            assert item[key].is_file(), item[key]
        strict = json.loads(item["strict"].read_text())
        raw_payload = json.loads(item["raw"].read_text())
        assert_strict(strict)
        raw = raw_payload["rows"][0]
        assert raw["query_valid"] is True
        assert raw["left_contact_points"] > 0 and raw["right_contact_points"] > 0
        pose_payload = torch.load(item["pose"], map_location="cpu", weights_only=False)
        assert len(pose_payload["samples"]) == 1
        sample = pose_payload["samples"][0]
        both = strict["runs"]["both"]
        assert both["isaac_robot_root_alignment_offset_xyz"] == [0.0, 0.0, 0.0]
        assert both["robot_collision_filter"] == 1
        assert both["gravity_m_s2"] == 9.8
        assert both["object_density_kg_m3"] == 50.0
        assert both["object_friction"] == 2.0
        episode_dir = OUT / episode
        episode_dir.mkdir(parents=True, exist_ok=True)
        copies = {
            "strict_success.pt": item["pose"],
            "strict_report.json": item["strict"],
            "raw_mesh_report.json": item["raw"],
            "strict_validation.log": item["log"],
            "object_mesh.obj": Path(sample["object_mesh_path"]),
        }
        for name, src in copies.items():
            copy_new(src, episode_dir / name)
        max_pen = max(
            both[phase][f"{side}_max_penetration_mm"]
            for phase in ("closure", "lifted", "settled")
            for side in ("left", "right")
        )
        rows.append(
            {
                "episode": episode,
                "candidate": item["candidate"],
                "object_name": strict["sample"]["object_name"],
                "physical_pass": True,
                "pose_path": str((episode_dir / "strict_success.pt").resolve()),
                "object_mesh_path": str((episode_dir / "object_mesh.obj").resolve()),
                "strict_report_path": str((episode_dir / "strict_report.json").resolve()),
                "raw_mesh_report_path": str((episode_dir / "raw_mesh_report.json").resolve()),
                "render_path": str((episode_dir / "render_clean/front_side.png").resolve()),
                "closure_left_contact_count": both["closure"]["left_contact_count"],
                "closure_right_contact_count": both["closure"]["right_contact_count"],
                "closure_left_contact_links": both["closure"]["left_contact_links"],
                "closure_right_contact_links": both["closure"]["right_contact_links"],
                "lifted_left_contact_count": both["lifted"]["left_contact_count"],
                "lifted_right_contact_count": both["lifted"]["right_contact_count"],
                "lift_mm": both["lift_object_displacement_m"] * 1000.0,
                "gravity_displacement_mm": both["gravity_displacement_m"] * 1000.0,
                "six_direction_displacements_mm": [x * 1000.0 for x in both["six_direction_displacements_m"]],
                "six_direction_max_displacement_mm": max(both["six_direction_displacements_m"]) * 1000.0,
                "max_physx_penetration_mm": max_pen,
                "left_only_physical_pass": strict["runs"]["left"]["physical_pass"],
                "right_only_physical_pass": strict["runs"]["right"]["physical_pass"],
                "raw_mesh_contact_points_left": raw["left_contact_points"],
                "raw_mesh_contact_points_right": raw["right_contact_points"],
                "raw_mesh_sampled_hand_clearance_mm": raw["hand_clearance_mm"],
                "raw_mesh_watertight": raw["mesh_watertight"],
                "raw_mesh_signed_distance_available": raw["signed_distance_available"],
                "raw_mesh_faces": raw["mesh_faces"],
                "object_mass_kg": both["object_mass_kg"],
                "isaac_pyroki_ee_residual_norm_mm": both["isaac_minus_pyroki_ee_residual_norm_mm"],
                "quality_note": (
                    "Low sampled inter-hand clearance; retain as an Isaac-validated success but treat as a low-margin pose."
                    if episode == "episode_000003"
                    else "No additional low-margin flag."
                ),
            }
        )

    manifest = {
        "schema": "objectflow_four_xhand_isaac_strict_success_v1",
        "robot_urdf": str(URDF),
        "strict_success_count": len(rows),
        "thresholds": {
            "bilateral_contact_at_closure": True,
            "bilateral_contact_after_lift": True,
            "minimum_lift_mm": 30.0,
            "maximum_gravity_displacement_mm": 10.0,
            "maximum_six_direction_displacement_mm": 15.0,
            "maximum_penetration_mm": 2.0,
            "left_only_must_fail": True,
            "right_only_must_fail": True,
        },
        "simulation": SIMULATION,
        "collision_interpretation": {
            "raw_triangle_mesh": "used for static surface-distance/contact precheck",
            "signed_sdf": "not claimed because all four scan meshes are non-watertight",
            "isaac_dynamic_collision": "high-resolution VHACD convex decomposition",
            "final_authority": "Isaac Gym dynamic both/left/right validation",
        },
        "samples": rows,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # 2x2 presentation sheet: each source already contains front and side views.
    card_w, image_h, metrics_h = 1520, 570, 94
    sheet = Image.new("RGB", (card_w * 2, (image_h + metrics_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        x, y = (i % 2) * card_w, (i // 2) * (image_h + metrics_h)
        image = Image.open(row["render_path"]).convert("RGB")
        sheet.paste(image, (x, y))
        y0 = y + image_h
        draw.rectangle((x, y0, x + card_w, y0 + metrics_h), fill=(244, 247, 250))
        line1 = (
            f"Lift {row['lift_mm']:.1f} mm | Gravity {row['gravity_displacement_mm']:.1f} mm | "
            f"6-dir max {row['six_direction_max_displacement_mm']:.1f} mm | Pen {row['max_physx_penetration_mm']:.1f} mm"
        )
        line2 = (
            f"Closure contacts L/R {row['closure_left_contact_count']}/{row['closure_right_contact_count']} | "
            "Left-only FAIL | Right-only FAIL"
        )
        draw.text((x + 18, y0 + 12), line1, fill=(27, 34, 43), font=font(22))
        draw.text((x + 18, y0 + 51), line2, fill=(31, 98, 64), font=font(21))
    gallery = OUT / "objectflow_four_isaac_strict_success_front_side.png"
    sheet.save(gallery)

    lines = [
        "# ObjectFlow 四个真实扫描物体：双 XHand 严格 Isaac 成功样本",
        "",
        f"结论：四个对象均存在至少一条通过严格 Isaac Gym 动态验证的双手抓取 pose（4/4）。",
        "",
        "| 对象 | 候选 | closure 接触 L/R | lift (mm) | 重力位移 (mm) | 六向最大位移 (mm) | 最大 PhysX 穿透 (mm) | 单手消融 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['episode']} | {r['candidate']} | {r['closure_left_contact_count']}/{r['closure_right_contact_count']} | "
            f"{r['lift_mm']:.2f} | {r['gravity_displacement_mm']:.2f} | "
            f"{r['six_direction_max_displacement_mm']:.2f} | {r['max_physx_penetration_mm']:.2f} | L FAIL / R FAIL |"
        )
    lines += [
        "",
        "## 判定口径",
        "",
        "- closure 和 lift 后左右手都必须有真实 PhysX 接触。",
        "- lift ≥ 30 mm；500 步重力位移 ≤ 10 mm；六方向各 100 步扰动的最大位移 ≤ 15 mm。",
        "- closure/lift/settled 最大穿透 ≤ 2 mm；left-only 与 right-only 都必须失败。",
        "- 原始三角网格用于静态表面距离复核，动态最终结论来自 Isaac Gym + 高精度 VHACD。四个扫描 mesh 都不是 watertight，因此不声称有可靠 signed-SDF 穿透值。",
        "- episode_000003 的稀疏 collision-mesh 采样手间距约 0.76 mm，属于低余量样本；它通过 Isaac 动态门槛，但用于 motion planning 时建议增加额外手间安全距离复核。",
        "",
        "## 复现资产",
        "",
        f"- 全身 URDF：`{URDF}`",
        f"- 机器可读清单：`{OUT / 'manifest.json'}`",
        f"- 四对象汇总图：`{gallery}`",
        "- 每个对象目录含 `strict_success.pt`、物体 mesh、严格报告、原网格报告、Isaac 日志以及正/侧视图。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"strict_success_count": len(rows), "manifest": str(OUT / 'manifest.json'), "report": str(OUT / 'REPORT.md'), "gallery": str(gallery)}, indent=2))


if __name__ == "__main__":
    main()
