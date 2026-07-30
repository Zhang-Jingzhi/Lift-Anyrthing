import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from matplotlib import font_manager

from make_checkpoint_grasp_report import (
    load_epoch,
    mesh_collection,
    set_equal_limits,
)
from utils.hand_model import create_hand_model


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "graph_exp/reproduction/train-multi-hand-rtx3070"
EVALUATION_ROOT = RUN_DIR / "evaluations"
OUTPUT_DIR = RUN_DIR / "checkpoint_visual_comparison/advisor_brief"
EPOCHS = [10, 30, 50, 60]

COLORS = {
    "shadow": "#3478bf",
    "allegro": "#ee8a2d",
    "success": "#18864b",
    "failure": "#c83b3b",
    "object": (0.94, 0.52, 0.65),
    "robot": (0.32, 0.69, 0.95),
}

SELECTIONS = {
    "shadowhand_unconditioned": [
        ("苹果", "contactdb+apple", 14),
        ("相机", "contactdb+camera", 17),
        ("水瓶", "contactdb+water_bottle", 18),
    ],
    "allegro_conditioned": [
        ("苹果", "contactdb+apple", 7),
        ("相机", "contactdb+camera", 19),
        ("水瓶", "contactdb+water_bottle", 14),
    ],
}

MODE_TITLES = {
    "shadowhand_unconditioned": "ShadowHand 无条件生成",
    "allegro_conditioned": "AllegroHand 条件生成",
}


def configure_chinese_font():
    font_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_metrics():
    rows = list(csv.DictReader((RUN_DIR / "evaluation_summary.csv").open()))
    metrics = {}
    for row in rows:
        key = (int(row["epoch"]), row["evaluation"])
        metrics[key] = {
            "success": float(row["success_rate"]) * 100,
            "penetration": float(row["penetration_pass_rate"]) * 100,
            "joint": float(row["joint_success_rate"]) * 100,
        }
    return metrics


def render_metric_trends():
    metrics = load_metrics()
    all_epochs = [10, 20, 30, 40, 50, 60]
    modes = [
        ("shadowhand_unconditioned", "ShadowHand", COLORS["shadow"]),
        ("allegro_conditioned", "AllegroHand", COLORS["allegro"]),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), constrained_layout=True)
    for mode, label, color in modes:
        success = [metrics[(epoch, mode)]["success"] for epoch in all_epochs]
        penetration = [
            metrics[(epoch, mode)]["penetration"] for epoch in all_epochs
        ]
        joint = [metrics[(epoch, mode)]["joint"] for epoch in all_epochs]
        axes[0].plot(
            all_epochs,
            success,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=color,
            label=label,
        )
        axes[1].plot(
            all_epochs,
            penetration,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=color,
            label=label,
        )
        axes[2].plot(
            all_epochs,
            joint,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=color,
            label=label,
        )
        for axis, values in zip(axes, [success, penetration, joint]):
            for epoch, value in zip(all_epochs, values):
                axis.annotate(
                    f"{value:.1f}%",
                    (epoch, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8.5,
                    color=color,
                )

    axes[0].set_title("Isaac 六方向扰动抓取成功率", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("成功率（%）")
    axes[0].set_ylim(60, 100)
    axes[0].annotate(
        "ShadowHand 从第30轮起进入平台",
        xy=(40, 95),
        xytext=(28, 98.2),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
    )
    axes[0].annotate(
        "AllegroHand 第60轮达到90%",
        xy=(60, 90),
        xytext=(42, 87),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
    )

    axes[1].set_title(
        "网格穿透深度 ≤ 5 mm 的比例",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_ylabel("合格率（%）")
    axes[1].set_ylim(50, 102)
    axes[1].annotate(
        "使用模型实际 rollout 起点",
        xy=(60, metrics[(60, "allegro_conditioned")]["penetration"]),
        xytext=(33, 72),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
    )

    axes[2].set_title(
        "联合成功率：穿透合格且扰动成功",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_ylabel("联合成功率（%）")
    axes[2].set_ylim(50, 100)
    axes[2].annotate(
        "AllegroHand 第30轮联合指标最高",
        xy=(30, metrics[(30, "allegro_conditioned")]["joint"]),
        xytext=(34, 74),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=9.5,
    )

    for axis in axes:
        axis.set_xlabel("训练轮数（Epoch）")
        axis.set_xticks(all_epochs)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    figure.suptitle(
        "TRO-Grasp 阶段性定量结果（固定3物体 × 每物体20个抓取）",
        fontsize=17,
        fontweight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "01_metric_trends.png"
    figure.savefig(path, dpi=210, facecolor="white")
    plt.close(figure)
    return path


def render_object_coverage(mode):
    epoch_samples = {
        epoch: load_epoch(EVALUATION_ROOT, epoch, mode) for epoch in EPOCHS
    }
    robot_name = next(iter(epoch_samples[EPOCHS[0]].values()))["robot_name"]
    hand = create_hand_model(robot_name, torch.device("cpu"))

    figure, axes = plt.subplots(
        3,
        len(EPOCHS),
        figsize=(15.2, 10.5),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    selection_rows = []
    for row_index, (object_cn, object_name, sample_index) in enumerate(
        SELECTIONS[mode]
    ):
        key = (object_name, sample_index)
        dataset, name = object_name.split("+")
        object_mesh = trimesh.load_mesh(
            ROOT / "data/data_urdf/object" / dataset / name / f"{name}.stl",
            force="mesh",
        )
        robot_meshes = {
            epoch: hand.get_trimesh_q(epoch_samples[epoch][key]["q"])["visual"]
            for epoch in EPOCHS
        }
        combined_bounds = np.vstack(
            [object_mesh.bounds]
            + [robot_meshes[epoch].bounds for epoch in EPOCHS]
        )
        bounds = np.vstack(
            [combined_bounds.min(axis=0), combined_bounds.max(axis=0)]
        )

        for column_index, epoch in enumerate(EPOCHS):
            axis = axes[row_index, column_index]
            success = epoch_samples[epoch][key]["success"]
            axis.add_collection3d(
                mesh_collection(
                    object_mesh,
                    color=COLORS["object"],
                    alpha=0.78,
                    max_faces=65000,
                )
            )
            axis.add_collection3d(
                mesh_collection(
                    robot_meshes[epoch],
                    color=COLORS["robot"],
                    alpha=0.95,
                    max_faces=50000,
                )
            )
            set_equal_limits(axis, bounds)
            axis.view_init(elev=24, azim=42)
            axis.set_axis_off()
            axis.set_title(
                f"第 {epoch} 轮｜{'成功' if success else '失败'}",
                fontsize=12,
                fontweight="bold",
                color=COLORS["success"] if success else COLORS["failure"],
            )
            selection_rows.append(
                {
                    "mode": mode,
                    "object": object_name,
                    "object_cn": object_cn,
                    "sample_index": sample_index,
                    "epoch": epoch,
                    "success": success,
                }
            )
        axes[row_index, 0].text2D(
            -0.10,
            0.5,
            f"{object_cn}\n固定样本 {sample_index}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    figure.suptitle(
        f"{MODE_TITLES[mode]}：同一物体、同一样本的跨 checkpoint 对比",
        fontsize=17,
        fontweight="bold",
    )
    path = OUTPUT_DIR / f"02_{mode}_three_objects.png"
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)

    csv_path = OUTPUT_DIR / f"02_{mode}_three_objects.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=selection_rows[0].keys())
        writer.writeheader()
        writer.writerows(selection_rows)
    return path


def main():
    configure_chinese_font()
    print(render_metric_trends())
    for mode in SELECTIONS:
        print(render_object_coverage(mode))


if __name__ == "__main__":
    main()
