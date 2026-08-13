#!/usr/bin/env python3
"""Diagnostic front/side renders for one formally validated sample."""

import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
from PIL import Image, ImageDraw, ImageFont
from yourdfpy import URDF

from render_xhand_fullbody_final import look_at, scene_for_sample

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "graph_exp/bimanual_data/xhand_compact_formal_1200_v1"
OUT = ROOT / "migration_4090/renders/xhand_compact_formal_diagnostic_v1"
URDF_PATH = Path(
    "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/"
    "xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"
)


def text_font(size):
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def one_sample(obj, method):
    pt = sorted((DATA_ROOT / method / obj / "accepted").glob("sample_*.pt"))[0]
    return torch.load(pt, map_location="cpu", weights_only=False), pt


def one_view(robot, sample, eye, target, path):
    # The Isaac fixed_ik URDF removes waist/wheel DOFs.  The formal sample
    # stores the 51-DOF source configuration, so project it by joint name to
    # the 38 actuated joints used by the fixed visualization URDF.
    names = list(getattr(robot, "actuated_joint_names"))
    values = dict(zip(sample["joint_names"], sample["full_body_q"]))
    render_sample = dict(sample)
    render_sample["joint_names"] = names
    render_sample["full_body_q"] = [float(values[name]) for name in names]
    scene = scene_for_sample(robot, render_sample)
    scene.add(
        pyrender.OrthographicCamera(xmag=0.95, ymag=0.72),
        pose=look_at(eye, target),
        name="diagnostic_camera",
    )
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 1.2, 2.4], target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=look_at([1.0, -1.2, 1.5], target))
    renderer = pyrender.OffscreenRenderer(760, 560)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).convert("RGB").save(path)
    finally:
        renderer.delete()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    panels = []
    for obj in ("cracker", "drill"):
        for method in ("baseline", "bidex_v3"):
            sample, source = one_sample(obj, method)
            center = np.asarray(sample["object_pose_world"], dtype=np.float32)[:3, 3]
            target = [float(center[0]), 0.0, float(center[2])]
            front_path = OUT / f"{obj}_{method}_front.png"
            side_path = OUT / f"{obj}_{method}_side.png"
            one_view(robot, sample, [2.75, 0.0, 1.25], target, front_path)
            one_view(robot, sample, [float(center[0]), -2.75, 1.15], target, side_path)
            front = Image.open(front_path).convert("RGB")
            side = Image.open(side_path).convert("RGB")
            canvas = Image.new("RGB", (1520, 610), "white")
            canvas.paste(front, (0, 50)); canvas.paste(side, (760, 50))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 10), f"{obj} | {method} | fixed_ik URDF | left: front, right: side", fill=(20, 24, 30), font=text_font(22))
            draw.text((12, 35), f"source={source.name} | object size index={sample['object_transform']['size_index']} | Isaac strict repeat=3/3", fill=(70, 76, 84), font=text_font(14))
            out = OUT / f"{obj}_{method}_front_side_fixed_ik.png"
            canvas.save(out)
            panels.append((obj, method, canvas))
    sheet = Image.new("RGB", (1520, 610 * len(panels)), "white")
    for i, (_, _, panel) in enumerate(panels):
        sheet.paste(panel, (0, i * 610))
    out = OUT / "cracker_drill_fixed_ik_front_side_diagnostic.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
