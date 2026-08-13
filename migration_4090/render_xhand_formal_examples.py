#!/usr/bin/env python3
"""Render representative, strictly Isaac-validated formal XHand samples."""

import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import pyrender
from PIL import Image, ImageDraw, ImageFont
from yourdfpy import URDF

from render_xhand_fullbody_final import look_at, scene_for_sample


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "graph_exp/bimanual_data/xhand_compact_formal_1200_v1"
OUT = ROOT / "migration_4090/renders/xhand_compact_formal_examples_v1"
URDF_PATH = Path(
    "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/"
    "xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf"
)
OBJECTS = ("sphere", "cracker", "pitcher", "drill")
METHODS = ("baseline", "bidex_v3")


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def pick(object_name, method):
    paths = sorted((DATA_ROOT / method / object_name / "accepted").glob("sample_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no accepted sample for {method}/{object_name}")
    # Use the first accepted record: it is fully formal-validated and keeps
    # the selection deterministic for repeated PPT rendering.
    sample_path = paths[0]
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    return sample_path, sample


def render_sample(robot, sample, path):
    object_xyz = np.asarray(sample["object_pose_world"], dtype=np.float32)[:3, 3]
    target = [float(object_xyz[0]), 0.0, float(object_xyz[2])]
    scene = scene_for_sample(robot, sample)
    scene.add(
        pyrender.OrthographicCamera(xmag=1.48, ymag=1.02),
        pose=look_at([2.75, 0.0, 1.40], target),
        name="formal_camera",
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=4.0),
        pose=look_at([2.0, 1.2, 2.4], target),
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=look_at([1.0, -1.2, 1.5], target),
    )
    renderer = pyrender.OffscreenRenderer(620, 440)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).convert("RGB").save(path)
    finally:
        renderer.delete()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    panels = []
    for object_name in OBJECTS:
        for method in METHODS:
            sample_path, sample = pick(object_name, method)
            stem = f"{object_name}_{method}"
            image_path = OUT / f"{stem}.png"
            render_sample(robot, sample, image_path)
            validation = sample["formal_validation"]
            title = (
                f"{object_name} | {method} | size={sample['object_transform']['size_index']}"
            )
            subtitle = (
                f"lift {validation['lift_height_m']*1000:.1f} mm | "
                f"gravity {validation['gravity_displacement_m']*1000:.1f} mm | "
                f"6-dir max {max(validation['six_direction_displacements_m'])*1000:.1f} mm | "
                f"contacts L/R {validation['left_lift_contact_count']}/"
                f"{validation['right_lift_contact_count']}"
            )
            image = Image.open(image_path).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 62), "white")
            canvas.paste(image, (0, 62))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 7), title, fill=(20, 24, 30), font=font(20))
            draw.text((10, 35), subtitle, fill=(70, 76, 84), font=font(14))
            canvas.save(image_path)
            panels.append((object_name, method, canvas))

    cell_w, cell_h = 620, 502
    sheet = Image.new("RGB", (cell_w * 2, cell_h * len(OBJECTS)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, object_name in enumerate(OBJECTS):
        for col, method in enumerate(METHODS):
            image = next(im for obj, meth, im in panels if obj == object_name and meth == method)
            sheet.paste(image, (col * cell_w, row * cell_h))
    comparison = OUT / "formal_validated_examples_4objects_2methods.png"
    sheet.save(comparison)
    print(comparison)
    for object_name, method, _ in panels:
        print(OUT / f"{object_name}_{method}.png")


if __name__ == "__main__":
    main()
