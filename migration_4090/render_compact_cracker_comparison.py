#!/usr/bin/env python3
"""Render the stretched and compact square Cracker XHand layouts."""
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
URDF_PATH = Path(
    "/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/"
    "xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf"
)
OUT = ROOT / "migration_4090/renders/xhand_compact_cracker_hand_posture_low_v1"
OLD = ROOT / (
    "migration_4090/results/xhand_external_candidates_v1/cracker/"
    "baseline__ycb__cracker_box_formal_large_random_v1_002.pt"
)
NEW = ROOT / "migration_4090/results/xhand_compact_cracker_hand_postures_low_v1.pt"


def sample(path, index):
    return torch.load(path, map_location="cpu", weights_only=False)["samples"][index]


def render_view(scene, output, eye, target, xmag, ymag, width=760, height=450):
    scene.add(
        pyrender.OrthographicCamera(xmag=xmag, ymag=ymag),
        pose=look_at(eye, target),
        name="camera",
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=4.0),
        pose=look_at([2.0, 1.2, 2.4], target),
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=look_at([1.0, -1.2, 1.5], target),
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).save(output)
    finally:
        renderer.delete()


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def add_label(path, title, subtitle, color):
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 76), "white")
    canvas.paste(image, (0, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), title, fill=(20, 24, 30), font=font(21))
    draw.text((12, 42), subtitle, fill=color, font=font(15))
    canvas.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    entries = (
        (
            "stretched",
            sample(OLD, 14),
            "Old stretched object",
            "Y 1.36 m; non-uniform scale; object close to chassis",
            (175, 48, 48),
        ),
        (
            "compact",
            sample(NEW, 5),
            "New compact square object",
            "0.461 x 0.600 x 0.202 m; uniform scale; optimized XHand closure",
            (24, 132, 72),
        ),
    )
    rendered = []
    for key, grasp, title, subtitle, color in entries:
        front = OUT / f"{key}_front.png"
        side = OUT / f"{key}_side.png"
        object_x = float(grasp["object_pose_world"][0][3])
        render_view(
            scene_for_sample(robot, grasp),
            front,
            [2.8, 0.0, 1.25],
            [object_x, 0.0, 0.84],
            0.98,
            0.64,
        )
        render_view(
            scene_for_sample(robot, grasp),
            side,
            [object_x, -2.9, 1.25],
            [object_x, 0.0, 0.84],
            0.98,
            0.64,
        )
        add_label(front, f"{title} | Front", subtitle, color)
        add_label(side, f"{title} | Side", subtitle, color)
        rendered.extend((front, side))

    images = [Image.open(path).convert("RGB") for path in rendered]
    sheet = Image.new(
        "RGB", (images[0].width * 2, images[0].height * 2 + 60), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, 60), fill=(30, 36, 45))
    draw.text(
        (16, 15),
        "Tianji + Dual XHand | Stretched vs Compact Square Object",
        fill="white",
        font=font(24),
    )
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * image.width, 60 + (i // 2) * image.height))
    output = OUT / "stretched_vs_compact_square_cracker.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
