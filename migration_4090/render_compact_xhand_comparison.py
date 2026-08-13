#!/usr/bin/env python3
"""Render old stretched and new compact XHand object layouts."""
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
URDF_PATH = Path("/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf")
OUT = ROOT / "migration_4090/renders/xhand_compact_piggy_smoke_v1"


def load_sample(path, index):
    return torch.load(path, map_location="cpu", weights_only=False)["samples"][index]


def render_view(scene, output, eye, target, xmag, ymag, width=720, height=430):
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag), pose=look_at(eye, target), name="camera")
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 1.2, 2.4], target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=look_at([1.0, -1.2, 1.5], target))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).save(output)
    finally:
        renderer.delete()


def font(size):
    p = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(p), size) if p.is_file() else ImageFont.load_default()


def label(path, title, subtitle, color):
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 72), "white")
    canvas.paste(image, (0, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(20, 24, 30), font=font(21))
    draw.text((12, 39), subtitle, fill=color, font=font(15))
    canvas.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    old_path = next((ROOT / "migration_4090/results/xhand_external_candidates_v1/piggy").glob("baseline__*.pt"))
    new_path = next((ROOT / "migration_4090/results/xhand_compact_piggy_smoke_v1").glob("baseline__*.pt"))
    entries = (
        ("old", load_sample(old_path, 21), "Old stretched layout", "Y extent 1.36 m; anisotropic stretch", (170, 52, 52)),
        ("compact", load_sample(new_path, 0), "New compact layout", "Y extent 0.55 m; uniform scale; object +0.40 m forward", (28, 132, 72)),
    )
    paths = []
    for key, sample, title, subtitle, color in entries:
        front = OUT / f"{key}_front.png"
        side = OUT / f"{key}_side.png"
        render_view(scene_for_sample(robot, sample), front, [2.6, 0.0, 1.18], [sample["object_pose_world"][0][3], 0.0, 0.86], 0.95, 0.62)
        render_view(scene_for_sample(robot, sample), side, [0.0, -2.8, 1.22], [0.16, 0.0, 0.86], 0.90, 0.62)
        label(front, f"{title} | Front", subtitle, color)
        label(side, f"{title} | Side", subtitle, color)
        paths.extend((front, side))
    images = [Image.open(p).convert("RGB") for p in paths]
    sheet = Image.new("RGB", (images[0].width * 2, images[0].height * 2 + 56), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, 56), fill=(30, 36, 45))
    draw.text((16, 13), "Tianji + Dual XHand | Stretched vs Compact Object Layout", fill="white", font=font(24))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % 2) * image.width, 56 + (i // 2) * image.height))
    output = OUT / "stretched_vs_compact_piggy_comparison.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
