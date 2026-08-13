#!/usr/bin/env python3
"""Render front and side views of one Tianji + dual-XHand candidate."""
import argparse
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

URDF_PATH = Path(os.environ.get(
    "XHAND_FULLBODY_IK_URDF",
    ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf",
))


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render(scene, eye, target, path, xmag=1.0, ymag=0.68):
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag), pose=look_at(eye, target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 1.2, 2.4], target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=look_at([1.0, -1.2, 1.5], target))
    renderer = pyrender.OffscreenRenderer(760, 500)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).convert("RGB").save(path)
    finally:
        renderer.delete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Tianji + dual XHand candidate")
    parser.add_argument(
        "--hide-tcp-markers",
        action="store_true",
        help="Hide the red/blue TCP debug spheres in presentation renders.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = torch.load(args.dataset, map_location="cpu", weights_only=False)["samples"][args.index]
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    # Dataset banks may store either the 38 fixed-IK joints or the complete
    # 51-DoF robot. Project explicitly by name so visualization cannot fail or
    # silently use the wrong joint order.
    render_names = list(robot.actuated_joint_names)
    q_by_name = dict(zip(sample["joint_names"], sample["full_body_q"]))
    render_sample = dict(sample)
    render_sample["joint_names"] = render_names
    render_sample["full_body_q"] = [float(q_by_name[n]) for n in render_names]
    object_xyz = np.asarray(sample["object_pose_world"], dtype=np.float32)[:3, 3]
    target = [float(object_xyz[0]), 0.0, float(object_xyz[2])]
    front = args.output / "front.png"
    side = args.output / "side.png"
    show_tcp = not args.hide_tcp_markers
    render(scene_for_sample(robot, render_sample, show_tcp), [2.9, 0.0, 1.35], target, front)
    render(scene_for_sample(robot, render_sample, show_tcp), [float(object_xyz[0]), -3.0, 1.35], target, side)
    images = [Image.open(front).convert("RGB"), Image.open(side).convert("RGB")]
    sheet = Image.new("RGB", (1520, 570), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, 1520, 70), fill=(30, 36, 45))
    draw.text((18, 18), args.title, fill="white", font=font(25))
    sheet.paste(images[0], (0, 70))
    sheet.paste(images[1], (760, 70))
    output = args.output / "front_side.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
