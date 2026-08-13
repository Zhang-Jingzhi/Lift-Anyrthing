#!/usr/bin/env python3
from pathlib import Path
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
from PIL import Image, ImageDraw
from yourdfpy import URDF

from render_xhand_fullbody_final import scene_for_sample, render

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "migration_4090/assets/xhand_fullbody_inward_v1/linkhou_xhand_fullbody_inward_v1.urdf"
OUT = ROOT / "migration_4090/renders/xhand_verified_samples_v1"


def load_one(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)["samples"][0]
    except TypeError:
        return torch.load(path, map_location="cpu")["samples"][0]


def add_title(path, title, subtitle):
    im = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (im.width, im.height + 58), "white")
    canvas.paste(im, (0, 58))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(20, 24, 30))
    draw.text((12, 31), subtitle, fill=(70, 76, 84))
    canvas.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    entries = [
        (
            "bleach_cleanser",
            ROOT / "migration_4090/results/xhand_external_width_1p36/baseline__ycb__bleach_cleanser_formal_large_random_v1_005.pt",
            "historical full gate; position controller also passed",
        ),
        (
            "cracker_box",
            ROOT / "migration_4090/results/xhand_targeted_pose_search_v2/final_gate/verified_candidate19.pt",
            "corrected full gate; teleport rollout 3/3, single-hand ablations failed",
        ),
    ]
    paths = []
    for name, source, subtitle in entries:
        sample = load_one(source)
        path = OUT / f"tianji_xhand_{name}.png"
        render(scene_for_sample(robot, sample), path, width=760, height=560)
        add_title(path, f"Tianji + dual XHand | {name}", subtitle)
        paths.append(path)
    images = [Image.open(p).convert("RGB") for p in paths]
    sheet = Image.new("RGB", (sum(im.width for im in images), max(im.height for im in images)), "white")
    x = 0
    for im in images:
        sheet.paste(im, (x, 0)); x += im.width
    comparison = OUT / "tianji_xhand_verified_comparison.png"
    sheet.save(comparison)
    print(comparison)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
