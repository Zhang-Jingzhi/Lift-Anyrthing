#!/usr/bin/env python3
"""Render representative poses from the final Tianji + dual-XHand datasets.

This is a visualization utility only. It reads the final .pt files and does
not modify any dataset or source mesh.
"""
from pathlib import Path
import math
import os

import numpy as np
import torch
import trimesh
from yourdfpy import URDF
import pyrender
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = Path(os.environ.get("XHAND_FULLBODY_URDF", ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf"))
DATA_DIR = Path(os.environ.get("XHAND_FINAL_DATA_DIR", ROOT / "migration_4090/results/xhand_fullbody_grasps_v1_final"))
OUT_DIR = Path(os.environ.get("XHAND_RENDER_OUT", ROOT / "migration_4090/renders/xhand_fullbody_final_v1"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTS = (
    "ycb+bleach_cleanser",
    "ycb+cracker_box",
    "ycb+pitcher_base",
    "contactdb+piggy_bank",
    "ycb+power_drill",
    "ycb+toy_airplane",
)


def pick_samples(method):
    payload = torch.load(DATA_DIR / f"{method}.pt", weights_only=False)
    chosen = []
    for base in OBJECTS:
        matches = [s for s in payload["samples"] if s["object_name"].startswith(base + "_formal_large_random_v1_")]
        if not matches:
            raise RuntimeError(f"No final sample for {base} in {method}")
        # Prefer a middle schedule entry for a representative, non-smoke pose.
        chosen.append(sorted(matches, key=lambda s: int(s["schedule_index"]))[len(matches) // 2])
    return chosen


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def sphere(center, radius, color):
    mesh = trimesh.creation.uv_sphere(radius=radius, count=[8, 8])
    mesh.apply_translation(center)
    return pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*color, 1.0), metallicFactor=0.0, roughnessFactor=0.8))


def scene_for_sample(robot, sample, show_tcp_markers=True):
    robot.update_cfg(np.asarray(sample["full_body_q"], dtype=np.float32))
    scene = pyrender.Scene(bg_color=[0.96, 0.97, 0.98, 1.0], ambient_light=[0.35, 0.35, 0.35])

    # Surface samples keep the full-body model legible while avoiding a huge
    # per-panel triangle upload from the original STL files.
    left_color = np.array([0.10, 0.38, 0.85, 1.0], dtype=np.float32)
    right_color = np.array([0.88, 0.20, 0.12, 1.0], dtype=np.float32)
    body_color = np.array([0.48, 0.52, 0.58, 1.0], dtype=np.float32)
    for node in robot.scene.graph.nodes_geometry:
        transform, geom_name = robot.scene.graph.get(node)
        geom = robot.scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        count = 140 if "hand" in geom_name.lower() else 90
        points, _ = trimesh.sample.sample_surface(geom, count)
        points = trimesh.transform_points(points, transform)
        if geom_name.lower().startswith("left_hand"):
            color = left_color
        elif geom_name.lower().startswith("right_hand"):
            color = right_color
        else:
            color = body_color
        colors = np.tile((color * 255).astype(np.uint8), (len(points), 1))
        scene.add(pyrender.Mesh.from_points(points, colors=colors))

    obj = trimesh.load_mesh(sample["object_mesh_path"], force="mesh", process=False)
    obj.apply_transform(np.asarray(sample["object_pose_world"], dtype=np.float32))
    obj.visual.face_colors = np.tile(np.array([242, 143, 24, 255], dtype=np.uint8), (len(obj.faces), 1))
    object_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.95, 0.48, 0.06, 1.0], metallicFactor=0.0, roughnessFactor=0.82
    )
    scene.add(pyrender.Mesh.from_trimesh(obj, material=object_material, smooth=False))

    if show_tcp_markers:
        tcp = np.asarray(sample["achieved_tcp_positions_world"], dtype=np.float32)
        scene.add(sphere(tcp[0], 0.028, (0.08, 0.25, 1.0)))
        scene.add(sphere(tcp[1], 0.028, (1.0, 0.08, 0.04)))

    # A thin tabletop indicates the fixed support height used by generation.
    table = trimesh.creation.box(extents=[0.9, 2.2, 0.025], transform=trimesh.transformations.translation_matrix([0.0, 0.0, 0.707]))
    table.visual.face_colors = [205, 210, 216, 255]
    scene.add(pyrender.Mesh.from_trimesh(table, smooth=False))
    return scene


def render(scene, path, width=620, height=460):
    camera = pyrender.OrthographicCamera(xmag=1.72, ymag=1.24)
    # Look along the robot's x axis so the opposed wrists appear on opposite
    # sides of the object instead of overlapping in a diagonal view.
    scene.add(camera, pose=look_at([2.60, 0.0, 1.40], [0.0, 0.0, 0.78]), name="camera")
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 1.2, 2.4], [0.0, 0.0, 0.7]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=look_at([1.0, -1.2, 1.5], [0.0, 0.0, 0.7]))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).save(path)
    finally:
        renderer.delete()


def label(path, title):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(245, 247, 249))
    draw.text((10, 8), title, fill=(24, 29, 35))
    image.save(path)


def main():
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    methods = {"baseline": pick_samples("baseline"), "bidex_v3": pick_samples("bidex_v3")}
    rendered = {}
    for method, samples in methods.items():
        paths = []
        for sample in samples:
            name = sample["object_name"].replace("+", "_")
            path = OUT_DIR / f"{method}_{name}_schedule{int(sample['schedule_index']):04d}.png"
            render(scene_for_sample(robot, sample), path)
            label(path, f"Tianji + dual XHand | {method} | {sample['object_name']} | schedule {sample['schedule_index']}")
            paths.append(path)
        rendered[method] = paths

    # Compose separate six-object sheets (3x2) and one wide method comparison.
    thumbs = []
    for method in ("baseline", "bidex_v3"):
        row = [Image.open(p).convert("RGB") for p in rendered[method]]
        thumbs.append(row)
    cell_w, cell_h = 420, 330
    for method, row in zip(("baseline", "bidex_v3"), thumbs):
        sheet = Image.new("RGB", (cell_w * 3, cell_h * 2), "white")
        for i, im in enumerate(row):
            im.thumbnail((cell_w, cell_h))
            sheet.paste(im, ((i % 3) * cell_w, (i // 3) * cell_h))
        sheet.save(OUT_DIR / f"{method}_six_objects.png")
    sheet = Image.new("RGB", (cell_w * 6, cell_h * 2), "white")
    for r, row in enumerate(thumbs):
        for c, im in enumerate(row):
            im.thumbnail((cell_w, cell_h))
            sheet.paste(im, (c * cell_w, r * cell_h))
    sheet.save(OUT_DIR / "baseline_vs_bidex_v3_six_objects.png")
    print("\n".join(str(p) for paths in rendered.values() for p in paths))
    print(OUT_DIR / "baseline_vs_bidex_v3_six_objects.png")


if __name__ == "__main__":
    main()
