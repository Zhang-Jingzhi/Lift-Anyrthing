#!/usr/bin/env python3
"""Render the current 6-object x 2-method external-XHand screening status."""
import json
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
CANDIDATES = ROOT / "migration_4090/results/xhand_external_candidates_v1"
REPORTS = ROOT / "migration_4090/results/xhand_external_strict_seed_search_v1/candidates"
OUT = ROOT / "migration_4090/renders/xhand_external_12way_status_v1"
URDF_PATH = Path("/media/home/st/curobo_robot_assets/tianji_xhand/migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf")

OBJECTS = (
    ("bleach", "Bleach Cleanser"),
    ("cracker", "Cracker Box"),
    ("pitcher", "Pitcher Base"),
    ("piggy", "Piggy Bank"),
    ("drill", "Power Drill"),
    ("toy", "Toy Airplane"),
)
METHODS = (("baseline", "Baseline"), ("bidex_v3", "BiDex-v3"))


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def is_pass(report):
    return report is not None and bool(report.get("physical_pass"))


def candidate_rank(candidate_dir):
    both = read_json(candidate_dir / "both.json")
    strict = read_json(candidate_dir / "strict.json")
    repeat2 = read_json(candidate_dir / "repeat2.json")
    repeat3 = read_json(candidate_dir / "repeat3.json")
    verified = all(is_pass(x) for x in (strict, repeat2, repeat3))
    strict_pass = is_pass(strict)
    both_pass = is_pass(both)
    report = strict if strict_pass else both
    if report is None or "both" not in report.get("runs", {}):
        return None
    run = report["runs"]["both"]
    closure = run["closure"]
    lifted = run["lifted"]
    settled = run["settled"]
    gravity = float(run["gravity_displacement_m"])
    lift = float(run["lift_object_displacement_m"])
    max6 = max(map(float, run["six_direction_displacements_m"]))
    rank = (
        int(verified),
        int(strict_pass),
        int(both_pass),
        int(settled["left_contact_count"] > 0 and settled["right_contact_count"] > 0),
        int(lifted["left_contact_count"] > 0 and lifted["right_contact_count"] > 0),
        int(closure["left_contact_count"] > 0 and closure["right_contact_count"] > 0),
        -min(gravity, 100.0),
        -abs(lift - 0.05),
    )
    return rank, {
        "candidate_dir": candidate_dir,
        "index": int(candidate_dir.name.rsplit("_", 1)[1]),
        "verified": verified,
        "strict_pass": strict_pass,
        "both_pass": both_pass,
        "run": run,
        "lift_mm": lift * 1000.0,
        "gravity_mm": gravity * 1000.0,
        "max6_mm": max6 * 1000.0,
    }


def select(object_key, method):
    choices = []
    for candidate_dir in sorted((REPORTS / object_key / method).glob("candidate_*")):
        ranked = candidate_rank(candidate_dir)
        if ranked is not None:
            choices.append(ranked)
    if not choices:
        raise RuntimeError(f"No completed report for {object_key}/{method}")
    _, selected = max(choices, key=lambda x: x[0])
    datasets = sorted((CANDIDATES / object_key).glob(f"{method}__*.pt"))
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one dataset for {object_key}/{method}, got {datasets}")
    payload = torch.load(datasets[0], map_location="cpu", weights_only=False)
    selected["sample"] = payload["samples"][selected["index"]]
    selected["dataset"] = datasets[0]
    return selected


def render_closeup(scene, path, width=560, height=360):
    camera = pyrender.OrthographicCamera(xmag=0.94, ymag=0.60)
    scene.add(camera, pose=look_at([2.60, 0.0, 1.15], [0.0, 0.0, 0.85]), name="camera")
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=look_at([2.0, 1.2, 2.4], [0.0, 0.0, 0.8]))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=look_at([1.0, -1.2, 1.5], [0.0, 0.0, 0.8]))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        Image.fromarray(color).save(path)
    finally:
        renderer.delete()


def font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def decorate(render_path, object_label, method_label, selected):
    image = Image.open(render_path).convert("RGB")
    top, bottom = 72, 66
    canvas = Image.new("RGB", (image.width, image.height + top + bottom), "white")
    canvas.paste(image, (0, top))
    draw = ImageDraw.Draw(canvas)
    if selected["verified"]:
        band = (27, 132, 72)
        status = "STRICT VERIFIED 3/3"
    elif selected["strict_pass"]:
        band = (190, 125, 20)
        status = "STRICT PASS; REPEATS PENDING"
    elif selected["both_pass"]:
        band = (190, 125, 20)
        status = "BOTH-HAND PASS; ABLATION PENDING"
    else:
        band = (168, 55, 55)
        status = "BEST CURRENT CANDIDATE - NOT STRICT"
    draw.rectangle((0, 0, canvas.width, top), fill=(246, 248, 250))
    draw.rectangle((0, top - 8, canvas.width, top), fill=band)
    draw.text((12, 8), f"{object_label} | {method_label}", fill=(18, 23, 29), font=font(22))
    draw.text((12, 38), f"{status} | index {selected['index']}", fill=band, font=font(15))
    run = selected["run"]
    closure = run["closure"]
    metrics = (
        f"contacts L/R {closure['left_contact_count']}/{closure['right_contact_count']}   "
        f"lift {selected['lift_mm']:.1f} mm   gravity {selected['gravity_mm']:.1f} mm   "
        f"max6 {selected['max6_mm']:.1f} mm"
    )
    draw.rectangle((0, top + image.height, canvas.width, canvas.height), fill=(246, 248, 250))
    draw.text((12, top + image.height + 18), metrics, fill=(38, 44, 52), font=font(14))
    canvas.save(render_path)


def compose(entries, output, columns, title):
    images = [Image.open(e["path"]).convert("RGB") for e in entries]
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    rows = (len(images) + columns - 1) // columns
    title_h = 58
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + title_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, title_h), fill=(30, 36, 45))
    draw.text((18, 14), title, fill="white", font=font(25))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % columns) * cell_w, title_h + (i // columns) * cell_h))
    sheet.save(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = URDF.load(str(URDF_PATH), build_scene_graph=True, load_meshes=True)
    entries = []
    selection = []
    for object_key, object_label in OBJECTS:
        for method, method_label in METHODS:
            selected = select(object_key, method)
            path = OUT / f"{object_key}_{method}.png"
            render_closeup(scene_for_sample(robot, selected["sample"]), path)
            decorate(path, object_label, method_label, selected)
            entry = {"object": object_key, "method": method, "path": path, **selected}
            entries.append(entry)
            selection.append({
                "object": object_key,
                "method": method,
                "candidate_index": selected["index"],
                "verified_3_of_3": selected["verified"],
                "strict_pass": selected["strict_pass"],
                "both_pass": selected["both_pass"],
                "lift_mm": selected["lift_mm"],
                "gravity_mm": selected["gravity_mm"],
                "max6_mm": selected["max6_mm"],
                "dataset": str(selected["dataset"]),
                "report_dir": str(selected["candidate_dir"]),
                "render": str(path),
            })

    # Pair baseline and BiDex panels for each object in a 4x3 PPT-friendly sheet.
    compose(entries, OUT / "xhand_6objects_2methods_status_12way.png", 4,
            "Tianji + Dual XHand | 6 Objects x 2 Generation Methods | Isaac Gym Status")
    for method, method_label in METHODS:
        subset = [entry for entry in entries if entry["method"] == method]
        compose(subset, OUT / f"xhand_{method}_six_objects_status.png", 3,
                f"Tianji + Dual XHand | {method_label} | Six Objects")
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(OUT / "xhand_6objects_2methods_status_12way.png")
    print(OUT / "xhand_baseline_six_objects_status.png")
    print(OUT / "xhand_bidex_v3_six_objects_status.png")
    print(OUT / "selection.json")


if __name__ == "__main__":
    main()
