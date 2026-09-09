#!/usr/bin/env python3
"""Is the geometry we audit against the geometry the simulator collides with?

Every candidate audit in this pipeline samples the URDF's visual meshes, and
every one of them passed while the same poses made no contact in PhysX.  The
collision approximation explains part of that -- convex hulls stand off their own
surfaces by up to 32.4 mm -- but not whether the underlying mesh is even the
same.  If the USD's collision source differs from the URDF's visual mesh, the
audit has been measuring a different object from the start.

Compares, per hand link, the vertex count and bounding box of the USD collision
mesh against the URDF visual mesh.  Both are read from the files themselves.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

SITE = Path(
    "/media/home/st/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache"
)
for pattern in ("omni.usd.libs-*",):
    for path in sorted(glob.glob(str(SITE / pattern))):
        if path not in sys.path:
            sys.path.insert(0, path)

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from yourdfpy import URDF  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--links", type=int, default=8)
args = parser.parse_args()

stage = Usd.Stage.Open(str(args.usd))
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

collision = {}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    # The collider is an Xform; the geometry hangs under it as a "mesh" child.
    points = None
    for child in prim.GetChildren():
        mesh = UsdGeom.Mesh(child)
        if mesh:
            candidate = mesh.GetPointsAttr().Get()
            if candidate:
                points = candidate
                break
    if not points:
        continue
    array = np.asarray(points, dtype=np.float64)
    # ".../<link>/collisions/<link>/node_..." -- the link name is the ancestor
    # directly above "collisions".
    parts = str(prim.GetPath()).split("/")
    link = None
    for index, part in enumerate(parts):
        if part == "collisions" and index > 0:
            link = parts[index - 1]
            break
    if link is None:
        continue
    entry = collision.setdefault(link, {"vertices": 0, "min": None, "max": None})
    entry["vertices"] += len(array)
    low, high = array.min(axis=0), array.max(axis=0)
    entry["min"] = low if entry["min"] is None else np.minimum(entry["min"], low)
    entry["max"] = high if entry["max"] is None else np.maximum(entry["max"], high)

robot = URDF.load(str(args.urdf))
visual = {}
# The URDF scene names geometries by mesh file, so go through the scene graph to
# recover which link each one belongs to.
for link in robot.scene.graph.nodes_geometry:
    _, geometry_name = robot.scene.graph.get(link)
    geometry = robot.scene.geometry.get(geometry_name)
    if geometry is None or not hasattr(geometry, "vertices"):
        continue
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    # The scene graph keys these by mesh file; the USD names the same links
    # without the extension.
    key = link[:-4] if link.lower().endswith(".stl") else link
    entry = visual.setdefault(
        key, {"vertices": 0, "min": None, "max": None}
    )
    entry["vertices"] += len(vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    entry["min"] = low if entry["min"] is None else np.minimum(entry["min"], low)
    entry["max"] = high if entry["max"] is None else np.maximum(entry["max"], high)

print(f"USD collision links: {len(collision)}   URDF visual geometries: {len(visual)}")
print()
shared = [name for name in collision if name in visual]
print(f"names present in both: {len(shared)}")
if not shared:
    print("USD link names (sample):   ", sorted(collision)[:6])
    print("URDF geometry names (sample):", sorted(visual)[:6])
    raise SystemExit(0)

print()
print("%-34s %10s %10s %12s" % ("link", "USD verts", "URDF verts", "size diff mm"))
worst = 0.0
for name in sorted(shared)[: args.links]:
    a, b = collision[name], visual[name]
    size_a = a["max"] - a["min"]
    size_b = b["max"] - b["min"]
    difference = float(np.abs(size_a - size_b).max()) * 1000.0
    worst = max(worst, difference)
    print(
        "%-34s %10d %10d %12.3f"
        % (name, a["vertices"], b["vertices"], difference)
    )
print()
print("largest bounding-box difference over the compared links: %.3f mm" % worst)
