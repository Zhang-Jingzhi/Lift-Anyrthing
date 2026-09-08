#!/usr/bin/env python3
"""Rewrite a robot USD's collision approximation.

Our robot collides as convex hulls: 192 colliders in
xhand_fixed_physics_flat.usd, 156 of them on the hands, all convexHull, read
from the USD's own attributes rather than guessed.  A hull of a curved finger
link fills in the curve -- measured against the visual meshes, the hulls stand
off their own surfaces by up to 32.4 mm -- so adjacent finger links overlap in
collision space while their real geometry is clear.  With self-collisions on,
PhysX pushes them apart and left_hand_pinky_joint1 sits 0.84 rad from its
commanded position for the rest of the episode, the actuator capped at 1.1 N.m
being unable to pull it back.

The neighbouring project shipped exactly this asset and then replaced it: their
tianji_xhand_fullbody_physics.usd is 96 convexHull colliders and their
tianji_xhand_fullbody_hand_decomp_v2_physics.usd is the same 96 at the same prim
paths as convexDecomposition.  Our source USD has that same structure, so the
approximation attribute is all that needs to change; the mesh data PhysX cooks
from is already in the file.

Writes a new file and leaves the input alone.
"""

from __future__ import annotations

import argparse
import collections
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

from pxr import Usd, UsdPhysics  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--approximation",
    default="convexDecomposition",
    choices=("convexDecomposition", "convexHull", "sdf", "boundingCube"),
)
parser.add_argument(
    "--hands-only",
    action="store_true",
    help="leave arm and body colliders as they are",
)
args = parser.parse_args()

if args.output.exists():
    raise SystemExit(f"{args.output} exists")

stage = Usd.Stage.Open(str(args.source))
if stage is None:
    raise SystemExit(f"could not open {args.source}")

before = collections.Counter()
after = collections.Counter()
changed = 0
skipped = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        continue
    attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
    current = str(attribute.Get() or "(unset)")
    before[current] += 1
    is_hand = "hand" in str(prim.GetPath()).lower()
    if args.hands_only and not is_hand:
        after[current] += 1
        skipped += 1
        continue
    attribute.Set(args.approximation)
    after[args.approximation] += 1
    changed += 1

stage.GetRootLayer().Export(str(args.output))

print(f"source {args.source}")
print(f"output {args.output}")
print(f"  changed {changed} colliders, left {skipped} alone")
print("  before:", dict(before))
print("  after: ", dict(after))
