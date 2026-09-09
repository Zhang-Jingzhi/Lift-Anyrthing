#!/usr/bin/env python3
"""Read the collision approximation a robot USD actually declares.

Grepping `strings` over a USD crate file cannot answer this: the token table is
compressed, so neither link names nor approximation tokens show up reliably.  An
earlier comparison of our asset against the neighbouring project's was made that
way and has to be redone properly.

Uses the USD libraries shipped inside the Isaac Sim install rather than a pxr
package, which none of the interpreters here provide on their own.
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
for pattern in ("omni.usd.libs-*", "omni.usd.schema.physx-*", "usdrt.scenegraph-*"):
    for path in sorted(glob.glob(str(SITE / pattern))):
        if path not in sys.path:
            sys.path.insert(0, path)
        lib = os.path.join(path, "bin")
        if os.path.isdir(lib):
            os.environ["LD_LIBRARY_PATH"] = (
                lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )

from pxr import Usd, UsdPhysics  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("usd", type=Path)
parser.add_argument("--show", type=int, default=8)
args = parser.parse_args()

stage = Usd.Stage.Open(str(args.usd))
if stage is None:
    raise SystemExit(f"could not open {args.usd}")

approximations = collections.Counter()
per_hand = collections.Counter()
samples = collections.defaultdict(list)
colliders = 0
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue
    colliders += 1
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        value = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        token = str(value) if value else "(unset)"
    else:
        token = "(primitive)"
    approximations[token] += 1
    path = str(prim.GetPath())
    if "hand" in path.lower():
        per_hand[token] += 1
    if len(samples[token]) < args.show:
        samples[token].append(path)

print(args.usd)
print(f"  prims with a collider: {colliders}")
print("  approximation over all colliders:")
for token, count in approximations.most_common():
    print(f"    {token:<24} {count}")
if per_hand:
    print("  approximation over hand colliders only:")
    for token, count in per_hand.most_common():
        print(f"    {token:<24} {count}")
print("  examples:")
for token, paths in samples.items():
    for path in paths[:2]:
        print(f"    {token:<24} {path[-70:]}")
