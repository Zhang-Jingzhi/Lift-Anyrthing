#!/usr/bin/env python3
"""Report every XHand URDF whose right mount disagrees with its left one.

Both XHand meshes are already mirrored, so the two mount rotations must be
equal for the palms to face each other.  A right mount half a turn away puts
the right palm's dorsal side toward the object, which no downstream artifact
reveals -- joint values, contact counts and success rates all look normal.

Exits non-zero if any URDF outside the excluded paths is asymmetric, so this
can gate a build.  Historical trees are excluded by default: the mount trials
under archive/ hold one deliberate variant per directory, and Download/ holds
finished run outputs; rewriting either would falsify a record.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# matched against path components, so this works for relative and absolute roots
DEFAULT_EXCLUDES = ("archive",)
TOLERANCE_RAD = 1e-6


def mount_rpy(root: ET.Element, side: str) -> list[float] | None:
    for joint in root.findall("joint"):
        if joint.get("name") != f"{side}_hand_mount_xhand_v1":
            continue
        origin = joint.find("origin")
        if origin is None:
            return None
        return [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="also check archive/, whose mount trials are deliberate variants",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    checked = 0
    skipped = 0
    bad: list[tuple[Path, list[float], list[float]]] = []
    for root in args.roots:
        for path in sorted(root.rglob("*.urdf")):
            if not args.include_historical and any(
                part in DEFAULT_EXCLUDES for part in path.parts
            ):
                continue
            try:
                tree = ET.parse(path).getroot()
            except (ET.ParseError, OSError):
                skipped += 1
                continue
            right = mount_rpy(tree, "right")
            left = mount_rpy(tree, "left")
            if right is None or left is None:
                continue
            checked += 1
            if not all(abs(r - l) <= TOLERANCE_RAD for r, l in zip(right, left)):
                bad.append((path, right, left))

    if not args.quiet:
        print(f"checked {checked} XHand URDFs ({skipped} unreadable)")
    for path, right, left in bad:
        print(f"ASYMMETRIC  {path}")
        print(f"            right rpy {right}")
        print(f"            left  rpy {left}")
    if bad:
        print(f"\n{len(bad)} URDF(s) would put the right palm the wrong way round")
        sys.exit(1)
    if not args.quiet:
        print("all right mounts match their left mount")


if __name__ == "__main__":
    main()
