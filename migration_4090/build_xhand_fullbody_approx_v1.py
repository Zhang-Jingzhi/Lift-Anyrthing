#!/usr/bin/env python3
"""Assemble a non-destructive full-body URDF with standalone XHand v1.3.

The existing LinkHou arm/waist chain is retained. The old custom hand
subtrees are removed and the v1.3 left/right hand packages are attached at the
old R_tcp/L_tcp mount transforms. This is an approximate kinematic asset; the
mount transform must be checked in IsaacLab before physical claims.
"""

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_URDF = Path("/media/home/tiechenrui/linkhou_urdf_0618/urdf/linkhou_urdf_0618.urdf")
HAND_ROOT = ROOT / "migration_4090/assets/xhand1_urdf_ver1.3/XHAND1_URDF_ver 1.3"
OUT_DIR = ROOT / "migration_4090/assets/xhand_fullbody_approx_v1"
OUT_URDF = OUT_DIR / "linkhou_xhand_fullbody_approx_v1.urdf"
OUT_MANIFEST = OUT_DIR / "manifest.json"


def absolute_mesh_filename(filename: str, source_root: Path, original: bool) -> str:
    if filename.startswith("package://linkhou_urdf_0618/meshes/"):
        rel = filename.split("/meshes/", 1)[1]
        return str(Path("/media/home/tiechenrui/linkhou_urdf_0618/meshes") / rel)
    if filename.startswith("package://xhand_left/meshes/"):
        return str(HAND_ROOT / "xhand1_left/meshes" / filename.split("/meshes/", 1)[1])
    if filename.startswith("package://xhand_right/meshes/"):
        return str(HAND_ROOT / "xhand1_right/meshes" / filename.split("/meshes/", 1)[1])
    if filename.startswith("meshes/"):
        return str(source_root / filename)
    return filename


def rewrite_meshes(root: ET.Element, source_root: Path, original: bool) -> None:
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", absolute_mesh_filename(filename, source_root, original))


def remove_hand_subtrees(robot: ET.Element, keep_roots=()) -> dict[str, list[str]]:
    links = {link.get("name") for link in robot.findall("link")}
    joints = robot.findall("joint")
    children = defaultdict(list)
    joints_by_child = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        p, c = parent.get("link"), child.get("link")
        children[p].append(c)
        joints_by_child[c] = joint

    remove_links = set()
    for root_link in ("R_tcp", "L_tcp"):
        queue = deque([root_link])
        while queue:
            link = queue.popleft()
            if link in remove_links:
                continue
            if link not in keep_roots:
                remove_links.add(link)
            queue.extend(children.get(link, []))

    remove_joints = {
        joint.get("name")
        for joint in joints
        if (joint.find("child") is not None and joint.find("child").get("link") in remove_links)
        or (joint.find("parent") is not None and joint.find("parent").get("link") in remove_links)
    }
    for element in list(robot):
        if element.tag == "link" and element.get("name") in remove_links:
            robot.remove(element)
        elif element.tag == "joint" and element.get("name") in remove_joints:
            robot.remove(element)
    return {"removed_links": sorted(remove_links), "removed_joints": sorted(remove_joints)}


def fixed_joint(name: str, parent: str, child: str, xyz: str, rpy: str) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": rpy})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": "0 0 0"})
    return joint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--mount-mode", choices=("shape_aligned", "no_transform", "tcp_identity", "inward_v1", "inward_v2", "right_flip_v1"), default="shape_aligned")
    args = parser.parse_args()
    if not OLD_URDF.is_file():
        raise FileNotFoundError(OLD_URDF)
    if not HAND_ROOT.is_dir():
        raise FileNotFoundError(HAND_ROOT)
    out_dir = args.output_dir
    out_urdf = out_dir / (("linkhou_xhand_fullbody_right_flip_v1.urdf" if args.mount_mode == "right_flip_v1" else ("linkhou_xhand_fullbody_inward_v2.urdf" if args.mount_mode == "inward_v2" else "linkhou_xhand_fullbody_inward_v1.urdf")) if args.mount_mode in ("inward_v1", "inward_v2", "right_flip_v1") else ("linkhou_xhand_fullbody_tcp_identity_v1.urdf" if args.mount_mode == "tcp_identity" else ("linkhou_xhand_fullbody_no_mount_v1.urdf" if args.mount_mode == "no_transform" else "linkhou_xhand_fullbody_approx_v1.urdf")))
    out_manifest = out_dir / "manifest.json"
    if (out_urdf.exists() or out_manifest.exists()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite {out_dir}; use --force only for an intentional rebuild")

    out_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(OLD_URDF).getroot()
    robot.set("name", "linkhou_xhand_fullbody_approx_v1")
    removed = remove_hand_subtrees(robot, keep_roots=("R_tcp", "L_tcp") if args.mount_mode == "tcp_identity" else ())
    rewrite_meshes(robot, OLD_URDF.parent, original=True)

    hand_roots = {}
    hand_joints = {}
    for side in ("left", "right"):
        hand_urdf = HAND_ROOT / f"xhand1_{side}/urdf/xhand_{side}.urdf"
        hand_robot = ET.parse(hand_urdf).getroot()
        rewrite_meshes(hand_robot, hand_urdf.parent.parent, original=False)
        for element in hand_robot:
            if element.tag == "link":
                robot.append(copy.deepcopy(element))
            elif element.tag == "joint":
                robot.append(copy.deepcopy(element))
        hand_roots[side] = hand_robot.find("link").get("name")
        hand_joints[side] = [
            j.get("name") for j in hand_robot.findall("joint") if j.get("type") != "fixed"
        ]

    # Either preserve the old shape-aligned mount or attach the original XHand
    # palm frame directly to the Tianji arm-end frame, with no extra transform.
    if args.mount_mode in ("no_transform", "tcp_identity"):
        right_xyz, right_rpy = "0 0 0", "0 0 0"
        left_xyz, left_rpy = "0 0 0", "0 0 0"
    elif args.mount_mode in ("inward_v1", "inward_v2", "right_flip_v1"):
        # Start from the shape-aligned Tianji/XHand orientation and move both
        # palms 25 mm toward the object center. This is a deliberate, small
        # opposed-grasp mounting hypothesis, not a replacement hand model.
        inward = 0.025 if args.mount_mode in ("inward_v1", "right_flip_v1") else 0.080
        right_rpy = "-1.57079632679489 0 -1.5707963267949" if args.mount_mode == "right_flip_v1" else "1.57079632679489 0 1.5707963267949"
        right_xyz, right_rpy = f"0.1189997193922 {inward} -0.000151978290524586", right_rpy
        left_xyz, left_rpy = f"0.1189997193922 {-inward} 0.00015197829051407", "-1.57079632679491 0 -1.57079632679489"
    else:
        right_xyz, right_rpy = "0.1189997193922 0 -0.000151978290524586", "1.57079632679489 0 1.5707963267949"
        left_xyz, left_rpy = "0.1189997193922 0 0.00015197829051407", "-1.57079632679491 0 -1.57079632679489"
    right_parent = "R_tcp" if args.mount_mode == "tcp_identity" else "right_j7"
    left_parent = "L_tcp" if args.mount_mode == "tcp_identity" else "left_j7"
    robot.append(fixed_joint(
        "right_hand_mount_xhand_v1", right_parent, "right_hand_link",
        right_xyz, right_rpy,
    ))
    robot.append(fixed_joint(
        "left_hand_mount_xhand_v1", left_parent, "left_hand_link",
        left_xyz, left_rpy,
    ))
    # Stable aliases preserve the existing TRO naming convention. In
    # tcp_identity mode the original arm TCP links are retained as parents.
    if args.mount_mode != "tcp_identity":
        robot.append(ET.Element("link", {"name": "R_tcp"}))
        robot.append(ET.Element("link", {"name": "L_tcp"}))
        robot.append(fixed_joint("R_tcp_alias_xhand_v1", "right_hand_ee_link", "R_tcp", "0 0 0", "0 0 0"))
        robot.append(fixed_joint("L_tcp_alias_xhand_v1", "left_hand_ee_link", "L_tcp", "0 0 0", "0 0 0"))

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(out_urdf, encoding="utf-8", xml_declaration=True)
    manifest = {
        "schema": "linkhou_xhand_fullbody_approx_v1",
        "source_fullbody_urdf": str(OLD_URDF),
        "left_hand_urdf": str(HAND_ROOT / "xhand1_left/urdf/xhand_left.urdf"),
        "right_hand_urdf": str(HAND_ROOT / "xhand1_right/urdf/xhand_right.urdf"),
        "output_urdf": str(out_urdf),
        "mount_mode": args.mount_mode,
        "mounts": {
            "right": {"parent": right_parent, "child": "right_hand_link", "xyz": [0.0, 0.0, 0.0] if args.mount_mode in ("no_transform", "tcp_identity") else ([0.1189997193922, 0.025 if args.mount_mode in ("inward_v1", "right_flip_v1") else 0.080, -0.000151978290524586] if args.mount_mode in ("inward_v1", "inward_v2", "right_flip_v1") else [0.1189997193922, 0.0, -0.000151978290524586]), "rpy": [0.0, 0.0, 0.0] if args.mount_mode in ("no_transform", "tcp_identity") else ([-math.pi / 2, 0.0, -math.pi / 2] if args.mount_mode == "right_flip_v1" else [math.pi / 2, 0.0, math.pi / 2])},
            "left": {"parent": left_parent, "child": "left_hand_link", "xyz": [0.0, 0.0, 0.0] if args.mount_mode in ("no_transform", "tcp_identity") else ([0.1189997193922, -0.025 if args.mount_mode == "inward_v1" else -0.080, 0.00015197829051407] if args.mount_mode in ("inward_v1", "inward_v2") else [0.1189997193922, 0.0, 0.00015197829051407]), "rpy": [0.0, 0.0, 0.0] if args.mount_mode in ("no_transform", "tcp_identity") else [-math.pi / 2, 0.0, -math.pi / 2]},
        },
        "tcp_aliases": {"L_tcp": "left_hand_ee_link", "R_tcp": "right_hand_ee_link"},
        "removed_old_hand": removed,
        "hand_joint_names": hand_joints,
        "fixed_during_ik": ["waist_extend2", "waist_yaw", "waist_pitch", "head_yaw", "head_pitch"],
        "note": "Original XHand v1.3 hand packages; right_flip_v1 tests the mirrored right palm orientation with the left-hand mount rotation." if args.mount_mode == "right_flip_v1" else ("Original XHand v1.3 hand packages; inward_v2 preserves the shape-aligned axes and adds +/-80 mm opposed inward palm offsets." if args.mount_mode == "inward_v2" else ("Original XHand v1.3 hand packages; inward_v1 preserves the shape-aligned axes and adds +/-25 mm opposed inward palm offsets." if args.mount_mode == "inward_v1" else ("Original XHand v1.3 hand packages; tcp_identity attaches palm frame identity to original arm TCP frames." if args.mount_mode == "tcp_identity" else ("Original XHand v1.3 hand packages; no_transform attaches palm frame directly to j7." if args.mount_mode == "no_transform" else "Approximate mount for kinematic smoke only; validate the mount in IsaacLab before physical use.")))),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
