#!/usr/bin/env python3
"""Fit cuRobo collision spheres and emit a BODex-compatible dual-XHand config."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml


PALM_LINKS = ("left_hand_ee_link", "right_hand_ee_link")

# BODex's first 6D axis is the inward palm normal.  Both XHand palm links use
# local +X for that same axis, so no additional link-frame rotation is needed.
# The asymmetric reachable wrist roll is expressed explicitly by the two
# tangent references in paired_surface.py instead of being hidden in this
# transfer matrix.
BODEX_TO_XHAND_LINK_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
BODEX_TO_XHAND_LINK_TRANSLATION = (0.0, 0.0, 0.0)
CONTACT_MESH_LINKS = tuple(
    f"{side}_hand_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "index_rota_link2",
        "mid_link2",
        "ring_link2",
        "pinky_link2",
        "thumb_rota_link2",
    )
)
CONTACT_POINT_LINKS = tuple(
    f"{side}_hand_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "index_rota_tip",
        "mid_tip",
        "ring_tip",
        "pinky_tip",
        "thumb_rota_tip",
    )
)

# RobotBuilder in newer cuRobo releases emits these fields, while the pinned
# BODex cuRobo fork predates them. None is required by BODex's
# CudaRobotGeneratorConfig; contacts and transferred palms are supplied below
# through the fields that exist in the pinned fork.
BODEX_UNSUPPORTED_KINEMATICS_KEYS = (
    "format_version",
    "grasp_contact_link_names",
    "load_tool_frames_with_mesh",
    "tool_frames",
)


def select_contact_point_sphere_index(spheres: list[dict]) -> int:
    """Select the small marker sphere centered at a ``*_tip`` link origin."""

    rows = []
    for index, sphere in enumerate(spheres):
        center = sphere.get("center")
        radius = float(sphere.get("radius", 0.0))
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError("contact-point sphere center must contain xyz")
        center_norm = math.sqrt(sum(float(value) ** 2 for value in center))
        if not math.isfinite(center_norm) or not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("contact-point sphere must be finite and positive")
        rows.append((index, center_norm, radius))
    if not rows:
        raise ValueError("contact-point link has no fitted spheres")
    marker_rows = [
        row for row in rows if row[1] <= 0.01 and 0.002 <= row[2] <= 0.01
    ]
    if not marker_rows:
        marker_rows = [row for row in rows if row[1] <= 0.01]
    if not marker_rows:
        marker_rows = rows
    return max(marker_rows, key=lambda row: (row[2], -row[1]))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--urdf", type=Path)
    source.add_argument(
        "--builder-config",
        type=Path,
        help="reuse a previously fitted RobotBuilder YAML and only rebuild the BODex adapter",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sphere-density", type=float, default=0.75)
    parser.add_argument("--self-collision-samples", type=int, default=1000)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_config = output / "xhand_fullbody_builder.yml"
    transfer_config = output / "xhand_bimanual_hand_pose_transfer.yml"
    final_config = output / "xhand_fullbody_bodex.yml"
    if any(path.exists() for path in (raw_config, transfer_config, final_config)):
        raise FileExistsError("refusing to overwrite an existing XHand BODex config")
    if args.builder_config is not None:
        if not args.builder_config.is_file():
            raise FileNotFoundError(args.builder_config)
        # This mode changes only the BODex-specific compatibility and palm
        # transfer layers.  The expensive fitted collision geometry remains
        # byte-for-byte attributable to the supplied RobotBuilder output.
        raw_config.write_text(args.builder_config.read_text())
    else:
        from curobo.robot_builder import RobotBuilder
        from curobo.types import DeviceCfg

        device_cfg = DeviceCfg(device=torch.device(args.device), dtype=torch.float32)
        builder = RobotBuilder(
            str(args.urdf.resolve()),
            asset_path="",
            tool_frames=list(PALM_LINKS),
            device_cfg=device_cfg,
        )
        builder.fit_collision_spheres(
            sphere_density=args.sphere_density,
            compute_metrics=True,
        )
        builder.compute_collision_matrix(
            prune_collisions=True, num_samples=args.self_collision_samples
        )
        config = builder.build()
        builder.save(config, str(raw_config))
    transfer = {
        link: {
            "r": [list(row) for row in BODEX_TO_XHAND_LINK_ROTATION],
            "t": list(BODEX_TO_XHAND_LINK_TRANSLATION),
        }
        for link in PALM_LINKS
    }
    transfer_config.write_text(yaml.safe_dump(transfer, sort_keys=False))
    generated = yaml.safe_load(raw_config.read_text())
    kinematics = generated.get("kinematics", generated)
    for key in BODEX_UNSUPPORTED_KINEMATICS_KEYS:
        kinematics.pop(key, None)
    cspace = kinematics.get("cspace", {})
    if "default_joint_position" in cspace and "retract_config" not in cspace:
        cspace["retract_config"] = cspace.pop("default_joint_position")
    cspace.pop("null_space_maximum_distance", None)
    collision_spheres = kinematics.get("collision_spheres", {})
    for link in CONTACT_POINT_LINKS:
        spheres = collision_spheres.get(link)
        if not isinstance(spheres, list) or not spheres:
            raise RuntimeError(f"contact-point marker has no fitted spheres: {link}")
        selected = select_contact_point_sphere_index(spheres)
        collision_spheres[link] = [spheres[selected]]
    kinematics["hand_pose_transfer_path"] = [str(transfer_config)]
    # BODex has two distinct contact representations. The tiny ``*_tip``
    # marker links provide point centers for the sphere/ESDF contact
    # constraint, while the physical ``*_link2`` meshes provide GJK contact
    # positions, normals and force closure. Conflating them makes the final
    # exact-contact constraint inconsistent with the physical mesh objective.
    kinematics["link_names"] = list(
        dict.fromkeys((*PALM_LINKS, *CONTACT_MESH_LINKS, *CONTACT_POINT_LINKS))
    )
    kinematics["ee_link"] = PALM_LINKS[0]
    kinematics["contact_mesh_names"] = list(CONTACT_MESH_LINKS)
    final_config.write_text(
        yaml.safe_dump({"robot_cfg": {"kinematics": kinematics}}, sort_keys=False)
    )
    print(final_config)


if __name__ == "__main__":
    main()
