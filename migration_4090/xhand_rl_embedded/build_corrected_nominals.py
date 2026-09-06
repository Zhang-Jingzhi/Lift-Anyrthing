#!/usr/bin/env python3
"""Build corrected, non-accepted nominal warm starts for resized objects.

The locked small sphere and 200% pyramid use different object dimensions from
the historical legacy poses.  This utility recomputes the arm configuration
with the fixed-base IK solver while retaining the hand closure references from
the legacy-derived warm start.  The result is only an RL warm start; it is
never copied into ``accepted/`` and remains subject to the embedded gates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as np
import torch

from migration_4090.generate_xhand_fullbody_grasps_v1 import fixed_solver, solve_group


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ik_to_full_q(solver, q: np.ndarray, source_names: list[str], source_values: list[float]) -> list[float]:
    """Merge IK arm values with the source hand posture in full-body order."""

    ik_names = list(solver.robot.joints.actuated_names)
    by_name = dict(zip(source_names, source_values))
    ik_by_name = dict(zip(ik_names, np.asarray(q, dtype=np.float32).tolist()))
    return [
        float(ik_by_name.get(name, by_name.get(name, 0.0)))
        for name in source_names
    ]


def _solve_targets(solver, targets: np.ndarray, source: dict) -> tuple[list[float], list[list[float]]]:
    """Solve TCP targets and return a full-body q plus actual TCP positions."""

    ik_names = list(solver.robot.joints.actuated_names)
    initial = np.zeros((1, len(ik_names)), dtype=np.float32)
    source_by_name = dict(zip(source["joint_names"], source["full_body_q"]))
    for index, name in enumerate(ik_names):
        if "_hand_" in name:
            initial[0, index] = float(source_by_name.get(name, 0.55))
    solved, poses, actual, errors = solve_group(solver, initial, targets[None])
    if not np.isfinite(solved).all() or float(np.max(errors)) > 0.002:
        raise RuntimeError(f"IK failed for targets={targets.tolist()} errors={errors.tolist()}")
    full_q = _ik_to_full_q(solver, solved[0], source["joint_names"], source["full_body_q"])
    tcp_indices = [solver.robot.links.names.index(name) for name in ("L_tcp", "R_tcp")]
    actual_tcp = np.asarray(jaxlie.SE3(poses[0, tcp_indices]).translation()).tolist()
    return full_q, actual_tcp


def _clean_sample(sample: dict, object_name: str, manifest: dict, source_path: Path) -> dict:
    row = manifest["objects"][object_name]
    extents = [float(value) for value in row["extents_m"]]
    cleaned = copy.deepcopy(sample)
    cleaned["object_name"] = object_name
    cleaned["object_id"] = object_name
    cleaned["base_object_name"] = object_name
    cleaned["object_mesh_path"] = row["mesh"]["path"]
    cleaned["object_transform"] = {
        "scale_mode": "locked_manifest_extents",
        "uniform_scale": 1.0,
        "canonical_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "extents_m": extents,
        "retargeted_from_extents_m": sample.get("object_transform", {}).get("extents_m"),
        "legacy_source": str(source_path.resolve()),
    }
    # The corrected nominal uses an explicit centered object frame.  The mesh
    # assets are centered, so this puts the bottom on the locked table plane.
    z = float(manifest["table_top_z_m"]) + 0.5 * extents[2]
    object_pose = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]
    cleaned["nominal_pose_lock"] = True
    cleaned["nominal_object_position_world"] = [0.5, 0.0, z]
    cleaned["nominal_object_quaternion_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    cleaned["object_pose_world"] = object_pose
    cleaned["commanded_object_pose_world"] = copy.deepcopy(object_pose)
    cleaned["physical_parameters"] = {
        "target_mass_kg": float(manifest["physics"]["object_mass_kg"]),
        "friction": float(manifest["physics"]["friction"]),
        "source_legacy_physical_parameters": sample.get("physical_parameters"),
    }
    for key in (
        "unified_strict_validation", "isaaclab_strict_validation", "strict_acceptance",
        "isaaclab_physical_validated", "isaac_gym_physical_validated",
    ):
        cleaned.pop(key, None)
    cleaned["isaaclab_physical_validated"] = False
    cleaned["isaac_gym_physical_validated"] = False
    cleaned["strict_acceptance"] = False
    cleaned["formal_validation_pending"] = True
    cleaned["generation_backend"] = "corrected_ik_nominal_warm_start_only"
    cleaned["legacy_source_file"] = str(source_path.resolve())
    cleaned["legacy_source_sha256"] = sha256(source_path)
    cleaned["legacy_data_reused_as_accepted"] = False
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    solver, _ = fixed_solver()
    object_names = (
        "sphere", "sphere_small", "cracker_large", "cracker", "pyramid", "cube"
    )
    source_paths = {
        object_name: (args.source_dir / f"{object_name}.pt").resolve()
        for object_name in object_names
    }
    # Object-centered target TCPs are deliberately conservative.  The targets
    # are expressed in the locked canonical frame (x=0.5, y=0) rather than
    # copied from the legacy tilted-object poses.  The y offsets were chosen
    # from the actual mesh half-widths with a small palm/finger allowance;
    # pregrasp adds 120 mm of lateral clearance and lift raises both arms by
    # 60--70 mm, above the 50 mm strict lift threshold.
    target_specs = {
        "sphere": {
            "grasp": np.asarray([[0.5, 0.130, 0.854], [0.5, -0.130, 0.854]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.250, 0.854], [0.5, -0.250, 0.854]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.130, 0.914], [0.5, -0.130, 0.914]], dtype=np.float32),
        },
        "sphere_small": {
            # The small sphere's widest section is at its mesh center.  The
            # former ±130 mm / z=840 mm target was inherited from the large
            # sphere and left both palms outside the 199.5 mm mesh.
            "grasp": np.asarray([[0.5, 0.102, 0.810], [0.5, -0.102, 0.810]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.222, 0.810], [0.5, -0.222, 0.810]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.102, 0.870], [0.5, -0.102, 0.870]], dtype=np.float32),
        },
        "cracker_large": {
            "grasp": np.asarray([[0.5, 0.170, 0.790], [0.5, -0.170, 0.790]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.290, 0.790], [0.5, -0.290, 0.790]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.170, 0.860], [0.5, -0.170, 0.860]], dtype=np.float32),
        },
        "cracker": {
            "grasp": np.asarray([[0.5, 0.145, 0.790], [0.5, -0.145, 0.790]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.265, 0.790], [0.5, -0.265, 0.790]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.145, 0.860], [0.5, -0.145, 0.860]], dtype=np.float32),
        },
        "pyramid": {
            "grasp": np.asarray([[0.5, 0.160, 0.840], [0.5, -0.160, 0.840]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.280, 0.840], [0.5, -0.280, 0.840]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.160, 0.900], [0.5, -0.160, 0.900]], dtype=np.float32),
        },
        "cube": {
            "grasp": np.asarray([[0.5, 0.098, 0.790], [0.5, -0.098, 0.790]], dtype=np.float32),
            "pregrasp": np.asarray([[0.5, 0.218, 0.790], [0.5, -0.218, 0.790]], dtype=np.float32),
            "lift": np.asarray([[0.5, 0.098, 0.860], [0.5, -0.098, 0.860]], dtype=np.float32),
        },
    }
    index = {
        "schema": "xhand_rl_corrected_nominal_index_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "legacy_data_reused_as_accepted": False,
        "formal_validation_pending": True,
        "objects": {},
    }
    for object_name, source_path in source_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples", []) if isinstance(payload, dict) else []
        if not samples:
            raise RuntimeError(f"source nominal has no samples: {source_path}")
        sample = _clean_sample(samples[0], object_name, manifest, source_path)
        specs = target_specs[object_name]
        grasp_q, grasp_actual = _solve_targets(solver, specs["grasp"], sample)
        pre_q, pre_actual = _solve_targets(solver, specs["pregrasp"], sample)
        lift_q, lift_actual = _solve_targets(solver, specs["lift"], sample)
        # Use the source finger closure posture for each phase while replacing
        # the arm joints with the corrected IK solution.
        sample["full_body_q"] = grasp_q
        sample["pregrasp_full_body_q"], _ = _solve_targets(solver, specs["pregrasp"], sample)
        sample["lift_full_body_q"], _ = _solve_targets(solver, specs["lift"], sample)
        # The previous two calls use sample['full_body_q'] as the hand source;
        # explicitly restore source hand values so the phase transitions stay
        # open -> close -> lift instead of inheriting the grasp fingers.
        source_sample = samples[0]
        source_by_name = dict(zip(source_sample["joint_names"], source_sample["pregrasp_full_body_q"]))
        for i, name in enumerate(sample["joint_names"]):
            if "_hand_" in name:
                sample["pregrasp_full_body_q"][i] = float(source_by_name[name])
        source_by_name = dict(zip(source_sample["joint_names"], source_sample["lift_full_body_q"]))
        for i, name in enumerate(sample["joint_names"]):
            if "_hand_" in name:
                sample["lift_full_body_q"][i] = float(source_by_name[name])
        sample["target_tcp_positions_world"] = specs["grasp"].tolist()
        sample["achieved_tcp_positions_world"] = grasp_actual
        sample["pregrasp_target_tcp_positions_world"] = specs["pregrasp"].tolist()
        sample["pregrasp_achieved_tcp_positions_world"] = pre_actual
        sample["lift_target_tcp_positions_world"] = specs["lift"].tolist()
        sample["lift_achieved_tcp_positions_world"] = lift_actual
        sample["corrected_nominal_retarget"] = {
            "schema": "xhand_corrected_nominal_ik_v1",
            "solver": "PyrokiRetarget_fixed_base",
            "target_tcp_positions_world": {
                "pregrasp": specs["pregrasp"].tolist(),
                "grasp": specs["grasp"].tolist(),
                "lift": specs["lift"].tolist(),
            },
            "legacy_source": str(source_path.resolve()),
            "legacy_data_reused_as_accepted": False,
        }
        out_path = output_dir / f"{object_name}.pt"
        out_payload = {
            "schema": "xhand_rl_corrected_nominal_warm_start_v1",
            "backend": "isaaclab_rsl_rl_ppo_nominal_curriculum_v1",
            "object": object_name,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "source_nominal": str(source_path.resolve()),
            "source_nominal_sha256": sha256(source_path),
            "legacy_data_reused_as_accepted": False,
            "strict_validated": False,
            "formal_validation_pending": True,
            "samples": [sample],
        }
        torch.save(out_payload, out_path)
        index["objects"][object_name] = {
            "path": str(out_path),
            "sha256": sha256(out_path),
            "target_tcp_positions_world": specs["grasp"].tolist(),
            "actual_tcp_positions_world": grasp_actual,
        }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
