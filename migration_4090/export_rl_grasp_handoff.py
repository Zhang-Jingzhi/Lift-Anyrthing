#!/usr/bin/env python3
"""Create a portable, non-destructive RL handoff bundle.

The bundle contains the exact left/right Allegro assets, object assets matching
all historical repeat-verified bimanual poses, the two target objects that do
not yet have poses, and flattened PT/NPZ/JSONL pose exports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation


TARGET_BASE_OBJECTS = (
    "ycb+bleach_cleanser",
    "ycb+cracker_box",
    "ycb+pitcher_base",
    "contactdb+piggy_bank",
    "ycb+power_drill",
    "ycb+toy_airplane",
)

JOINT_ORDER = (
    "virtual_joint_x",
    "virtual_joint_y",
    "virtual_joint_z",
    "virtual_joint_roll",
    "virtual_joint_pitch",
    "virtual_joint_yaw",
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
)

POSE_KEYS = (
    "left_q",
    "right_q",
    "left_q_seed",
    "right_q_seed",
    "left_q_outer",
    "right_q_outer",
    "left_q_command",
    "right_q_command",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def classify_method(path: Path) -> str | None:
    lower = str(path).lower()
    if "bidex" in lower:
        return "bidex"
    if "baseline" in lower:
        return "baseline"
    return None


def pose_digest(sample: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(sample["object_name"].encode("utf-8"))
    for key in ("left_q", "right_q"):
        value = sample[key].detach().cpu().to(torch.float32).contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def object_asset_relpath(object_name: str) -> Path:
    dataset, name = object_name.split("+", 1)
    return Path("data/data_urdf/object") / dataset / name


def root_components(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = q[:, :3].astype(np.float32, copy=False)
    rotation = Rotation.from_euler("XYZ", q[:, 3:6])
    quaternion_xyzw = rotation.as_quat().astype(np.float32)
    transform = np.repeat(np.eye(4, dtype=np.float32)[None], len(q), axis=0)
    transform[:, :3, :3] = rotation.as_matrix().astype(np.float32)
    transform[:, :3, 3] = translation
    return translation, quaternion_xyzw, transform


def copy_tree_exact(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    roots = (
        repo / "graph_exp/bimanual_data",
        repo / "migration_4090/renders",
        repo / "migration_4090/results",
    )
    dataset_paths = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("verified_dataset.pt")
        if classify_method(path) is not None
    )
    if not dataset_paths:
        raise RuntimeError("No baseline/BiDex verified_dataset.pt files found")

    unique: dict[str, dict[str, Any]] = {}
    source_manifests: dict[str, Any] = {}
    raw_counts = Counter()
    for dataset_path in dataset_paths:
        method = classify_method(dataset_path)
        assert method is not None
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise TypeError(f"Unexpected dataset structure: {dataset_path}")
        relative_source = str(dataset_path.relative_to(repo))
        source_manifests[relative_source] = jsonable(payload.get("manifest", {}))
        for source_index, original in enumerate(payload["samples"]):
            if not bool(original.get("metrics", {}).get("strict_success", False)):
                continue
            for key in ("left_q", "right_q"):
                if tuple(original[key].shape) != (22,):
                    raise ValueError(f"{dataset_path}:{source_index} {key} is not [22]")
                if not torch.isfinite(original[key]).all():
                    raise ValueError(f"{dataset_path}:{source_index} {key} is not finite")
            raw_counts[method] += 1
            digest = pose_digest(original)
            if digest not in unique:
                record: dict[str, Any] = {
                    "object_name": str(original["object_name"]),
                    "methods": set(),
                    "source_records": [],
                    "sample": copy.deepcopy(original),
                }
                for key in POSE_KEYS:
                    if key in record["sample"]:
                        record["sample"][key] = (
                            record["sample"][key]
                            .detach()
                            .cpu()
                            .to(torch.float32)
                            .contiguous()
                        )
                unique[digest] = record
            unique[digest]["methods"].add(method)
            unique[digest]["source_records"].append(
                {"dataset": relative_source, "sample_index": source_index, "method": method}
            )

    records = []
    for ordinal, (digest, item) in enumerate(
        sorted(unique.items(), key=lambda pair: (pair[1]["object_name"], pair[0]))
    ):
        methods = sorted(item["methods"])
        sample = item["sample"]
        record = {
            "sample_id": f"grasp_{ordinal:06d}",
            "pose_sha256": digest,
            "method": methods[0] if len(methods) == 1 else "+".join(methods),
            "object_name": item["object_name"],
            "source_records": item["source_records"],
            **sample,
        }
        records.append(record)

    methods = sorted({record["method"] for record in records})
    object_names = sorted({record["object_name"] for record in records})
    object_index = {name: index for index, name in enumerate(object_names)}
    method_index = {name: index for index, name in enumerate(methods)}

    hand_assets = {
        "left": Path("data/data_urdf/robot/allegro"),
        "right": Path("data/data_urdf/robot/allegro_right"),
    }
    for relative in hand_assets.values():
        copy_tree_exact(repo / relative, output / relative)

    asset_names = sorted(set(object_names).union(TARGET_BASE_OBJECTS))
    object_assets: dict[str, Any] = {}
    for name in asset_names:
        relative = object_asset_relpath(name)
        copy_tree_exact(repo / relative, output / relative)
        directory = output / relative
        meshes = sorted(
            str(path.relative_to(output))
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".obj", ".stl", ".ply"}
        )
        urdfs = sorted(
            str(path.relative_to(output))
            for path in directory.rglob("*.urdf")
        )
        object_assets[name] = {
            "asset_directory": str(relative),
            "mesh_files": meshes,
            "urdf_files": urdfs,
            "pose_count": sum(record["object_name"] == name for record in records),
            "target_base_object": name in TARGET_BASE_OBJECTS,
        }

    grasps_dir = output / "grasps"
    grasps_dir.mkdir()
    torch_payload = {
        "manifest": {
            "schema": "tro_grasp_rl_handoff_v1",
            "pose_frame": "object_local",
            "q_layout": list(JOINT_ORDER),
            "q_units": {"translation": "metre", "rotation_and_joints": "radian"},
            "q_rotation": "intrinsic Euler XYZ",
            "preferred_target": "left_q/right_q (realized settled pose)",
            "pose_count": len(records),
            "historical_protocol_warning": (
                "These are historical repeat-verified poses. They are not the unfinished "
                "formal 6x100 high-resolution VHACD dataset."
            ),
        },
        "samples": records,
    }
    torch.save(torch_payload, grasps_dir / "grasp_poses.pt")

    arrays: dict[str, np.ndarray] = {}
    for key in POSE_KEYS:
        arrays[key] = np.stack(
            [
                record[key].numpy()
                if key in record
                else np.full(22, np.nan, dtype=np.float32)
                for record in records
            ]
        ).astype(np.float32)
    arrays["object_index"] = np.asarray(
        [object_index[record["object_name"]] for record in records], dtype=np.int32
    )
    arrays["method_index"] = np.asarray(
        [method_index[record["method"]] for record in records], dtype=np.int32
    )
    arrays["sample_id"] = np.asarray([record["sample_id"] for record in records])
    for side in ("left", "right"):
        q = arrays[f"{side}_q"]
        translation, quaternion, transform = root_components(q)
        arrays[f"{side}_root_translation_m"] = translation
        arrays[f"{side}_root_quaternion_xyzw"] = quaternion
        arrays[f"{side}_object_T_hand"] = transform
        arrays[f"{side}_joint_position_rad"] = q[:, 6:]
    np.savez_compressed(grasps_dir / "grasp_poses.npz", **arrays)

    with (grasps_dir / "grasp_poses.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            compact = {
                "sample_id": record["sample_id"],
                "pose_sha256": record["pose_sha256"],
                "method": record["method"],
                "object_name": record["object_name"],
                **{key: jsonable(record.get(key)) for key in POSE_KEYS},
                "metrics": jsonable(record.get("metrics", {})),
                "source_records": record["source_records"],
            }
            handle.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")

    schema = {
        "schema": "tro_grasp_rl_handoff_v1",
        "coordinate_convention": {
            "pose_frame": "object local frame; meshes use the same local frame",
            "q_shape": [22],
            "q_0_2": "hand root xyz translation in metres",
            "q_3_5": "hand root intrinsic Euler XYZ rotation in radians",
            "q_6_21": "16 Allegro actuated joints in URDF/pytorch_kinematics order",
            "quaternion_order": "xyzw",
            "object_T_hand": "4x4 transform mapping hand-local points into object frame",
        },
        "joint_order": list(JOINT_ORDER),
        "pose_fields": {
            "left_q/right_q": "realized pose after the verified bimanual rollout; recommended RL target",
            "left_q_seed/right_q_seed": "initial pose used to start the physics rollout",
            "left_q_outer/right_q_outer": "open/approach pose",
            "left_q_command/right_q_command": "simultaneous closing position target",
        },
        "objects": object_assets,
        "object_index": object_names,
        "method_index": methods,
    }
    (grasps_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (grasps_dir / "source_datasets.json").write_text(
        json.dumps(source_manifests, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    counts_by_method = Counter(record["method"] for record in records)
    counts_by_object = Counter(record["object_name"] for record in records)
    counts_by_method_object: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts_by_method_object[record["method"]][record["object_name"]] += 1
    manifest = {
        "schema": "tro_grasp_rl_handoff_v1",
        "source_repository": str(repo),
        "hands": {
            "left_urdf": "data/data_urdf/robot/allegro/allegro_hand_left_extended.urdf",
            "right_urdf": "data/data_urdf/robot/allegro_right/allegro_hand_right_extended.urdf",
            "distinct_left_and_right_assets": True,
        },
        "verified_dataset_file_count": len(dataset_paths),
        "raw_strict_pose_count": dict(raw_counts),
        "unique_pose_count": len(records),
        "unique_pose_count_by_method": dict(sorted(counts_by_method.items())),
        "unique_pose_count_by_object": dict(sorted(counts_by_object.items())),
        "unique_pose_count_by_method_object": {
            method: dict(sorted(counts.items()))
            for method, counts in sorted(counts_by_method_object.items())
        },
        "target_base_objects_without_pose": [
            name for name in TARGET_BASE_OBJECTS if counts_by_object[name] == 0
        ],
        "historical_protocol_warning": (
            "The formal high-resolution VHACD 6 objects x 100 samples per method job is "
            "paused and has no final samples. These poses passed their source historical "
            "strict/repeat verifier but must not be counted toward that final dataset."
        ),
        "object_assets": object_assets,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readme = f"""# TRO-Grasp RL handoff

This is a portable snapshot for implementing the existing bimanual targets in RL.
It does not modify or depend on the original result directories.

## Contents

- Real Allegro left URDF: `data/data_urdf/robot/allegro/allegro_hand_left_extended.urdf`
- Validated mirrored Allegro right URDF: `data/data_urdf/robot/allegro_right/allegro_hand_right_extended.urdf`
- Exact URDF mesh dependencies are inside the two adjacent `meshes/` directories.
- Matching object meshes/URDFs: `data/data_urdf/object/`
- PyTorch poses with full metrics and provenance: `grasps/grasp_poses.pt`
- Framework-neutral NumPy poses: `grasps/grasp_poses.npz`
- Line-oriented human-readable poses: `grasps/grasp_poses.jsonl`
- Coordinate, joint and object mapping: `grasps/schema.json`

## Pose convention

Each hand pose has 22 values in the exact URDF kinematic order:

`[x, y, z, intrinsic_X, intrinsic_Y, intrinsic_Z, joint_0.0, ..., joint_15.0]`

Translation is in metres; rotations and finger joints are in radians. The root
pose is expressed in the object's local mesh frame. In the NPZ export,
`left_object_T_hand` / `right_object_T_hand` are ready-to-use 4x4 transforms,
and quaternions use XYZW order.

For imitation targets, use `left_q` / `right_q`: these are the realized settled
poses from verified rollouts. `*_outer` is the open approach, `*_seed` is the
rollout initialization, and `*_command` is the simultaneous closing target.

## Counts and limitation

The bundle has {len(records)} unique historical strict/repeat-verified poses:
{json.dumps(dict(sorted(counts_by_method.items())), ensure_ascii=False)}.

The newer formal high-resolution VHACD 6x100-per-method dataset is still paused
and currently has zero final samples. Historical poses in this handoff must not
be reported as that unfinished dataset. The target objects with meshes but no
current strict pose are: {', '.join(name for name in TARGET_BASE_OBJECTS if counts_by_object[name] == 0)}.

Run `python load_example.py` from this directory for a dependency-light sanity check.
"""
    (output / "README_RL_HANDOFF.md").write_text(readme, encoding="utf-8")

    loader = '''#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
schema = json.loads((root / "grasps/schema.json").read_text())
data = np.load(root / "grasps/grasp_poses.npz")
assert data["left_q"].shape[1] == data["right_q"].shape[1] == 22
assert np.isfinite(data["left_q"]).all() and np.isfinite(data["right_q"]).all()
i = 0
print("samples:", len(data["left_q"]))
print("sample_id:", data["sample_id"][i])
print("object:", schema["object_index"][int(data["object_index"][i])])
print("method:", schema["method_index"][int(data["method_index"][i])])
print("left object_T_hand:\\n", data["left_object_T_hand"][i])
print("right object_T_hand:\\n", data["right_object_T_hand"][i])
print("left joints [rad]:", data["left_joint_position_rad"][i])
print("right joints [rad]:", data["right_joint_position_rad"][i])
'''
    (output / "load_example.py").write_text(loader, encoding="utf-8")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksum_lines = [f"{sha256(path)}  {path.relative_to(output)}" for path in files]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
