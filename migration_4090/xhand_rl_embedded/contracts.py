"""Fail-closed contracts for the embedded-physics RL pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EMBEDDED_BACKEND = "isaaclab_rsl_rl_ppo_embedded_physics_v2"
MANIFEST_SCHEMA = "xhand_locked_six_rl_embedded_pipeline_v2"
ROOT_MARKER_SCHEMA = "xhand_rl_embedded_generation_root_v2"
RECEIPT_SCHEMA = "xhand_rl_embedded_acceptance_receipt_v2"
TRAINING_RUN_SCHEMA = "xhand_rl_embedded_training_run_v2"
SINGLE_GPU_KIT_ARGS = "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
KIT_SETTINGS_PROFILE = "single_gpu_isolated_portable_v2"
REWARD_REVISION = "meshaware_closed_hand_seed_residual_v5"
PPO_REVISION = "fixed_lr_scaled_return_v4"
TRAINING_REWARD_SCALE = 0.01

LOCKED_OBJECTS = (
    "sphere",
    "sphere_small",
    "cracker_large",
    "cracker",
    "pyramid",
    "cube",
)

EMBEDDED_GATE_NAMES = (
    "asset_hashes",
    "rl_provenance",
    "embedded_isaaclab_episode",
    "finite_state",
    "bilateral_contact_continuity",
    "arm_joint_lift",
    "lift_height",
    "gravity_hold",
    "translation_disturbance",
    "rotation_disturbance",
    "physx_penetration",
    "formal_force_closure",
    "single_hand_ablations",
    "unique_pose",
    "trajectory_recorded",
)

OMITTED_EXPENSIVE_GATES = (
    "multistage_visual_mesh",
    "visual_mesh_table_clearance",
)


class ContractError(RuntimeError):
    """Raised when a v2 artifact does not meet its declared contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ContractError(f"invalid RSL-RL checkpoint name: {path.name}")
    return int(match.group(1))


def resume_checkpoint_iteration(path: Path) -> int:
    """Return the source iteration for a safe training-resume checkpoint.

    Production checkpoint selection intentionally remains restricted to the
    canonical ``model_<iteration>.pt`` form via :func:`checkpoint_iteration`.
    Training may additionally resume from an exact policy snapshot captured
    before the PPO update that produced a strict-prefix or hard-success
    rollout.  Those snapshots encode the source iteration in their filename.
    """

    patterns = (
        r"model_(\d+)\.pt",
        r"hard_prefix_preupdate_iter_(\d+)_level_\d+\.pt",
        r"hard_success_preupdate_iter_(\d+)_serial_\d+\.pt",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, path.name)
        if match is not None:
            return int(match.group(1))
    raise ContractError(f"invalid training-resume checkpoint name: {path.name}")


def latest_policy_checkpoint(directory: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if directory.exists():
        for path in directory.glob("model_*.pt"):
            try:
                candidates.append((checkpoint_iteration(path), path))
            except ContractError:
                continue
    return max(candidates, default=(None, None), key=lambda row: row[0])[1]


def validate_precheckpoint_restart(
    directory: Path,
    *,
    manifest_path: Path,
    nominal_dataset: Path,
    object_name: str,
    nominal_sample_index: int,
    seed: int,
    num_envs: int,
    max_iterations: int,
    device: str,
    kit_settings_profile: str,
    renderer_multi_gpu: bool,
) -> None:
    """Allow only an identical interrupted v2 run before its first checkpoint."""
    entries = list(directory.iterdir()) if directory.exists() else []
    if not entries:
        return
    allowed_names = {"training_provenance.json", "locked_embedded_manifest.json"}
    unexpected = sorted(
        path.name
        for path in entries
        if not (
            path.is_file()
            and (
                path.name in allowed_names
                or path.name.startswith("events.out.tfevents.")
                or re.fullmatch(r"training_provenance\.precheckpoint_retry_\d+\.json", path.name)
            )
        )
    )
    _require(not unexpected, f"unsafe files in pre-checkpoint training root: {unexpected}")
    _require(latest_policy_checkpoint(directory) is None, "checkpoint exists but resume was not requested")
    locked_manifest = directory / "locked_embedded_manifest.json"
    if locked_manifest.is_file():
        _require(
            sha256_file(locked_manifest) == sha256_file(manifest_path),
            "pre-checkpoint locked manifest changed",
        )
    provenance_paths = sorted(directory.glob("training_provenance*.json"))
    _require(provenance_paths, "non-empty pre-checkpoint root has no training provenance")
    expected = {
        "schema": TRAINING_RUN_SCHEMA,
        "backend": EMBEDDED_BACKEND,
        "object": object_name,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "nominal_dataset": str(nominal_dataset.resolve()),
        "nominal_dataset_sha256": sha256_file(nominal_dataset),
        "nominal_sample_index": nominal_sample_index,
        "seed": seed,
        "num_envs": num_envs,
        "max_iterations": max_iterations,
        "device": device,
        "kit_settings_profile": kit_settings_profile,
        "renderer_multi_gpu": renderer_multi_gpu,
        "reward_revision": REWARD_REVISION,
        "ppo_revision": PPO_REVISION,
        "training_reward_scale": TRAINING_REWARD_SCALE,
        "resume_checkpoint_iteration": None,
        "start_iteration": 0,
    }
    for path in provenance_paths:
        try:
            provenance = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"invalid pre-checkpoint training provenance: {path.name}") from error
        mismatched = sorted(key for key, value in expected.items() if provenance.get(key) != value)
        _require(not mismatched, f"pre-checkpoint training identity changed: {mismatched}")
        _require("completed_at" not in provenance, "completed provenance has no checkpoint")


def validate_manifest(payload: dict[str, Any], *, verify_files: bool = True) -> None:
    _require(payload.get("schema") == MANIFEST_SCHEMA, "wrong embedded manifest schema")
    algorithm = payload.get("algorithm", {})
    _require(algorithm.get("backend") == EMBEDDED_BACKEND, "wrong embedded backend")
    _require(algorithm.get("family") == "reinforcement_learning", "v2 must remain RL")
    _require(algorithm.get("trainer") == "rsl_rl.ppo", "v2 requires RSL-RL PPO")
    _require(
        algorithm.get("reward_revision") == REWARD_REVISION,
        "wrong embedded reward revision",
    )
    _require(algorithm.get("ppo_revision") == PPO_REVISION, "wrong embedded PPO revision")
    _require(algorithm.get("ppo_schedule") == "fixed", "embedded PPO schedule must be fixed")
    _require(
        float(algorithm.get("ppo_learning_rate", 0.0)) == 3.0e-4,
        "wrong embedded PPO learning rate",
    )
    _require(
        float(algorithm.get("training_reward_scale", 0.0)) == TRAINING_REWARD_SCALE,
        "wrong embedded training reward scale",
    )
    _require(algorithm.get("bodex_algorithm") is False, "BODex must not be relabeled as RL")
    _require(
        payload.get("acceptance_profile") == "embedded_physics_pass",
        "wrong v2 acceptance profile",
    )
    _require(payload.get("strict_visual_mesh_accepted") is False, "v2 cannot claim visual acceptance")
    _require(payload.get("target_per_object") == 100, "target must remain 100 per object")
    _require(tuple(payload.get("objects", {})) == LOCKED_OBJECTS, "locked object order changed")
    _require(
        tuple(payload.get("required_embedded_gates", ())) == EMBEDDED_GATE_NAMES,
        "embedded gate set changed",
    )
    _require(
        tuple(payload.get("omitted_expensive_gates", ())) == OMITTED_EXPENSIVE_GATES,
        "omitted gate declaration changed",
    )
    physics = payload.get("physics", {})
    for key in (
        "dt_s",
        "gravity_m_s2",
        "friction",
        "object_mass_kg",
        "position_iterations",
        "velocity_iterations",
        "contact_offset_m",
        "rest_offset_m",
    ):
        _require(key in physics, f"missing physics value: {key}")
    thresholds = payload.get("embedded_thresholds", {})
    for key in (
        "lift_min_m",
        "lift_reward_target_m",
        "gravity_drift_max_m",
        "translation_disturbance_max_m",
        "rotation_disturbance_max_rad",
        "contact_presence_fraction_min",
        "inactive_hand_contact_fraction_max",
        "physx_penetration_max_m",
        "force_closure_residual_max",
        "force_closure_epsilon_min",
        "disturbance_force_n",
        "disturbance_torque_nm",
    ):
        _require(key in thresholds, f"missing embedded threshold: {key}")
    schedule = payload.get("episode_schedule", {})
    fractions = schedule.get("fractions", {})
    _require(
        tuple(fractions) == ("approach", "close", "lift", "hold", "disturbance", "ablation"),
        "episode phase order changed",
    )
    _require(abs(sum(float(value) for value in fractions.values()) - 1.0) < 1.0e-9, "phase fractions must sum to one")
    _require(int(schedule.get("disturbance_count", 0)) == 12, "all 12 wrench disturbances are required")
    _require(int(schedule.get("single_hand_ablation_count", 0)) == 2, "both single-hand ablations are required")
    if not verify_files:
        return
    assets = {
        "robot_usd": payload.get("robot_usd", {}),
        "robot_visual_urdf": payload.get("robot_visual_urdf", {}),
    }
    for name, row in payload["objects"].items():
        assets[f"{name}.mesh"] = row.get("mesh", {})
        assets[f"{name}.usd"] = row.get("usd", {})
    for label, row in assets.items():
        path = Path(row.get("path", ""))
        _require(path.is_file(), f"missing locked asset: {label}: {path}")
        _require(bool(row.get("sha256")), f"missing asset hash: {label}")
        _require(sha256_file(path) == row["sha256"], f"asset hash mismatch: {label}")


def validate_root(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    marker_path = root / "EMBEDDED_RL_PIPELINE_ROOT.json"
    _require(marker_path.is_file(), f"missing embedded root marker: {marker_path}")
    marker = json.loads(marker_path.read_text())
    _require(marker.get("schema") == ROOT_MARKER_SCHEMA, "wrong embedded root schema")
    _require(marker.get("backend") == EMBEDDED_BACKEND, "embedded root backend mismatch")
    _require(marker.get("manifest") == str(manifest_path.resolve()), "embedded root manifest mismatch")
    _require(marker.get("manifest_sha256") == sha256_file(manifest_path), "manifest hash mismatch")
    _require(marker.get("reward_revision") == REWARD_REVISION, "reward revision mismatch")
    _require(marker.get("ppo_revision") == PPO_REVISION, "PPO revision mismatch")
    _require(
        marker.get("training_reward_scale") == TRAINING_REWARD_SCALE,
        "training reward scale mismatch",
    )
    _require(tuple(marker.get("objects", ())) == LOCKED_OBJECTS, "embedded root objects changed")
    _require(marker.get("target_per_object") == manifest["target_per_object"], "embedded root target changed")
    _require(marker.get("legacy_data_reused_as_accepted") is False, "legacy data reuse is forbidden")
    _require(marker.get("strict_visual_mesh_accepted") is False, "root falsely claims visual acceptance")
    return marker


def validate_gate_report(report: dict[str, Any]) -> None:
    gates = report.get("gates", {})
    _require(set(gates) == set(EMBEDDED_GATE_NAMES), "embedded gate report is incomplete")
    failed = sorted(name for name, value in gates.items() if value is not True)
    _require(not failed, f"embedded gates failed: {failed}")
    omitted = report.get("omitted_expensive_gates", [])
    _require(tuple(omitted) == OMITTED_EXPENSIVE_GATES, "omitted gates are not explicit")
    _require(report.get("strict_visual_mesh_accepted") is False, "report falsely claims visual acceptance")


def validate_receipt(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "wrong embedded receipt schema")
    _require(receipt.get("backend") == EMBEDDED_BACKEND, "receipt is not from v2 RL")
    _require(receipt.get("acceptance_profile") == "embedded_physics_pass", "wrong acceptance profile")
    _require(receipt.get("object") in manifest["objects"], "unknown locked object")
    _require(receipt.get("accepted") is True, "receipt is not accepted")
    _require(receipt.get("visual_mesh_audit_pending") is True, "visual audit status must remain pending")
    _require(receipt.get("policy_checkpoint_sha256"), "missing policy checkpoint provenance")
    _require(
        receipt.get("reward_revision") == manifest["algorithm"]["reward_revision"],
        "receipt reward revision mismatch",
    )
    _require(
        receipt.get("ppo_revision") == manifest["algorithm"]["ppo_revision"],
        "receipt PPO revision mismatch",
    )
    _require(receipt.get("trajectory_sha256"), "missing recorded trajectory hash")
    validate_gate_report(receipt)
