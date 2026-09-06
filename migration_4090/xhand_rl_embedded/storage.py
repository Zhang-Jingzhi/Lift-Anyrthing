"""Atomic storage and indexes for v2 embedded-physics successes."""

from __future__ import annotations

import copy
import csv
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    EMBEDDED_BACKEND,
    EMBEDDED_GATE_NAMES,
    OMITTED_EXPENSIVE_GATES,
    RECEIPT_SCHEMA,
    sha256_file,
    validate_gate_report,
    validate_manifest,
    validate_receipt,
    validate_root,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def accepted_count(root: Path, object_name: str) -> int:
    return len(list((root / "accepted" / object_name).glob("sample_*/receipt.json")))


def _validate_trajectory_evidence(
    trajectory: dict[str, Any], episode_report: dict[str, Any]
) -> None:
    if trajectory.get("schema") != "xhand_rl_embedded_actual_trajectory_v2":
        raise RuntimeError("wrong embedded trajectory schema")
    if trajectory.get("backend") != EMBEDDED_BACKEND:
        raise RuntimeError("trajectory backend is not v2 embedded RL")
    states = trajectory.get("states")
    required = ("joint_positions", "object_root_state", "policy_actions", "phase_code")
    if not isinstance(states, dict) or set(states) != set(required):
        raise RuntimeError("embedded trajectory state set is incomplete")
    tensors = {name: states[name] for name in required}
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RuntimeError("embedded trajectory states must be tensors")
    lengths = {int(value.shape[0]) for value in tensors.values() if value.ndim >= 1}
    if len(lengths) != 1 or any(value.ndim < 1 for value in tensors.values()):
        raise RuntimeError("embedded trajectory state lengths differ")
    (steps,) = lengths
    if steps <= 0 or steps != int(episode_report.get("evaluated_policy_steps", -1)):
        raise RuntimeError("embedded trajectory does not cover the audited episode")
    if episode_report.get("final_audited_policy_step") != steps:
        raise RuntimeError("terminal state was not audited before acceptance")
    joint_count = len(trajectory.get("joint_names", ()))
    if tensors["joint_positions"].shape != (steps, joint_count):
        raise RuntimeError("joint trajectory shape does not match joint names")
    if tensors["policy_actions"].shape != (steps, joint_count):
        raise RuntimeError("action trajectory shape does not match joint names")
    if tensors["object_root_state"].shape != (steps, 13):
        raise RuntimeError("object trajectory must contain 13D root states")
    if tensors["phase_code"].shape != (steps,):
        raise RuntimeError("phase trajectory must be one-dimensional")
    if any(not torch.isfinite(value).all().item() for value in tensors.values()):
        raise RuntimeError("embedded trajectory contains non-finite values")
    phase_values = {int(value) for value in tensors["phase_code"].tolist()}
    if phase_values != set(range(18)) or int(tensors["phase_code"][-1].item()) != 17:
        raise RuntimeError("embedded trajectory does not cover all physical test phases")
    simulation_dt = float(trajectory.get("simulation_dt_s", 0.0))
    policy_dt = float(trajectory.get("policy_dt_s", 0.0))
    if simulation_dt <= 0.0 or policy_dt < simulation_dt:
        raise RuntimeError("invalid embedded trajectory timing")


def _accepted_samples(accepted_root: Path) -> list[dict[str, Any]]:
    samples = []
    for path in sorted(accepted_root.glob("*/sample_*/sample.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples.append(payload["samples"][0])
    return samples


def _write_indexes(accepted_root: Path, manifest: dict[str, Any]) -> None:
    samples = _accepted_samples(accepted_root)
    object_names = list(manifest["objects"])
    for name in object_names:
        selected = [sample for sample in samples if sample["object_name"] == name]
        object_root = accepted_root / name
        object_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"schema": "xhand_rl_embedded_object_dataset_v2", "object": name, "samples": selected},
            object_root / "accepted.pt",
        )
        (object_root / "accepted.json").write_text(json.dumps(jsonable(selected), indent=2) + "\n")
    torch.save(
        {"schema": "xhand_rl_embedded_dataset_v2", "samples": samples},
        accepted_root / "all_accepted.pt",
    )
    (accepted_root / "all_accepted.json").write_text(json.dumps(jsonable(samples), indent=2) + "\n")
    rows = [
        {
            "object": sample["object_name"],
            "pose_sha256": sample["embedded_rl_validation"]["pose_sha256"],
            "policy_checkpoint_sha256": sample["policy_checkpoint_sha256"],
            "sample_pt": sample["embedded_rl_validation"]["sample_pt"],
            "receipt": sample["embedded_rl_validation"]["receipt"],
            "trajectory_pt": sample["embedded_rl_validation"]["trajectory_pt"],
            "trajectory_json": sample["embedded_rl_validation"]["trajectory_json"],
            "acceptance_profile": "embedded_physics_pass",
            "visual_mesh_audit_pending": True,
        }
        for sample in samples
    ]
    fields = [
        "object",
        "pose_sha256",
        "policy_checkpoint_sha256",
        "sample_pt",
        "receipt",
        "trajectory_pt",
        "trajectory_json",
        "acceptance_profile",
        "visual_mesh_audit_pending",
    ]
    with (accepted_root / "accepted.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(sample["object_name"] == name for sample in samples) for name in object_names}
    target = int(manifest["target_per_object"])
    progress = {
        "schema": "xhand_rl_embedded_progress_v2",
        "target_per_object": target,
        "counts": counts,
        "accepted_total": len(samples),
        "acceptance_profile": "embedded_physics_pass",
        "strict_visual_mesh_accepted": False,
        "omitted_expensive_gates": list(OMITTED_EXPENSIVE_GATES),
        "complete": all(value >= target for value in counts.values()),
    }
    (accepted_root / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    if progress["complete"]:
        torch.save(
            {"schema": "xhand_rl_embedded_600_dataset_v2", "samples": samples},
            accepted_root / "all_600.pt",
        )
        shutil.copy2(accepted_root / "all_accepted.json", accepted_root / "all_600.json")
        shutil.copy2(accepted_root / "accepted.csv", accepted_root / "all_600.csv")


def materialize_success(
    *,
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    trajectory: dict[str, Any],
    episode_report: dict[str, Any],
    force_closure_report: dict[str, Any],
) -> Path | None:
    """Store one unique embedded pass, returning None for an exact duplicate."""

    validate_manifest(manifest, verify_files=True)
    validate_root(root, manifest_path, manifest)
    if sample.get("generation_backend") != EMBEDDED_BACKEND:
        raise RuntimeError("refusing non-v2 sample")
    object_name = sample["object_name"]
    if object_name not in manifest["objects"]:
        raise RuntimeError("unknown locked object")
    checkpoint = Path(sample["policy_checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != sample["policy_checkpoint_sha256"]:
        raise RuntimeError("policy checkpoint provenance changed")
    pose_hash = sample["policy_candidate_pose_sha256"]
    if len(pose_hash) != 64 or any(character not in "0123456789abcdef" for character in pose_hash):
        raise RuntimeError("invalid v2 pose hash")
    _validate_trajectory_evidence(trajectory, episode_report)
    physics_gates = episode_report.get("physics_gates", {})
    gates = {
        "asset_hashes": True,
        "rl_provenance": True,
        **physics_gates,
        "unique_pose": True,
        "trajectory_recorded": True,
    }
    report_for_validation = {
        "gates": gates,
        "omitted_expensive_gates": list(OMITTED_EXPENSIVE_GATES),
        "strict_visual_mesh_accepted": False,
    }
    validate_gate_report(report_for_validation)
    accepted_root = root / "accepted"
    object_root = accepted_root / object_name
    object_root.mkdir(parents=True, exist_ok=True)
    lock_path = accepted_root / ".materialize.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing_receipts = list(accepted_root.glob("*/sample_*/receipt.json"))
        for receipt_path in existing_receipts:
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("pose_sha256") == pose_hash:
                return None
        if accepted_count(root, object_name) >= int(manifest["target_per_object"]):
            return None
        target = object_root / f"sample_{pose_hash[:16]}"
        if target.exists():
            return None
        temporary = Path(tempfile.mkdtemp(prefix=".incoming_", dir=object_root))
        try:
            trajectory_pt = temporary / "trajectory.pt"
            trajectory_json = temporary / "trajectory.json"
            torch.save(trajectory, trajectory_pt)
            trajectory_json.write_text(json.dumps(jsonable(trajectory), indent=2) + "\n")
            force_closure_path = temporary / "force_closure.json"
            force_closure_path.write_text(json.dumps(jsonable(force_closure_report), indent=2) + "\n")
            episode_path = temporary / "embedded_episode_report.json"
            episode_path.write_text(json.dumps(jsonable(episode_report), indent=2) + "\n")
            target_paths = {
                "sample_pt": str((target / "sample.pt").resolve()),
                "sample_json": str((target / "sample.json").resolve()),
                "receipt": str((target / "receipt.json").resolve()),
                "trajectory_pt": str((target / "trajectory.pt").resolve()),
                "trajectory_json": str((target / "trajectory.json").resolve()),
                "force_closure": str((target / "force_closure.json").resolve()),
                "episode_report": str((target / "embedded_episode_report.json").resolve()),
            }
            stored_sample = copy.deepcopy(sample)
            stored_sample["embedded_physics_pass"] = True
            stored_sample["strict_visual_mesh_accepted"] = False
            stored_sample["visual_mesh_audit_pending"] = True
            stored_sample["embedded_rl_validation"] = {
                "schema": "xhand_rl_embedded_validation_v2",
                "pose_sha256": pose_hash,
                "gates": gates,
                **target_paths,
            }
            torch.save(
                {"schema": "xhand_rl_embedded_sample_v2", "samples": [stored_sample]},
                temporary / "sample.pt",
            )
            (temporary / "sample.json").write_text(json.dumps(jsonable(stored_sample), indent=2) + "\n")
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "backend": EMBEDDED_BACKEND,
                "acceptance_profile": "embedded_physics_pass",
                "accepted": True,
                "object": object_name,
                "pose_sha256": pose_hash,
                "policy_checkpoint": sample["policy_checkpoint"],
                "policy_checkpoint_sha256": sample["policy_checkpoint_sha256"],
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "reward_revision": manifest["algorithm"]["reward_revision"],
                "ppo_revision": manifest["algorithm"]["ppo_revision"],
                "training_reward_scale": manifest["algorithm"]["training_reward_scale"],
                "trajectory_sha256": sha256_file(trajectory_pt),
                "trajectory_json_sha256": sha256_file(trajectory_json),
                "force_closure_sha256": sha256_file(force_closure_path),
                "episode_report_sha256": sha256_file(episode_path),
                "gates": gates,
                "omitted_expensive_gates": list(OMITTED_EXPENSIVE_GATES),
                "strict_visual_mesh_accepted": False,
                "visual_mesh_audit_pending": True,
                "artifacts": target_paths,
            }
            validate_receipt(receipt, manifest)
            (temporary / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            os.rename(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        _write_indexes(accepted_root, manifest)
    return target
