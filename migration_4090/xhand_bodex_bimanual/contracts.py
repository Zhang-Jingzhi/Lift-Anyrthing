"""Provenance contract for genuine jointly optimized bimanual BODex banks."""

from __future__ import annotations

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any

import torch


BODEX_REPOSITORY = "https://github.com/JYChen18/BODex"
BODEX_COMMIT = "06b9a3c90870d33bde9d6c665d4ed2819471407e"
BODEX_BACKEND = "bodex_bimanual_joint_grasp_optimization_v1"
BODEX_BANK_SCHEMA = "xhand_bodex_bimanual_bank_v1"
BODEX_CURRICULUM_BANK_SCHEMA = "xhand_bodex_bimanual_curriculum_bank_v1"
SMOKE_BANK_SCHEMA = "xhand_bodex_bimanual_smoke_fixture_v1"
SMOKE_BACKEND = "synthetic_smoke_fixture_not_bodex"

REQUIRED_BODEX_SOURCE_FILES = (
    "src/curobo/util/sample_grasp.py",
    "src/curobo/wrap/reacher/grasp_solver.py",
    "src/curobo/rollout/cost/grasp_cost.py",
    "src/curobo/rollout/cost/grasp_energy/base.py",
    "src/curobo/rollout/cost/grasp_energy/qp.py",
)
REQUIRED_TRAJECTORY_FIELDS = (
    "pregrasp_full_body_q",
    "full_body_q",
    "lift_full_body_q",
)


class BODexContractError(RuntimeError):
    """Raised when a candidate bank is not provably joint bimanual BODex."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BODexContractError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def bodex_source_hashes(source_root: str | Path) -> dict[str, str]:
    root = Path(source_root).resolve()
    hashes: dict[str, str] = {}
    for relative in REQUIRED_BODEX_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise BODexContractError(f"BODex source file is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def verify_bodex_checkout(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root).resolve()
    _require(root.is_dir(), f"BODex source checkout does not exist: {root}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BODexContractError(f"cannot read BODex git commit: {root}") from exc
    _require(commit == BODEX_COMMIT, f"BODex commit mismatch: {commit}")
    return {
        "repository": BODEX_REPOSITORY,
        "commit": commit,
        "source_root": str(root),
        "source_files_sha256": bodex_source_hashes(root),
    }


def _supported_curriculum_stages(payload: dict[str, Any], *, smoke: bool) -> tuple[int, ...]:
    if smoke:
        return tuple(range(1, 7))
    stages = payload.get("supported_curriculum_stages")
    _require(isinstance(stages, list) and stages, "bank lacks supported curriculum stages")
    normalized = tuple(int(stage) for stage in stages)
    _require(
        normalized == tuple(sorted(set(normalized)))
        and all(1 <= stage <= 6 for stage in normalized),
        "bank supported curriculum stages are invalid",
    )
    return normalized


def _validate_sample(
    sample: dict[str, Any],
    object_name: str,
    *,
    smoke: bool,
    stage_seed_profile: int | None,
    requires_lift_ik: bool,
) -> None:
    _require(sample.get("object_name") == object_name, "sample object does not match bank")
    names = sample.get("joint_names")
    _require(isinstance(names, list) and len(names) >= 38, "sample joint_names are incomplete")
    _require(len(set(names)) == len(names), "sample joint_names contain duplicates")
    for field in REQUIRED_TRAJECTORY_FIELDS:
        values = sample.get(field)
        _require(isinstance(values, (list, tuple)), f"sample lacks {field}")
        _require(len(values) == len(names), f"{field} length does not match joint_names")
        _require(
            all(math.isfinite(float(value)) for value in values),
            f"{field} contains a non-finite value",
        )
    if smoke:
        _require(sample.get("test_fixture") is True, "smoke sample is not marked as a fixture")
        return
    _require(
        int(sample.get("stage_seed_profile", -1)) == stage_seed_profile,
        "sample stage seed profile does not match bank",
    )
    result = sample.get("bodex_result")
    _require(isinstance(result, dict), "sample lacks BODex optimization result")
    _require(result.get("optimizer") == "BODex.GraspSolver", "wrong BODex optimizer")
    _require(result.get("joint_bimanual_optimization") is True, "hands were not optimized jointly")
    _require(result.get("combined_grasp_matrix") is True, "combined left/right grasp matrix is absent")
    _require(
        result.get("independent_single_hand_pairing") is False,
        "independent single-hand pairing is forbidden",
    )
    side_counts = result.get("contact_counts_by_side")
    _require(isinstance(side_counts, dict), "BODex result lacks side-aware contacts")
    _require(
        int(side_counts.get("left", 0)) > 0 and int(side_counts.get("right", 0)) > 0,
        "BODex result is not bilateral",
    )
    matrix_shape = result.get("grasp_matrix_shape")
    _require(
        isinstance(matrix_shape, (list, tuple))
        and len(matrix_shape) >= 2
        and int(matrix_shape[-2]) == 6,
        "BODex grasp matrix shape is invalid",
    )
    _require(
        result.get("curriculum_seed_accepted") is True,
        "BODex result was not accepted for its curriculum seed profile",
    )
    if requires_lift_ik:
        _require(
            result.get("strict_bodex_success") is True,
            "lift-ready bank sample lacks strict BODex success",
        )
        _require(
            sample.get("lift_mode") == "curobo_collision_aware_two_palm_ik_from_bodex_grasp",
            "lift-ready sample lacks collision-aware two-palm IK",
        )
        lift_result = sample.get("lift_result")
        _require(
            isinstance(lift_result, dict) and lift_result.get("success") is True,
            "lift-ready sample has no successful lift IK result",
        )
    else:
        _require(stage_seed_profile is not None and stage_seed_profile <= 3, "invalid pre-lift profile")
        _require(
            sample.get("lift_mode") == "not_required_for_curriculum_stages_1_to_3",
            "pre-lift sample has an ambiguous lift mode",
        )
        _require(
            list(sample["lift_full_body_q"]) == list(sample["full_body_q"]),
            "pre-lift sample must keep lift_full_body_q equal to full_body_q",
        )
        acceptance = result.get("stage_seed_acceptance")
        _require(isinstance(acceptance, dict), "pre-lift sample lacks acceptance diagnostics")
        _require(acceptance.get("accepted") is True, "pre-lift sample failed its acceptance gate")
        _require(
            acceptance.get("positive_world_penetration") is False,
            "pre-lift sample has positive world penetration",
        )
        _require(
            acceptance.get("joint_bounds_exact_zero") is True
            and acceptance.get("self_collision_exact_zero") is True,
            "pre-lift sample violates exact joint/self-collision constraints",
        )
        maximum_gap = float(acceptance.get("maximum_gap_m", -1.0))
        gap = float(acceptance.get("world_gap_m", math.inf))
        _require(
            math.isfinite(maximum_gap)
            and maximum_gap > 0.0
            and math.isfinite(gap)
            and 0.0 <= gap <= maximum_gap,
            "pre-lift sample exceeds its declared world-gap gate",
        )
        grasp_threshold = float(acceptance.get("grasp_threshold", -1.0))
        distance_threshold = float(acceptance.get("distance_threshold", -1.0))
        grasp_error = float(acceptance.get("grasp_error_max", math.inf))
        distance_error = float(acceptance.get("distance_error_max", math.inf))
        _require(
            math.isfinite(grasp_error)
            and math.isfinite(distance_error)
            and grasp_error <= grasp_threshold
            and distance_error <= distance_threshold,
            "pre-lift sample exceeds BODex grasp/distance thresholds",
        )


def validate_bodex_bank(
    payload: dict[str, Any],
    *,
    expected_object: str | None = None,
    verify_source: bool = False,
    allow_test_fixture: bool = False,
    intended_stage: int | None = None,
) -> None:
    """Validate a bank and reject legacy/manual poses by default."""

    _require(isinstance(payload, dict), "BODex bank must be a mapping")
    smoke = payload.get("schema") == SMOKE_BANK_SCHEMA
    if smoke:
        _require(allow_test_fixture, "synthetic smoke bank is forbidden for training")
        _require(payload.get("generation_backend") == SMOKE_BACKEND, "wrong smoke backend")
        _require(payload.get("test_fixture") is True, "smoke bank marker is absent")
    else:
        schema = payload.get("schema")
        _require(
            schema in (BODEX_BANK_SCHEMA, BODEX_CURRICULUM_BANK_SCHEMA),
            "wrong BODex bank schema",
        )
        _require(payload.get("generation_backend") == BODEX_BACKEND, "wrong BODex backend")
        provenance = payload.get("bodex_provenance")
        _require(isinstance(provenance, dict), "bank lacks BODex provenance")
        _require(provenance.get("repository") == BODEX_REPOSITORY, "wrong BODex repository")
        _require(provenance.get("commit") == BODEX_COMMIT, "wrong BODex commit")
        _require(provenance.get("joint_bimanual_optimization") is True, "bank is not joint bimanual")
        _require(provenance.get("combined_grasp_matrix") is True, "bank lacks combined grasp matrix")
        _require(
            provenance.get("independent_single_hand_pairing") is False,
            "bank pairs independent single-hand grasps",
        )
        source_hashes = provenance.get("source_files_sha256")
        _require(isinstance(source_hashes, dict), "bank lacks BODex source hashes")
        for relative in REQUIRED_BODEX_SOURCE_FILES:
            _require(_is_sha256(source_hashes.get(relative)), f"missing source hash: {relative}")
        if verify_source:
            actual = verify_bodex_checkout(provenance.get("source_root", ""))
            _require(
                actual["source_files_sha256"] == source_hashes,
                "BODex source changed after bank generation",
            )
    stages = _supported_curriculum_stages(payload, smoke=smoke)
    if intended_stage is not None:
        _require(1 <= int(intended_stage) <= 6, "intended curriculum stage is invalid")
        _require(
            int(intended_stage) in stages,
            f"bank does not support curriculum stage {intended_stage}; supports {list(stages)}",
        )
    stage_seed_profile: int | None = None
    requires_lift_ik = False
    if not smoke:
        stage_seed_profile = int(payload.get("stage_seed_profile", -1))
        _require(1 <= stage_seed_profile <= 6, "bank stage seed profile is invalid")
        requires_lift_ik = bool(payload.get("requires_lift_ik"))
        if payload.get("schema") == BODEX_CURRICULUM_BANK_SCHEMA:
            _require(stage_seed_profile <= 3, "pre-lift curriculum bank profile must be <= 3")
            _require(not requires_lift_ik, "pre-lift curriculum bank cannot require lift IK")
            _require(
                stages == tuple(range(stage_seed_profile, 4)),
                "pre-lift curriculum bank has inconsistent supported stages",
            )
        else:
            _require(stage_seed_profile >= 4, "lift-ready bank profile must be >= 4")
            _require(requires_lift_ik, "lift-ready bank must require lift IK")
    object_name = payload.get("object")
    _require(isinstance(object_name, str) and object_name, "bank object is missing")
    if expected_object is not None:
        _require(object_name == expected_object, "bank object does not match requested object")
    samples = payload.get("samples")
    _require(isinstance(samples, list) and samples, "BODex bank has no samples")
    candidate_ids = [sample.get("candidate_id") for sample in samples if isinstance(sample, dict)]
    _require(
        all(isinstance(candidate_id, str) and candidate_id for candidate_id in candidate_ids)
        and len(set(candidate_ids)) == len(samples),
        "BODex bank candidate identifiers are missing or duplicated",
    )
    for sample in samples:
        _require(isinstance(sample, dict), "BODex sample must be a mapping")
        _validate_sample(
            sample,
            object_name,
            smoke=smoke,
            stage_seed_profile=stage_seed_profile,
            requires_lift_ik=requires_lift_ik,
        )


def load_bodex_bank(
    path: str | Path,
    *,
    expected_object: str | None = None,
    verify_source: bool = False,
    allow_test_fixture: bool = False,
    intended_stage: int | None = None,
) -> dict[str, Any]:
    bank_path = Path(path)
    if not bank_path.is_file():
        raise FileNotFoundError(bank_path)
    payload = torch.load(bank_path, map_location="cpu", weights_only=False)
    validate_bodex_bank(
        payload,
        expected_object=expected_object,
        verify_source=verify_source,
        allow_test_fixture=allow_test_fixture,
        intended_stage=intended_stage,
    )
    return payload
