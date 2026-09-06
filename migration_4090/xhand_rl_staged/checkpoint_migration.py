"""Checkpoint-compatible observation expansion for candidate-conditioned PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from migration_4090.xhand_bodex_bimanual.contracts import sha256_file


OLD_OBSERVATION_DIMENSION = 269
CANDIDATE_CONTEXT_DIMENSION = 4
NEW_OBSERVATION_DIMENSION = (
    OLD_OBSERVATION_DIMENSION + CANDIDATE_CONTEXT_DIMENSION
)
GATE_MEMORY_DIMENSION = 4
GATE_MEMORY_OBSERVATION_DIMENSION = (
    NEW_OBSERVATION_DIMENSION + GATE_MEMORY_DIMENSION
)
CANDIDATE_GATED_MEMORY_DIMENSION = (
    CANDIDATE_CONTEXT_DIMENSION * GATE_MEMORY_DIMENSION
)
CANDIDATE_GATED_MEMORY_OBSERVATION_DIMENSION = (
    GATE_MEMORY_OBSERVATION_DIMENSION + CANDIDATE_GATED_MEMORY_DIMENSION
)


def append_candidate_context_to_checkpoint_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Append zero-effect candidate inputs while preserving policy outputs.

    The four new actor/critic input columns are exactly zero, so evaluating a
    migrated checkpoint produces the same network outputs as its 269D source
    for every original observation.  Normalizer statistics use mean 0/std 1
    for the appended raw one-hot context.
    """

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model_state_dict")
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"checkpoint lacks a matrix tensor at {key}")
        if weight.shape[1] == NEW_OBSERVATION_DIMENSION:
            raise ValueError("checkpoint already contains candidate observation context")
        if weight.shape[1] != OLD_OBSERVATION_DIMENSION:
            raise ValueError(
                f"unexpected checkpoint observation dimension at {key}: "
                f"{weight.shape[1]}"
            )
        state[key] = torch.nn.functional.pad(
            weight, (0, CANDIDATE_CONTEXT_DIMENSION), value=0.0
        )
    normalizer_suffix_values = {
        "_mean": 0.0,
        "_var": 1.0,
        "_std": 1.0,
    }
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for suffix, value in normalizer_suffix_values.items():
            key = f"{prefix}.{suffix}"
            tensor = state.get(key)
            if not isinstance(tensor, torch.Tensor) or tensor.shape[-1] != OLD_OBSERVATION_DIMENSION:
                raise ValueError(f"unexpected or missing normalizer tensor at {key}")
            extension = torch.full(
                (*tensor.shape[:-1], CANDIDATE_CONTEXT_DIMENSION),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            state[key] = torch.cat((tensor, extension), dim=-1)
    payload["candidate_context_migration"] = {
        "schema": "xhand_rl_staged_candidate_context_checkpoint_migration_v1",
        "old_observation_dimension": OLD_OBSERVATION_DIMENSION,
        "new_observation_dimension": NEW_OBSERVATION_DIMENSION,
        "candidate_context": "four_way_one_hot",
        "new_actor_critic_columns_zero_initialized": True,
    }
    return payload


def insert_gate_memory_to_checkpoint_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Insert zero-effect gate-memory inputs before candidate context.

    The candidate one-hot remains the final four observation dimensions so
    candidate-column masking keeps a stable ABI.  The four new gate-memory
    columns are inserted immediately before it and initialized to zero in the
    actor and critic.  Existing 273D network outputs are therefore preserved
    exactly when the new gate-memory observation is zero.
    """

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model_state_dict")
    insertion = NEW_OBSERVATION_DIMENSION - CANDIDATE_CONTEXT_DIMENSION
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"checkpoint lacks a matrix tensor at {key}")
        if weight.shape[1] == GATE_MEMORY_OBSERVATION_DIMENSION:
            raise ValueError("checkpoint already contains gate-memory observation")
        if weight.shape[1] != NEW_OBSERVATION_DIMENSION:
            raise ValueError(
                f"unexpected checkpoint observation dimension at {key}: "
                f"{weight.shape[1]}"
            )
        extension = torch.zeros(
            (weight.shape[0], GATE_MEMORY_DIMENSION),
            dtype=weight.dtype,
            device=weight.device,
        )
        state[key] = torch.cat(
            (weight[:, :insertion], extension, weight[:, insertion:]), dim=1
        )
    normalizer_suffix_values = {
        "_mean": 0.0,
        "_var": 1.0,
        "_std": 1.0,
    }
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for suffix, value in normalizer_suffix_values.items():
            key = f"{prefix}.{suffix}"
            tensor = state.get(key)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape[-1] != NEW_OBSERVATION_DIMENSION
            ):
                raise ValueError(f"unexpected or missing normalizer tensor at {key}")
            extension = torch.full(
                (*tensor.shape[:-1], GATE_MEMORY_DIMENSION),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            state[key] = torch.cat(
                (tensor[..., :insertion], extension, tensor[..., insertion:]),
                dim=-1,
            )
    payload["gate_memory_migration"] = {
        "schema": "xhand_rl_staged_gate_memory_checkpoint_migration_v1",
        "old_observation_dimension": NEW_OBSERVATION_DIMENSION,
        "new_observation_dimension": GATE_MEMORY_OBSERVATION_DIMENSION,
        "gate_memory_dimension": GATE_MEMORY_DIMENSION,
        "candidate_context_preserved_as_final_dimensions": True,
        "new_actor_critic_columns_zero_initialized": True,
    }
    return payload


def insert_candidate_gated_memory_to_checkpoint_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Insert zero-effect candidate-specific gate-memory interactions.

    The 16 new values are four contiguous gate-memory features per BODex
    candidate.  They are inserted immediately before the final candidate
    one-hot, preserving that long-lived context ABI.  Zero actor/critic input
    columns make the migrated 293D checkpoint exactly reproduce its 277D
    source before the new adapter is trained.
    """

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model_state_dict")
    insertion = GATE_MEMORY_OBSERVATION_DIMENSION - CANDIDATE_CONTEXT_DIMENSION
    for key in ("actor.0.weight", "critic.0.weight"):
        weight = state.get(key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"checkpoint lacks a matrix tensor at {key}")
        if weight.shape[1] == CANDIDATE_GATED_MEMORY_OBSERVATION_DIMENSION:
            raise ValueError(
                "checkpoint already contains candidate-gated memory observation"
            )
        if weight.shape[1] != GATE_MEMORY_OBSERVATION_DIMENSION:
            raise ValueError(
                f"unexpected checkpoint observation dimension at {key}: "
                f"{weight.shape[1]}"
            )
        extension = torch.zeros(
            (weight.shape[0], CANDIDATE_GATED_MEMORY_DIMENSION),
            dtype=weight.dtype,
            device=weight.device,
        )
        state[key] = torch.cat(
            (weight[:, :insertion], extension, weight[:, insertion:]), dim=1
        )
    normalizer_suffix_values = {
        "_mean": 0.0,
        "_var": 1.0,
        "_std": 1.0,
    }
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        for suffix, value in normalizer_suffix_values.items():
            key = f"{prefix}.{suffix}"
            tensor = state.get(key)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape[-1] != GATE_MEMORY_OBSERVATION_DIMENSION
            ):
                raise ValueError(f"unexpected or missing normalizer tensor at {key}")
            extension = torch.full(
                (*tensor.shape[:-1], CANDIDATE_GATED_MEMORY_DIMENSION),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            state[key] = torch.cat(
                (tensor[..., :insertion], extension, tensor[..., insertion:]),
                dim=-1,
            )
    payload["candidate_gated_memory_migration"] = {
        "schema": "xhand_rl_staged_candidate_gated_memory_checkpoint_migration_v1",
        "old_observation_dimension": GATE_MEMORY_OBSERVATION_DIMENSION,
        "new_observation_dimension": CANDIDATE_GATED_MEMORY_OBSERVATION_DIMENSION,
        "candidate_count": CANDIDATE_CONTEXT_DIMENSION,
        "gate_memory_dimension": GATE_MEMORY_DIMENSION,
        "candidate_gated_memory_dimension": CANDIDATE_GATED_MEMORY_DIMENSION,
        "candidate_context_preserved_as_final_dimensions": True,
        "new_actor_critic_columns_zero_initialized": True,
    }
    return payload


def migrate_candidate_context_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    append_candidate_context_to_checkpoint_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "schema": "xhand_rl_staged_candidate_context_checkpoint_file_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "old_observation_dimension": OLD_OBSERVATION_DIMENSION,
        "new_observation_dimension": NEW_OBSERVATION_DIMENSION,
        "candidate_context": "four_way_one_hot",
        "network_output_preservation": "zero_padded_actor_critic_input_columns",
    }


def migrate_gate_memory_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    insert_gate_memory_to_checkpoint_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "schema": "xhand_rl_staged_gate_memory_checkpoint_file_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "old_observation_dimension": NEW_OBSERVATION_DIMENSION,
        "new_observation_dimension": GATE_MEMORY_OBSERVATION_DIMENSION,
        "gate_memory_dimension": GATE_MEMORY_DIMENSION,
        "candidate_context": "four_way_one_hot_final_dimensions",
        "network_output_preservation": "zero_inserted_actor_critic_input_columns",
    }


def migrate_candidate_gated_memory_checkpoint(
    source: Path, output: Path
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    insert_candidate_gated_memory_to_checkpoint_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "schema": "xhand_rl_staged_candidate_gated_memory_checkpoint_file_v1",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "old_observation_dimension": GATE_MEMORY_OBSERVATION_DIMENSION,
        "new_observation_dimension": CANDIDATE_GATED_MEMORY_OBSERVATION_DIMENSION,
        "candidate_count": CANDIDATE_CONTEXT_DIMENSION,
        "gate_memory_dimension": GATE_MEMORY_DIMENSION,
        "candidate_gated_memory_dimension": CANDIDATE_GATED_MEMORY_DIMENSION,
        "candidate_context": "four_way_one_hot_final_dimensions",
        "network_output_preservation": "zero_inserted_actor_critic_input_columns",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--migration",
        choices=("candidate_context", "gate_memory", "candidate_gated_memory"),
        default="candidate_context",
    )
    args = parser.parse_args()
    if args.migration == "candidate_context":
        result = migrate_candidate_context_checkpoint(args.source, args.output)
    elif args.migration == "gate_memory":
        result = migrate_gate_memory_checkpoint(args.source, args.output)
    else:
        result = migrate_candidate_gated_memory_checkpoint(args.source, args.output)
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
