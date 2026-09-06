"""Helpers for making resumed PPO exploration noise explicit and auditable."""

from __future__ import annotations

import math
from typing import Any

import torch


def reset_policy_noise_std(policy: Any, value: float) -> float:
    """Reset a checkpoint-loaded RSL-RL policy's learnable action std.

    RSL-RL restores the action-noise parameter as part of the model state even
    when optimizer state is intentionally not loaded.  Curriculum transitions
    therefore need an explicit post-load reset when the target exploration
    scale changes.
    """

    target = float(value)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("policy noise standard deviation must be positive and finite")
    if bool(getattr(policy, "state_dependent_std", False)):
        raise ValueError("post-load noise reset does not support state-dependent std")
    noise_type = getattr(policy, "noise_std_type", None)
    with torch.no_grad():
        if noise_type == "scalar":
            policy.std.fill_(target)
        elif noise_type == "log":
            policy.log_std.fill_(math.log(target))
        else:
            raise ValueError(f"unsupported policy noise std type: {noise_type!r}")
    return policy_noise_std(policy)


def policy_noise_std(policy: Any) -> float:
    """Read the mean learnable action std without requiring a distribution."""

    if bool(getattr(policy, "state_dependent_std", False)):
        raise ValueError("cannot summarize state-dependent policy std without observations")
    noise_type = getattr(policy, "noise_std_type", None)
    if noise_type == "scalar":
        values = policy.std
    elif noise_type == "log":
        values = torch.exp(policy.log_std)
    else:
        raise ValueError(f"unsupported policy noise std type: {noise_type!r}")
    return float(values.detach().mean().cpu().item())


def freeze_policy_noise(policy: Any) -> None:
    """Freeze the learnable scalar/log action-noise parameter in place."""

    if bool(getattr(policy, "state_dependent_std", False)):
        raise ValueError("cannot freeze state-dependent policy std with this helper")
    noise_type = getattr(policy, "noise_std_type", None)
    if noise_type == "scalar":
        parameter = policy.std
    elif noise_type == "log":
        parameter = policy.log_std
    else:
        raise ValueError(f"unsupported policy noise std type: {noise_type!r}")
    parameter.requires_grad_(False)


def restrict_actor_updates_to_candidate_context(
    policy: Any,
    context_width: int,
    trainable_context_indices: tuple[int, ...] | list[int] | None = None,
    gate_memory_width: int = 0,
    intervening_width: int = 0,
) -> dict[str, int | bool | list[int]]:
    """Train only the appended gate-memory/candidate columns of the actor.

    A migrated policy initially has zero weights for the appended BODex
    candidate one-hot.  Updating the complete actor while those columns learn
    can destroy the already validated shared contact policy.  This helper
    freezes every actor parameter, freezes updates to the actor observation
    normalizer, and masks the first layer's gradient.  By default only selected
    final candidate-context columns can change.  When ``gate_memory_width`` is
    positive, the immediately preceding gate-memory columns are also trained.
    The critic and its normalizer remain untouched and may still learn normally.

    The caller must start with a fresh optimizer state.  Adam momentum from a
    previous unrestricted update could otherwise move masked entries even
    when their current gradients are zero.
    """

    width = int(context_width)
    memory_width = int(gate_memory_width)
    skipped_width = int(intervening_width)
    if width <= 0:
        raise ValueError("candidate context width must be positive")
    if memory_width < 0:
        raise ValueError("gate memory width must be non-negative")
    if skipped_width < 0:
        raise ValueError("intervening observation width must be non-negative")
    selected = tuple(
        range(width)
        if trainable_context_indices is None
        else sorted({int(index) for index in trainable_context_indices})
    )
    if not selected:
        raise ValueError("at least one candidate context index must be trainable")
    if any(index < 0 or index >= width for index in selected):
        raise ValueError("trainable candidate context index is outside context width")
    actor = getattr(policy, "actor", None)
    if actor is None or len(actor) == 0:
        raise ValueError("policy actor is missing")
    input_layer = actor[0]
    if not isinstance(input_layer, torch.nn.Linear):
        raise ValueError("candidate-context actor input layer is not linear")
    if width + memory_width + skipped_width > input_layer.in_features:
        raise ValueError("observation adapter is wider than the actor observation")
    if not bool(getattr(policy, "actor_obs_normalization", False)):
        raise ValueError(
            "candidate-context-only actor training requires actor normalization"
        )

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    # ActorCritic.act() still applies the existing normalizer when this flag is
    # false; ActorCritic.update_normalization() merely stops mutating its
    # running statistics.  This prevents an indirect change to all shared
    # observation dimensions while the critic continues to update normally.
    policy.actor_obs_normalization = False
    input_layer.weight.requires_grad_(True)
    gradient_mask = torch.zeros_like(input_layer.weight)
    context_start = input_layer.in_features - width
    if memory_width:
        memory_end = context_start - skipped_width
        gradient_mask[:, memory_end - memory_width : memory_end] = 1.0
    gradient_mask[:, [context_start + index for index in selected]] = 1.0

    def mask_shared_observation_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return gradient * gradient_mask

    handle = input_layer.weight.register_hook(mask_shared_observation_gradient)
    handles = list(getattr(policy, "_candidate_context_gradient_hooks", ()))
    handles.append(handle)
    policy._candidate_context_gradient_hooks = handles
    return {
        "context_width": width,
        "gate_memory_width": memory_width,
        "intervening_width": skipped_width,
        "observation_dimension": int(input_layer.in_features),
        "hidden_dimension": int(input_layer.out_features),
        "trainable_context_indices": list(selected),
        "trainable_actor_elements": int(
            (memory_width + len(selected)) * input_layer.out_features
        ),
        "actor_observation_normalizer_frozen": True,
    }


def restrict_actor_updates_to_candidate_gated_memory(
    policy: Any,
    *,
    context_width: int,
    gate_memory_width: int,
    trainable_candidate_indices: tuple[int, ...] | list[int] | None = None,
) -> dict[str, int | bool | list[int]]:
    """Train only candidate-specific gate-memory interaction columns.

    The interaction block is laid out as one contiguous ``gate_memory_width``
    slice per candidate immediately before the final candidate one-hot.  This
    preserves every existing actor path, including shared gate memory and
    candidate biases, while allowing weak BODex candidates to learn distinct
    state-dependent corrections.
    """

    candidate_count = int(context_width)
    memory_width = int(gate_memory_width)
    if candidate_count <= 0 or memory_width <= 0:
        raise ValueError("candidate and gate-memory widths must be positive")
    selected = tuple(
        range(candidate_count)
        if trainable_candidate_indices is None
        else sorted({int(index) for index in trainable_candidate_indices})
    )
    if not selected:
        raise ValueError("at least one candidate-gated memory block must be trainable")
    if any(index < 0 or index >= candidate_count for index in selected):
        raise ValueError("trainable candidate index is outside context width")
    actor = getattr(policy, "actor", None)
    if actor is None or len(actor) == 0:
        raise ValueError("policy actor is missing")
    input_layer = actor[0]
    if not isinstance(input_layer, torch.nn.Linear):
        raise ValueError("candidate-gated memory actor input layer is not linear")
    interaction_width = candidate_count * memory_width
    if candidate_count + interaction_width > input_layer.in_features:
        raise ValueError("candidate-gated memory block is wider than the observation")
    if not bool(getattr(policy, "actor_obs_normalization", False)):
        raise ValueError(
            "candidate-gated-memory-only actor training requires actor normalization"
        )

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    policy.actor_obs_normalization = False
    input_layer.weight.requires_grad_(True)
    gradient_mask = torch.zeros_like(input_layer.weight)
    interaction_start = input_layer.in_features - candidate_count - interaction_width
    for index in selected:
        start = interaction_start + index * memory_width
        gradient_mask[:, start : start + memory_width] = 1.0

    def mask_noninteraction_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return gradient * gradient_mask

    handle = input_layer.weight.register_hook(mask_noninteraction_gradient)
    handles = list(getattr(policy, "_candidate_context_gradient_hooks", ()))
    handles.append(handle)
    policy._candidate_context_gradient_hooks = handles
    return {
        "context_width": candidate_count,
        "gate_memory_width": memory_width,
        "candidate_gated_memory_width": interaction_width,
        "observation_dimension": int(input_layer.in_features),
        "hidden_dimension": int(input_layer.out_features),
        "trainable_candidate_indices": list(selected),
        "trainable_actor_elements": int(
            len(selected) * memory_width * input_layer.out_features
        ),
        "actor_observation_normalizer_frozen": True,
    }


def restrict_actor_updates_to_candidate_gated_action_rows(
    policy: Any,
    *,
    context_width: int,
    gate_memory_width: int,
    action_indices: tuple[int, ...] | list[int],
    trainable_candidate_indices: tuple[int, ...] | list[int] | None = None,
) -> dict[str, int | bool | list[int]]:
    """Train candidate adapters together with selected residual output rows.

    This is the conservative bridge update used when a shared wrist output
    head improves one BODex candidate while degrading another.  Only the
    candidate-gated gate-memory columns for the selected candidates and the
    selected action rows are trainable.  The actor backbone, all other input
    columns, all other output rows, and actor observation normalization remain
    frozen.  Callers must construct a fresh optimizer after applying this
    restriction so stale Adam moments cannot move masked parameters.
    """

    candidate_count = int(context_width)
    memory_width = int(gate_memory_width)
    if candidate_count <= 0 or memory_width <= 0:
        raise ValueError("candidate and gate-memory widths must be positive")
    selected_candidates = tuple(
        range(candidate_count)
        if trainable_candidate_indices is None
        else sorted({int(index) for index in trainable_candidate_indices})
    )
    if not selected_candidates:
        raise ValueError("at least one candidate-gated memory block must be trainable")
    if any(index < 0 or index >= candidate_count for index in selected_candidates):
        raise ValueError("trainable candidate index is outside context width")
    selected_actions = tuple(sorted({int(index) for index in action_indices}))
    if not selected_actions:
        raise ValueError("at least one actor output row must be trainable")

    actor = getattr(policy, "actor", None)
    if actor is None or len(actor) == 0:
        raise ValueError("policy actor is missing")
    input_layer = actor[0]
    output_layer = actor[-1]
    if not isinstance(input_layer, torch.nn.Linear):
        raise ValueError("candidate-gated memory actor input layer is not linear")
    if not isinstance(output_layer, torch.nn.Linear):
        raise ValueError("actor output layer is not linear")
    if any(index < 0 or index >= output_layer.out_features for index in selected_actions):
        raise ValueError("trainable action row is outside the action dimension")
    interaction_width = candidate_count * memory_width
    if candidate_count + interaction_width > input_layer.in_features:
        raise ValueError("candidate-gated memory block is wider than the observation")
    if not bool(getattr(policy, "actor_obs_normalization", False)):
        raise ValueError(
            "candidate-gated action-row training requires actor normalization"
        )

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    policy.actor_obs_normalization = False
    input_layer.weight.requires_grad_(True)
    output_layer.weight.requires_grad_(True)
    output_layer.bias.requires_grad_(True)

    input_mask = torch.zeros_like(input_layer.weight)
    interaction_start = input_layer.in_features - candidate_count - interaction_width
    for index in selected_candidates:
        start = interaction_start + index * memory_width
        input_mask[:, start : start + memory_width] = 1.0
    output_weight_mask = torch.zeros_like(output_layer.weight)
    output_bias_mask = torch.zeros_like(output_layer.bias)
    output_weight_mask[list(selected_actions), :] = 1.0
    output_bias_mask[list(selected_actions)] = 1.0

    input_handle = input_layer.weight.register_hook(
        lambda gradient: gradient * input_mask
    )
    output_weight_handle = output_layer.weight.register_hook(
        lambda gradient: gradient * output_weight_mask
    )
    output_bias_handle = output_layer.bias.register_hook(
        lambda gradient: gradient * output_bias_mask
    )
    input_handles = list(getattr(policy, "_candidate_context_gradient_hooks", ()))
    input_handles.append(input_handle)
    policy._candidate_context_gradient_hooks = input_handles
    action_handles = list(getattr(policy, "_action_row_gradient_hooks", ()))
    action_handles.extend((output_weight_handle, output_bias_handle))
    policy._action_row_gradient_hooks = action_handles
    return {
        "context_width": candidate_count,
        "gate_memory_width": memory_width,
        "candidate_gated_memory_width": interaction_width,
        "observation_dimension": int(input_layer.in_features),
        "hidden_dimension": int(input_layer.out_features),
        "trainable_candidate_indices": list(selected_candidates),
        "trainable_action_indices": list(selected_actions),
        "trainable_actor_elements": int(
            len(selected_candidates) * memory_width * input_layer.out_features
            + len(selected_actions) * (output_layer.in_features + 1)
        ),
        "actor_observation_normalizer_frozen": True,
        "actor_backbone_frozen": True,
    }


def restrict_actor_updates_to_action_rows(
    policy: Any,
    action_indices: tuple[int, ...] | list[int],
) -> dict[str, int | bool | list[int]]:
    """Freeze the shared actor backbone and train selected output rows only.

    Lift-bridge training starts from a validated contact policy.  Updating the
    complete MLP can rewrite shared features and create cross-candidate
    regression even when only hand actions are physically active.  Keeping
    the feature extractor fixed while adapting the active residual rows lets
    the policy use all existing state features without moving inactive arm
    rows or the contact backbone.
    """

    actor = getattr(policy, "actor", None)
    if actor is None or len(actor) == 0:
        raise ValueError("policy actor is missing")
    output_layer = actor[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise ValueError("actor output layer is not linear")
    selected = tuple(sorted({int(index) for index in action_indices}))
    if not selected:
        raise ValueError("at least one actor output row must be trainable")
    if any(index < 0 or index >= output_layer.out_features for index in selected):
        raise ValueError("trainable action row is outside the action dimension")
    if not bool(getattr(policy, "actor_obs_normalization", False)):
        raise ValueError("output-head-only actor training requires actor normalization")

    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    policy.actor_obs_normalization = False
    output_layer.weight.requires_grad_(True)
    output_layer.bias.requires_grad_(True)
    weight_mask = torch.zeros_like(output_layer.weight)
    bias_mask = torch.zeros_like(output_layer.bias)
    weight_mask[list(selected), :] = 1.0
    bias_mask[list(selected)] = 1.0

    weight_handle = output_layer.weight.register_hook(
        lambda gradient: gradient * weight_mask
    )
    bias_handle = output_layer.bias.register_hook(lambda gradient: gradient * bias_mask)
    handles = list(getattr(policy, "_action_row_gradient_hooks", ()))
    handles.extend((weight_handle, bias_handle))
    policy._action_row_gradient_hooks = handles
    return {
        "action_dimension": int(output_layer.out_features),
        "hidden_dimension": int(output_layer.in_features),
        "trainable_action_indices": list(selected),
        "trainable_actor_elements": int(
            len(selected) * (output_layer.in_features + 1)
        ),
        "actor_observation_normalizer_frozen": True,
        "actor_backbone_frozen": True,
    }


def zero_initialize_residual_actor(policy: Any) -> None:
    """Make a fresh residual policy exactly reproduce the BODex trajectory."""

    if bool(getattr(policy, "state_dependent_std", False)):
        raise ValueError("zero residual initialization requires state-independent std")
    output_layer = policy.actor[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise ValueError("residual actor output layer is not linear")
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.zero_()


def zero_initialize_residual_actor_rows(
    policy: Any, action_indices: list[int] | tuple[int, ...]
) -> tuple[int, ...]:
    """Zero newly unmasked residual outputs when expanding a stage action set.

    Masked actions still participate in PPO's sampled log probability, so an
    old checkpoint may contain arbitrary actor rows for joints that previously
    had no physical effect.  Those rows must be reset before the joints are
    exposed to PhysX; otherwise checkpoint migration changes the nominal BODex
    trajectory before the new actions receive a single useful gradient.
    """

    if bool(getattr(policy, "state_dependent_std", False)):
        raise ValueError("row initialization requires state-independent std")
    output_layer = policy.actor[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise ValueError("residual actor output layer is not linear")
    rows = tuple(sorted({int(index) for index in action_indices}))
    if any(index < 0 or index >= output_layer.out_features for index in rows):
        raise ValueError("residual actor row index is outside the action dimension")
    if not rows:
        return rows
    row_tensor = torch.tensor(rows, dtype=torch.long, device=output_layer.weight.device)
    with torch.no_grad():
        output_layer.weight.index_fill_(0, row_tensor, 0.0)
        output_layer.bias.index_fill_(0, row_tensor, 0.0)
    return rows
