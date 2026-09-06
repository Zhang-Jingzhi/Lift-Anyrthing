"""Embedded-gate Isaac Lab PPO generation for the locked XHand objects."""

from .contracts import (
    EMBEDDED_BACKEND,
    EMBEDDED_GATE_NAMES,
    OMITTED_EXPENSIVE_GATES,
    validate_manifest,
)

__all__ = [
    "EMBEDDED_BACKEND",
    "EMBEDDED_GATE_NAMES",
    "OMITTED_EXPENSIVE_GATES",
    "validate_manifest",
]
