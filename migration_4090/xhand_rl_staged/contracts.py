"""Schemas and promotion rules for staged PPO artifacts."""

from __future__ import annotations


STAGED_BACKEND = "isaaclab_rsl_rl_ppo_bodex_bimanual_staged_v1"
STAGED_TRAINING_SCHEMA = "xhand_rl_staged_training_attempt_v1"
STAGED_EVALUATION_SCHEMA = "xhand_rl_staged_fixed_evaluation_v1"
PROMOTION_SUCCESS_RATE = 0.80
PROMOTION_CONSECUTIVE_WINDOWS = 3

