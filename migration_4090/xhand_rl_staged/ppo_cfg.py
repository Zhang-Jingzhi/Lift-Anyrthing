"""Checkpoint-compatible PPO settings for all six stages."""

from isaaclab.utils import configclass

from migration_4090.xhand_rl_embedded.ppo_cfg import XHandEmbeddedPPORunnerCfg


@configclass
class XHandStagedPPORunnerCfg(XHandEmbeddedPPORunnerCfg):
    num_steps_per_env = 64
    max_iterations = 1000
    save_interval = 25
    experiment_name = "xhand_bodex_bimanual_staged_rl_v1"
