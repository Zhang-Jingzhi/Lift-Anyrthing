"""Map the immutable v2 manifest into the Isaac Lab environment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .env import XHandEmbeddedEnvCfg


def configure_env(
    cfg: XHandEmbeddedEnvCfg,
    *,
    manifest: dict[str, Any],
    object_name: str,
    nominal_dataset: Path,
    nominal_sample_index: int,
    num_envs: int,
    device: str,
    seed: int,
    record_trajectory: bool,
) -> XHandEmbeddedEnvCfg:
    row = manifest["objects"][object_name]
    physics = manifest["physics"]
    thresholds = manifest["embedded_thresholds"]
    schedule = manifest["episode_schedule"]
    algorithm = manifest["algorithm"]
    fractions = schedule["fractions"]
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.sim.dt = float(physics["dt_s"])
    cfg.sim.gravity = (0.0, 0.0, -float(physics["gravity_m_s2"]))
    cfg.sim.physx.min_position_iteration_count = int(physics["position_iterations"])
    cfg.sim.physx.max_position_iteration_count = int(physics["position_iterations"])
    cfg.sim.physx.min_velocity_iteration_count = int(physics["velocity_iterations"])
    cfg.sim.physx.max_velocity_iteration_count = int(physics["velocity_iterations"])
    cfg.sim.physx.enable_external_forces_every_iteration = True
    cfg.seed = seed
    cfg.robot_usd = manifest["robot_usd"]["path"]
    cfg.object_usd = row["usd"]["path"]
    cfg.object_key = object_name
    cfg.object_extents_m = tuple(float(value) for value in row["extents_m"])
    cfg.object_mass_kg = float(physics["object_mass_kg"])
    cfg.contact_friction = float(physics["friction"])
    cfg.contact_offset_m = float(physics["contact_offset_m"])
    cfg.rest_offset_m = float(physics["rest_offset_m"])
    cfg.table_top_z_m = float(manifest["table_top_z_m"])
    cfg.nominal_dataset = str(nominal_dataset.resolve())
    cfg.nominal_sample_index = nominal_sample_index
    cfg.episode_length_s = float(schedule["episode_length_s"])
    cfg.approach_fraction = float(fractions["approach"])
    cfg.close_fraction = float(fractions["close"])
    cfg.lift_fraction = float(fractions["lift"])
    cfg.hold_fraction = float(fractions["hold"])
    cfg.disturbance_fraction = float(fractions["disturbance"])
    cfg.ablation_fraction = float(fractions["ablation"])
    cfg.lift_min_m = float(thresholds["lift_min_m"])
    cfg.lift_reward_target_m = float(thresholds["lift_reward_target_m"])
    cfg.gravity_drift_max_m = float(thresholds["gravity_drift_max_m"])
    cfg.translation_disturbance_max_m = float(thresholds["translation_disturbance_max_m"])
    cfg.rotation_disturbance_max_rad = float(thresholds["rotation_disturbance_max_rad"])
    cfg.contact_presence_fraction_min = float(thresholds["contact_presence_fraction_min"])
    cfg.inactive_hand_contact_fraction_max = float(thresholds["inactive_hand_contact_fraction_max"])
    cfg.physx_penetration_max_m = float(thresholds["physx_penetration_max_m"])
    cfg.force_closure_residual_max = float(thresholds["force_closure_residual_max"])
    cfg.force_closure_epsilon_min = float(thresholds["force_closure_epsilon_min"])
    cfg.disturbance_force_n = float(thresholds["disturbance_force_n"])
    cfg.disturbance_torque_nm = float(thresholds["disturbance_torque_nm"])
    cfg.training_reward_scale = float(algorithm["training_reward_scale"])
    cfg.record_trajectory = record_trajectory
    return cfg
