"""Load the mirrored Allegro pair in Isaac Gym and compare asset metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacgym import gymapi

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    repo = args.repo.resolve()

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 0.01
    params.substeps = 2
    params.physx.use_gpu = True
    sim = gym.create_sim(args.gpu, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    options = gymapi.AssetOptions()
    options.disable_gravity = True
    options.fix_base_link = True
    options.collapse_fixed_joints = True
    asset_root = str(repo / "data/data_urdf/robot")
    left = gym.load_asset(
        sim,
        asset_root,
        "allegro/allegro_hand_left_extended.urdf",
        options,
    )
    right = gym.load_asset(
        sim,
        asset_root,
        "allegro_right/allegro_hand_right_extended.urdf",
        options,
    )
    if left is None or right is None:
        gym.destroy_sim(sim)
        raise RuntimeError("Failed to load one or both Allegro assets")

    left_properties = gym.get_asset_dof_properties(left)
    right_properties = gym.get_asset_dof_properties(right)
    checks = {
        "dof_count_equal": gym.get_asset_dof_count(left) == gym.get_asset_dof_count(right),
        "rigid_body_count_equal": gym.get_asset_rigid_body_count(left)
        == gym.get_asset_rigid_body_count(right),
        "rigid_shape_count_equal": gym.get_asset_rigid_shape_count(left)
        == gym.get_asset_rigid_shape_count(right),
        "dof_names_equal": gym.get_asset_dof_names(left) == gym.get_asset_dof_names(right),
        "rigid_body_names_equal": gym.get_asset_rigid_body_names(left)
        == gym.get_asset_rigid_body_names(right),
        "lower_limits_equal": bool(
            np.allclose(left_properties["lower"], right_properties["lower"])
        ),
        "upper_limits_equal": bool(
            np.allclose(left_properties["upper"], right_properties["upper"])
        ),
    }
    environment = gym.create_env(
        sim,
        gymapi.Vec3(-1, -1, -1),
        gymapi.Vec3(1, 1, 1),
        1,
    )
    gym.create_actor(environment, left, gymapi.Transform(), "left", 0)
    gym.create_actor(environment, right, gymapi.Transform(), "right", 0)
    gym.prepare_sim(sim)
    for _ in range(5):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "left": {
            "dofs": gym.get_asset_dof_count(left),
            "rigid_bodies": gym.get_asset_rigid_body_count(left),
            "rigid_shapes": gym.get_asset_rigid_shape_count(left),
        },
        "right": {
            "dofs": gym.get_asset_dof_count(right),
            "rigid_bodies": gym.get_asset_rigid_body_count(right),
            "rigid_shapes": gym.get_asset_rigid_shape_count(right),
        },
        "simulation_steps": 5,
    }
    output = repo / "data/data_urdf/robot/allegro_right/ISAAC_VALIDATION_REPORT.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    gym.destroy_env(environment)
    gym.destroy_sim(sim)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
