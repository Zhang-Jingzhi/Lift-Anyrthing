"""Headless GPU-PhysX smoke test based on Isaac Gym's joint_monkey example."""

from pathlib import Path

import numpy as np
from isaacgym import gymapi
import isaacgym


gym = gymapi.acquire_gym()
params = gymapi.SimParams()
params.dt = 1.0 / 60.0
params.substeps = 2
params.physx.solver_type = 1
params.physx.num_position_iterations = 6
params.physx.num_velocity_iterations = 0
params.physx.use_gpu = True
params.use_gpu_pipeline = False

sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
if sim is None:
    raise RuntimeError("Failed to create headless GPU-PhysX simulation")

plane = gymapi.PlaneParams()
gym.add_ground(sim, plane)

asset_root = Path(isaacgym.__file__).resolve().parents[2] / "assets"
asset_file = "urdf/cartpole.urdf"
options = gymapi.AssetOptions()
options.fix_base_link = True
asset = gym.load_asset(sim, str(asset_root), asset_file, options)
if asset is None:
    raise RuntimeError(f"Failed to load {asset_root / asset_file}")

dof_count = gym.get_asset_dof_count(asset)
body_count = gym.get_asset_rigid_body_count(asset)
if dof_count <= 0 or body_count <= 0:
    raise RuntimeError("Loaded asset has no DoFs or rigid bodies")

env = gym.create_env(
    sim,
    gymapi.Vec3(-2.0, -2.0, -2.0),
    gymapi.Vec3(2.0, 2.0, 2.0),
    1,
)
actor = gym.create_actor(
    env,
    asset,
    gymapi.Transform(),
    "joint_monkey_cartpole",
    0,
    1,
)

states = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
states["pos"] = 0.0
gym.set_actor_dof_states(env, actor, states, gymapi.STATE_ALL)

for _ in range(240):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

final_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
if not np.isfinite(final_states["pos"]).all():
    raise RuntimeError("Non-finite DoF state after simulation")

print("HEADLESS_GPU_PHYSX_OK")
print("COMPUTE_DEVICE 0")
print("GRAPHICS_DEVICE -1")
print(f"ASSET {asset_root / asset_file}")
print(f"RIGID_BODY_COUNT {body_count}")
print(f"DOF_COUNT {dof_count}")
print(f"STEPS 240")
print("FINAL_DOF_POS " + " ".join(f"{value:.9f}" for value in final_states["pos"]))

gym.destroy_env(env)
gym.destroy_sim(sim)
