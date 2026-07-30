"""Isaac Gym validator for two identical dexterous hands and one object."""

from isaacgym import gymapi
from isaacgym import gymtorch

import math
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.controller import controller


class BimanualIsaacValidator:
    def __init__(
        self,
        robot_name,
        joint_orders,
        batch_size,
        gpu=0,
        use_gui=False,
        robot_friction=3.0,
        object_friction=3.0,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01,
    ):
        self.gym = gymapi.acquire_gym()
        self.robot_name = robot_name
        self.joint_orders = joint_orders
        self.batch_size = batch_size
        self.gpu = gpu
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        self.envs = []
        self.object_handles = []
        self.left_handles = []
        self.right_handles = []
        self.robot_asset = None
        self.object_asset = None
        self.rigid_body_num = None
        self.object_force = None
        self.urdf2isaac_order = None
        self.isaac2urdf_order = None

        params = gymapi.SimParams()
        params.dt = 1 / steps_per_sec
        params.substeps = 2
        params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        params.physx.use_gpu = True
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 8
        params.physx.num_velocity_iterations = 0
        params.physx.contact_offset = 0.01
        params.physx.rest_offset = 0.0
        self.sim = self.gym.create_sim(
            self.gpu,
            self.gpu,
            gymapi.SIM_PHYSX,
            params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        self.viewer = None
        self.has_viewer = use_gui
        if use_gui:
            camera = gymapi.CameraProperties()
            camera.width = 1600
            camera.height = 900
            camera.use_collision_geometry = True
            self.viewer = self.gym.create_viewer(self.sim, camera)
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(0.6, 0.6, 0.5),
                gymapi.Vec3(0.0, 0.0, 0.0),
            )

        self.robot_options = gymapi.AssetOptions()
        self.robot_options.disable_gravity = True
        self.robot_options.fix_base_link = True
        self.robot_options.collapse_fixed_joints = True

        self.object_options = gymapi.AssetOptions()
        self.object_options.override_com = True
        self.object_options.override_inertia = True
        self.object_options.density = 500

    def set_asset(self, robot_path, robot_file, object_path, object_file):
        self.robot_asset = self.gym.load_asset(
            self.sim,
            robot_path,
            robot_file,
            self.robot_options,
        )
        self.object_asset = self.gym.load_asset(
            self.sim,
            object_path,
            object_file,
            self.object_options,
        )
        self.rigid_body_num = (
            self.gym.get_asset_rigid_body_count(self.object_asset)
            + 2 * self.gym.get_asset_rigid_body_count(self.robot_asset)
        )

    def _configure_robot(self, env, handle):
        properties = self.gym.get_actor_dof_properties(env, handle)
        properties["driveMode"].fill(gymapi.DOF_MODE_POS)
        properties["stiffness"].fill(1000)
        properties["damping"].fill(200)
        self.gym.set_actor_dof_properties(env, handle, properties)

        shapes = self.gym.get_actor_rigid_shape_properties(env, handle)
        for shape in shapes:
            shape.friction = self.robot_friction
        self.gym.set_actor_rigid_shape_properties(env, handle, shapes)

    def create_envs(self):
        per_row = max(1, int(math.sqrt(self.batch_size)))
        for env_idx in range(self.batch_size):
            env = self.gym.create_env(
                self.sim,
                gymapi.Vec3(-1, -1, -1),
                gymapi.Vec3(1, 1, 1),
                per_row,
            )
            self.envs.append(env)

            object_handle = self.gym.create_actor(
                env,
                self.object_asset,
                gymapi.Transform(),
                f"object_{env_idx}",
                env_idx,
            )
            self.object_handles.append(object_handle)
            object_shapes = self.gym.get_actor_rigid_shape_properties(
                env,
                object_handle,
            )
            for shape in object_shapes:
                shape.friction = self.object_friction
            self.gym.set_actor_rigid_shape_properties(
                env,
                object_handle,
                object_shapes,
            )

            left_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                gymapi.Transform(),
                f"left_robot_{env_idx}",
                env_idx,
            )
            right_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                gymapi.Transform(),
                f"right_robot_{env_idx}",
                env_idx,
            )
            self.left_handles.append(left_handle)
            self.right_handles.append(right_handle)
            self._configure_robot(env, left_handle)
            self._configure_robot(env, right_handle)

        properties = self.gym.get_actor_rigid_body_properties(
            self.envs[0],
            self.object_handles[0],
        )
        object_mass = sum(property.mass for property in properties)
        self.object_force = 0.5 * object_mass

        self.urdf2isaac_order = np.zeros(
            len(self.joint_orders),
            dtype=np.int32,
        )
        self.isaac2urdf_order = np.zeros(
            len(self.joint_orders),
            dtype=np.int32,
        )
        for urdf_idx, joint_name in enumerate(self.joint_orders):
            isaac_idx = self.gym.find_actor_dof_index(
                self.envs[0],
                self.left_handles[0],
                joint_name,
                gymapi.DOMAIN_ACTOR,
            )
            self.urdf2isaac_order[isaac_idx] = urdf_idx
            self.isaac2urdf_order[urdf_idx] = isaac_idx

    def _set_hand_state(self, env, handle, initial_q, target_q):
        states = self.gym.get_actor_dof_states(
            env,
            handle,
            gymapi.STATE_ALL,
        ).copy()
        states["pos"] = initial_q[self.urdf2isaac_order]
        self.gym.set_actor_dof_states(
            env,
            handle,
            states,
            gymapi.STATE_ALL,
        )
        targets = target_q[self.urdf2isaac_order]
        self.gym.set_actor_dof_position_targets(env, handle, targets)

    def set_actor_pose_dof(self, left_q, right_q):
        self.gym.prepare_sim(self.sim)
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_tensor)
        root_state[:] = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        )
        self.gym.set_actor_root_state_tensor(self.sim, root_tensor)

        left_outer, left_inner = controller(self.robot_name, left_q)
        right_outer, right_inner = controller(self.robot_name, right_q)
        for index, env in enumerate(self.envs):
            self._set_hand_state(
                env,
                self.left_handles[index],
                left_outer[index],
                left_inner[index],
            )
            self._set_hand_state(
                env,
                self.right_handles[index],
                right_outer[index],
                right_inner[index],
            )

    def _draw(self):
        if not self.has_viewer:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            return
        start = time.time()
        while time.time() - start < self.debug_interval:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(
                self.viewer,
                self.sim,
                render_collision=True,
            )

    def _settled_hands_in_object_frame(self, rigid_state):
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        dof_state = gymtorch.wrap_tensor(dof_tensor).reshape(
            self.batch_size,
            2 * len(self.joint_orders),
            2,
        )[:, :, 0]
        dof_count = len(self.joint_orders)
        left_q_world = dof_state[
            :,
            :dof_count,
        ][:, self.isaac2urdf_order].clone().cpu()
        right_q_world = dof_state[
            :,
            dof_count:,
        ][:, self.isaac2urdf_order].clone().cpu()

        object_pose = rigid_state[:, 0, :7].clone().cpu().numpy()
        object_transform = np.repeat(
            np.eye(4, dtype=np.float64)[None],
            self.batch_size,
            axis=0,
        )
        object_transform[:, :3, 3] = object_pose[:, :3]
        object_transform[:, :3, :3] = Rotation.from_quat(
            object_pose[:, 3:7]
        ).as_matrix()
        object_inverse = np.linalg.inv(object_transform)

        def convert(q_world):
            q_numpy = q_world.numpy()
            hand_transform = np.repeat(
                np.eye(4, dtype=np.float64)[None],
                self.batch_size,
                axis=0,
            )
            hand_transform[:, :3, 3] = q_numpy[:, :3]
            hand_transform[:, :3, :3] = Rotation.from_euler(
                "XYZ",
                q_numpy[:, 3:6],
            ).as_matrix()
            relative = object_inverse @ hand_transform
            result = q_world.clone()
            result[:, :3] = torch.as_tensor(
                relative[:, :3, 3],
                dtype=result.dtype,
            )
            result[:, 3:6] = torch.as_tensor(
                Rotation.from_matrix(
                    relative[:, :3, :3]
                ).as_euler("XYZ"),
                dtype=result.dtype,
            )
            return result

        return convert(left_q_world), convert(right_q_world)

    def run_sim(self):
        for _ in range(self.grasp_step):
            self.gym.simulate(self.sim)
            self._draw()

        rigid_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_state = gymtorch.wrap_tensor(rigid_tensor).reshape(
            self.batch_size,
            self.rigid_body_num,
            13,
        )
        settled_pos = rigid_state[:, 0, :3].clone()
        settle_displacement = settled_pos.norm(dim=-1)
        left_q_final, right_q_final = (
            self._settled_hands_in_object_frame(rigid_state)
        )

        base_force = torch.zeros(
            [self.batch_size, self.rigid_body_num, 3],
            dtype=torch.float32,
        )
        forces = []
        for axis in range(3):
            positive = base_force.clone()
            positive[:, 0, axis] = self.object_force
            forces.append(positive)
        for axis in range(3):
            negative = base_force.clone()
            negative[:, 0, axis] = -self.object_force
            forces.append(negative)

        direction_displacements = []
        for step in range(self.steps_per_sec * 6):
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(
                    forces[step // self.steps_per_sec]
                ),
                None,
                gymapi.ENV_SPACE,
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._draw()
            if (step + 1) % self.steps_per_sec == 0:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                direction_pos = rigid_state[:, 0, :3].clone()
                direction_displacements.append(
                    (direction_pos - settled_pos).norm(dim=-1)
                )

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        final_pos = rigid_state[:, 0, :3].clone()
        disturbance_displacement = (final_pos - settled_pos).norm(dim=-1)
        direction_displacements = torch.stack(
            direction_displacements,
            dim=-1,
        )
        max_direction_displacement = direction_displacements.max(dim=-1).values
        success = (
            (settle_displacement <= 0.05)
            & (max_direction_displacement <= 0.02)
        )
        return {
            "success": success.cpu(),
            "settle_displacement": settle_displacement.cpu(),
            "disturbance_displacement": disturbance_displacement.cpu(),
            "direction_displacements": direction_displacements.cpu(),
            "max_direction_displacement": (
                max_direction_displacement.cpu()
            ),
            "left_q_final": left_q_final,
            "right_q_final": right_q_final,
        }

    def destroy(self):
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
