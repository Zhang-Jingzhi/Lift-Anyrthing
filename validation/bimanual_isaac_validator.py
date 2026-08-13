"""Isaac Gym validator for a configurable left/right dexterous-hand pair."""

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
        right_robot_name=None,
        right_joint_orders=None,
        gpu=0,
        use_gui=False,
        gravity=0.0,
        gravity_settle_step=100,
        staged_gravity=False,
        independent_directions=False,
        active_hands="both",
        support_height=None,
        support_thickness=0.02,
        fixture_during_closure=False,
        lift_height=0.0,
        lift_step=100,
        min_lift_height=0.03,
        robot_friction=3.0,
        object_friction=3.0,
        finger_effort_limit=None,
        contact_offset=0.01,
        max_gravity_displacement=0.02,
        max_direction_displacement=0.02,
        object_density=500.0,
        object_vhacd=False,
        object_vhacd_resolution=300000,
        object_vhacd_max_convex_hulls=64,
        object_vhacd_max_vertices=64,
        capture_contacts=False,
        steps_per_sec=100,
        grasp_step=100,
        debug_interval=0.01,
    ):
        self.gym = gymapi.acquire_gym()
        self.robot_name = robot_name
        self.left_robot_name = robot_name
        self.right_robot_name = right_robot_name or robot_name
        self.left_joint_orders = joint_orders
        self.right_joint_orders = right_joint_orders or joint_orders
        if len(self.left_joint_orders) != len(self.right_joint_orders):
            raise ValueError("Left and right hands must have matching DoF counts")
        # Kept for compatibility with older callers and report code.
        self.joint_orders = self.left_joint_orders
        self.batch_size = batch_size
        self.gpu = gpu
        self.gravity = float(gravity)
        self.gravity_settle_step = int(gravity_settle_step)
        self.staged_gravity = staged_gravity
        self.independent_directions = independent_directions
        if active_hands not in {"both", "left", "right"}:
            raise ValueError(f"Unknown active_hands mode: {active_hands}")
        self.active_hands = active_hands
        self.support_height = (
            None if support_height is None else float(support_height)
        )
        self.support_thickness = float(support_thickness)
        self.fixture_during_closure = bool(fixture_during_closure)
        self.lift_height = float(lift_height)
        self.lift_step = int(lift_step)
        self.min_lift_height = float(min_lift_height)
        self.robot_friction = robot_friction
        self.object_friction = object_friction
        self.finger_effort_limit = (
            None
            if finger_effort_limit is None
            else float(finger_effort_limit)
        )
        self.contact_offset = float(contact_offset)
        self.max_gravity_displacement = float(max_gravity_displacement)
        self.max_direction_displacement = float(max_direction_displacement)
        self.object_density = float(object_density)
        self.object_vhacd = bool(object_vhacd)
        self.object_vhacd_resolution = int(object_vhacd_resolution)
        self.object_vhacd_max_convex_hulls = int(
            object_vhacd_max_convex_hulls
        )
        self.object_vhacd_max_vertices = int(object_vhacd_max_vertices)
        self.capture_contacts = bool(capture_contacts)
        self.steps_per_sec = steps_per_sec
        self.grasp_step = grasp_step
        self.debug_interval = debug_interval

        self.envs = []
        self.object_handles = []
        self.left_handles = []
        self.right_handles = []
        self.support_handles = []
        self.robot_asset = None
        self.left_robot_asset = None
        self.right_robot_asset = None
        self.object_asset = None
        self.support_asset = None
        self.rigid_body_num = None
        self.object_force = None
        self.left_urdf2isaac_order = None
        self.left_isaac2urdf_order = None
        self.right_urdf2isaac_order = None
        self.right_isaac2urdf_order = None

        params = gymapi.SimParams()
        params.dt = 1 / steps_per_sec
        params.substeps = 2
        initial_gravity = 0.0 if staged_gravity else self.gravity
        params.gravity = gymapi.Vec3(0.0, 0.0, -initial_gravity)
        params.physx.use_gpu = True
        params.physx.solver_type = 1
        params.physx.num_position_iterations = 8
        params.physx.num_velocity_iterations = 0
        params.physx.contact_offset = self.contact_offset
        params.physx.rest_offset = 0.0
        # CUDA_VISIBLE_DEVICES remaps the PhysX compute ordinal, but Isaac
        # Gym's graphics ordinal is a physical-device index.  Reusing
        # ``self.gpu`` in headless multi-GPU jobs therefore opened an extra
        # graphics context on physical GPU 0 for every worker.  Headless
        # validation does not render, so disable the graphics device entirely
        # and keep the selected logical CUDA device only for PhysX compute.
        graphics_device = self.gpu if use_gui else -1
        self.sim = self.gym.create_sim(
            self.gpu,
            graphics_device,
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
        self.object_options.density = self.object_density
        # The runtime object URDF references a CoACD all-in-one OBJ.  Without
        # decomposition Isaac turns the entire non-convex mesh into one convex
        # hull, which creates false contacts across handles/wings and inflates
        # mass.  Keep this opt-in so legacy results remain reproducible.
        if self.object_vhacd:
            self.object_options.vhacd_enabled = True
            self.object_options.vhacd_params.resolution = (
                self.object_vhacd_resolution
            )
            self.object_options.vhacd_params.max_convex_hulls = (
                self.object_vhacd_max_convex_hulls
            )
            self.object_options.vhacd_params.max_num_vertices_per_ch = (
                self.object_vhacd_max_vertices
            )

        self.support_options = gymapi.AssetOptions()
        self.support_options.fix_base_link = True
        self.support_options.disable_gravity = True

    def set_asset(
        self,
        robot_path,
        robot_file,
        object_path,
        object_file,
        right_robot_path=None,
        right_robot_file=None,
    ):
        self.left_robot_asset = self.gym.load_asset(
            self.sim,
            robot_path,
            robot_file,
            self.robot_options,
        )
        self.right_robot_asset = self.gym.load_asset(
            self.sim,
            right_robot_path or robot_path,
            right_robot_file or robot_file,
            self.robot_options,
        )
        self.robot_asset = self.left_robot_asset
        self.object_asset = self.gym.load_asset(
            self.sim,
            object_path,
            object_file,
            self.object_options,
        )
        self.rigid_body_num = (
            self.gym.get_asset_rigid_body_count(self.object_asset)
            + self.gym.get_asset_rigid_body_count(self.left_robot_asset)
            + self.gym.get_asset_rigid_body_count(self.right_robot_asset)
        )
        if self.support_height is not None:
            self.support_asset = self.gym.create_box(
                self.sim,
                1.0,
                1.0,
                self.support_thickness,
                self.support_options,
            )
            self.rigid_body_num += self.gym.get_asset_rigid_body_count(
                self.support_asset
            )

    def _configure_robot(self, env, handle):
        properties = self.gym.get_actor_dof_properties(env, handle)
        properties["driveMode"].fill(gymapi.DOF_MODE_POS)
        properties["stiffness"].fill(1000)
        properties["damping"].fill(200)
        # The first six extended-URDF DoFs are the externally actuated wrist
        # pose.  Limit only the 16 physical Allegro finger joints here.
        if self.finger_effort_limit is not None:
            if len(properties) < 16:
                raise ValueError(
                    "Allegro asset has fewer than 16 finger DoFs"
                )
            properties["effort"][-16:] = self.finger_effort_limit
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
                self.left_robot_asset,
                gymapi.Transform(),
                f"left_robot_{env_idx}",
                env_idx,
            )
            right_handle = self.gym.create_actor(
                env,
                self.right_robot_asset,
                gymapi.Transform(),
                f"right_robot_{env_idx}",
                env_idx,
            )
            self.left_handles.append(left_handle)
            self.right_handles.append(right_handle)
            self._configure_robot(env, left_handle)
            self._configure_robot(env, right_handle)

            if self.support_asset is not None:
                support_pose = gymapi.Transform()
                support_pose.p.z = (
                    self.support_height - 0.5 * self.support_thickness
                )
                support_handle = self.gym.create_actor(
                    env,
                    self.support_asset,
                    support_pose,
                    f"support_{env_idx}",
                    env_idx,
                )
                self.support_handles.append(support_handle)

        properties = self.gym.get_actor_rigid_body_properties(
            self.envs[0],
            self.object_handles[0],
        )
        object_mass = sum(property.mass for property in properties)
        self.object_force = 0.5 * object_mass

        def build_order(handle, joint_orders):
            urdf2isaac = np.zeros(len(joint_orders), dtype=np.int32)
            isaac2urdf = np.zeros(len(joint_orders), dtype=np.int32)
            for urdf_idx, joint_name in enumerate(joint_orders):
                isaac_idx = self.gym.find_actor_dof_index(
                    self.envs[0],
                    handle,
                    joint_name,
                    gymapi.DOMAIN_ACTOR,
                )
                if isaac_idx < 0:
                    raise ValueError(f"Isaac asset is missing joint {joint_name}")
                urdf2isaac[isaac_idx] = urdf_idx
                isaac2urdf[urdf_idx] = isaac_idx
            return urdf2isaac, isaac2urdf

        (
            self.left_urdf2isaac_order,
            self.left_isaac2urdf_order,
        ) = build_order(self.left_handles[0], self.left_joint_orders)
        (
            self.right_urdf2isaac_order,
            self.right_isaac2urdf_order,
        ) = build_order(self.right_handles[0], self.right_joint_orders)

    def _set_hand_state(
        self,
        env,
        handle,
        initial_q,
        target_q,
        urdf2isaac_order,
    ):
        states = self.gym.get_actor_dof_states(
            env,
            handle,
            gymapi.STATE_ALL,
        ).copy()
        states["pos"] = initial_q[urdf2isaac_order]
        self.gym.set_actor_dof_states(
            env,
            handle,
            states,
            gymapi.STATE_ALL,
        )
        targets = target_q[urdf2isaac_order]
        self.gym.set_actor_dof_position_targets(env, handle, targets)

    def set_actor_pose_dof(self, left_q, right_q):
        self.gym.prepare_sim(self.sim)
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_state = gymtorch.wrap_tensor(root_tensor)
        root_state[:] = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        )
        if self.support_asset is not None:
            actor_count = 4
            support_states = root_state.reshape(
                self.batch_size,
                actor_count,
                13,
            )[:, 3]
            support_states[:, 2] = (
                self.support_height - 0.5 * self.support_thickness
            )
        self.gym.set_actor_root_state_tensor(self.sim, root_tensor)

        left_outer, left_inner = controller(self.left_robot_name, left_q)
        right_outer, right_inner = controller(self.right_robot_name, right_q)
        # Moving an ablated hand several metres away is equivalent to
        # removing it while preserving identical tensor layouts and solver
        # settings across the three experimental conditions.
        if self.active_hands == "left":
            right_outer = right_outer.clone()
            right_inner = right_inner.clone()
            right_outer[:, 0] += 5.0
            right_inner[:, 0] += 5.0
        elif self.active_hands == "right":
            left_outer = left_outer.clone()
            left_inner = left_inner.clone()
            left_outer[:, 0] += 5.0
            left_inner[:, 0] += 5.0
        self.left_inner_q = left_inner.clone()
        self.right_inner_q = right_inner.clone()
        for index, env in enumerate(self.envs):
            self._set_hand_state(
                env,
                self.left_handles[index],
                left_outer[index],
                left_inner[index],
                self.left_urdf2isaac_order,
            )
            self._set_hand_state(
                env,
                self.right_handles[index],
                right_outer[index],
                right_inner[index],
                self.right_urdf2isaac_order,
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
            len(self.left_joint_orders) + len(self.right_joint_orders),
            2,
        )[:, :, 0]
        left_dof_count = len(self.left_joint_orders)
        left_q_world = dof_state[
            :,
            :left_dof_count,
        ][:, self.left_isaac2urdf_order].clone().cpu()
        right_q_world = dof_state[
            :,
            left_dof_count:,
        ][:, self.right_isaac2urdf_order].clone().cpu()

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

    def _simulate(self, steps, hold_object=False):
        root_tensor = None
        root_state = None
        actor_count = 4 if self.support_asset is not None else 3
        if hold_object:
            root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            root_state = gymtorch.wrap_tensor(root_tensor).reshape(
                self.batch_size,
                actor_count,
                13,
            )
        for _ in range(steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if hold_object:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                object_state = root_state[:, 0]
                object_state[:] = 0.0
                object_state[:, 6] = 1.0
                self.gym.set_actor_root_state_tensor(
                    self.sim,
                    root_tensor,
                )
            self._draw()

    def _set_gravity(self, magnitude):
        params = self.gym.get_sim_params(self.sim)
        params.gravity = gymapi.Vec3(0.0, 0.0, -float(magnitude))
        self.gym.set_sim_params(self.sim, params)

    def _set_lift_targets(self, fraction=1.0):
        left_targets = self.left_inner_q.clone()
        right_targets = self.right_inner_q.clone()
        offset = self.lift_height * float(fraction)
        left_targets[:, 2] += offset
        right_targets[:, 2] += offset
        for index, env in enumerate(self.envs):
            self.gym.set_actor_dof_position_targets(
                env,
                self.left_handles[index],
                left_targets[index][self.left_urdf2isaac_order],
            )
            self.gym.set_actor_dof_position_targets(
                env,
                self.right_handles[index],
                right_targets[index][self.right_urdf2isaac_order],
            )

    def _execute_lift(self):
        """Move both wrists upward together along a smooth common path."""
        if self.lift_step <= 0:
            self._set_lift_targets(1.0)
            return
        for step in range(1, self.lift_step + 1):
            self._set_lift_targets(step / self.lift_step)
            self._simulate(1)

    def _remove_support(self, actor_root_state, actor_root_tensor):
        if self.support_asset is None:
            return
        support_states = actor_root_state.reshape(
            self.batch_size,
            4,
            13,
        )[:, 3]
        support_states[:, 2] -= 2.0
        support_states[:, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(
            self.sim,
            actor_root_tensor,
        )

    @staticmethod
    def _contact_vec3(value):
        """Convert either a RigidContact Vec3 record or object to floats."""
        try:
            return [float(value[name]) for name in ("x", "y", "z")]
        except (IndexError, KeyError, TypeError, ValueError):
            return [float(value.x), float(value.y), float(value.z)]

    @staticmethod
    def _contact_field(contact, *names):
        available = contact.dtype.names or ()
        for name in names:
            if name in available:
                return contact[name]
        raise KeyError(
            f"RigidContact fields {names} are unavailable; got {available}"
        )

    def _capture_object_hand_contacts(self, rigid_state):
        """Record all PhysX object/hand contacts in every environment."""
        snapshots = []
        for env_index, env in enumerate(self.envs):
            body_map = {}
            actors = (
                ("object", self.object_handles[env_index]),
                ("left", self.left_handles[env_index]),
                ("right", self.right_handles[env_index]),
            )
            for actor_name, handle in actors:
                names = self.gym.get_actor_rigid_body_names(env, handle)
                for local_index, body_name in enumerate(names):
                    body_index = self.gym.get_actor_rigid_body_index(
                        env,
                        handle,
                        local_index,
                        gymapi.DOMAIN_ENV,
                    )
                    body_map[int(body_index)] = (actor_name, body_name)

            rows = []
            contacts = self.gym.get_env_rigid_contacts(env)
            for contact_index, contact in enumerate(contacts):
                body0 = int(contact["body0"])
                body1 = int(contact["body1"])
                actor0, name0 = body_map.get(body0, ("other", str(body0)))
                actor1, name1 = body_map.get(body1, ("other", str(body1)))
                actors_in_contact = {actor0, actor1}
                if "object" not in actors_in_contact or not (
                    {"left", "right"} & actors_in_contact
                ):
                    continue

                local0 = np.asarray(
                    self._contact_vec3(
                        self._contact_field(
                            contact, "local_pos0", "localPos0"
                        )
                    ),
                    dtype=np.float64,
                )
                local1 = np.asarray(
                    self._contact_vec3(
                        self._contact_field(
                            contact, "local_pos1", "localPos1"
                        )
                    ),
                    dtype=np.float64,
                )

                def world_point(body_index, local_point):
                    state = rigid_state[env_index, body_index].cpu().numpy()
                    return (
                        Rotation.from_quat(state[3:7]).apply(local_point)
                        + state[:3]
                    )

                world0 = world_point(body0, local0)
                world1 = world_point(body1, local1)
                rows.append(
                    {
                        "contact_index": contact_index,
                        "body0_index": body0,
                        "body1_index": body1,
                        "actor0": actor0,
                        "actor1": actor1,
                        "body0_name": name0,
                        "body1_name": name1,
                        "local_pos0_m": local0.tolist(),
                        "local_pos1_m": local1.tolist(),
                        "world_pos0_m": world0.tolist(),
                        "world_pos1_m": world1.tolist(),
                        "world_midpoint_m": ((world0 + world1) * 0.5).tolist(),
                        "normal": self._contact_vec3(contact["normal"]),
                        "initial_overlap_m": float(
                            self._contact_field(
                                contact, "initial_overlap", "initialOverlap"
                            )
                        ),
                        "min_dist_m": float(
                            self._contact_field(
                                contact, "min_dist", "minDist"
                            )
                        ),
                        "normal_impulse": float(contact["lambda"]),
                        "friction": float(contact["friction"]),
                    }
                )
            snapshots.append(rows)
        return snapshots

    def run_sim(self, gravity_only=False):
        self._simulate(
            self.grasp_step,
            hold_object=self.fixture_during_closure,
        )

        rigid_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        rigid_state = gymtorch.wrap_tensor(rigid_tensor).reshape(
            self.batch_size,
            self.rigid_body_num,
            13,
        )
        actor_root_state = gymtorch.wrap_tensor(actor_root_tensor)
        dof_state = gymtorch.wrap_tensor(dof_tensor)
        closure_pos = rigid_state[:, 0, :3].clone()
        settle_displacement = closure_pos.norm(dim=-1)
        closure_contacts = None
        if self.capture_contacts:
            closure_contacts = self._capture_object_hand_contacts(rigid_state)

        if self.lift_height > 0.0:
            self._execute_lift()
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        else:
            self._remove_support(actor_root_state, actor_root_tensor)
        lifted_pos = rigid_state[:, 0, :3].clone()
        lift_displacement_z = lifted_pos[:, 2] - closure_pos[:, 2]
        lifted_contacts = None
        if self.capture_contacts:
            lifted_contacts = self._capture_object_hand_contacts(rigid_state)

        if self.gravity > 0.0:
            if self.staged_gravity:
                self._set_gravity(self.gravity)
            self._simulate(self.gravity_settle_step)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        settled_pos = rigid_state[:, 0, :3].clone()
        gravity_displacement = (settled_pos - lifted_pos).norm(dim=-1)
        settled_contacts = None
        if self.capture_contacts:
            settled_contacts = self._capture_object_hand_contacts(rigid_state)
        left_q_final, right_q_final = (
            self._settled_hands_in_object_frame(rigid_state)
        )

        settled_actor_root = actor_root_state.clone()
        settled_dof_state = dof_state.clone()

        def restore_settled_state():
            actor_root_state.copy_(settled_actor_root)
            dof_state.copy_(settled_dof_state)
            self.gym.set_actor_root_state_tensor(
                self.sim,
                actor_root_tensor,
            )
            self.gym.set_dof_state_tensor(self.sim, dof_tensor)

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
        if not gravity_only and self.independent_directions:
            for force in forces:
                restore_settled_state()
                for _ in range(self.steps_per_sec):
                    self.gym.apply_rigid_body_force_tensors(
                        self.sim,
                        gymtorch.unwrap_tensor(force),
                        None,
                        gymapi.ENV_SPACE,
                    )
                    self.gym.simulate(self.sim)
                    self.gym.fetch_results(self.sim, True)
                    self._draw()
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                direction_pos = rigid_state[:, 0, :3].clone()
                direction_displacements.append(
                    (direction_pos - settled_pos).norm(dim=-1)
                )
            restore_settled_state()
        elif not gravity_only:
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
        else:
            direction_displacements = [
                torch.zeros(self.batch_size) for _ in forces
            ]

        direction_displacements = torch.stack(direction_displacements, dim=-1)
        max_direction_displacement = direction_displacements.max(dim=-1).values
        disturbance_displacement = direction_displacements[:, -1]
        gravity_success = (
            (settle_displacement <= 0.05)
            & (gravity_displacement <= self.max_gravity_displacement)
        )
        if self.lift_height > 0.0:
            gravity_success &= lift_displacement_z >= self.min_lift_height
        success = gravity_success & (
            max_direction_displacement <= self.max_direction_displacement
        )
        result = {
            "success": success.cpu(),
            "gravity_success": gravity_success.cpu(),
            "settle_displacement": settle_displacement.cpu(),
            "gravity_displacement": gravity_displacement.cpu(),
            "lift_displacement_z": lift_displacement_z.cpu(),
            "disturbance_displacement": disturbance_displacement.cpu(),
            "direction_displacements": direction_displacements.cpu(),
            "max_direction_displacement": (
                max_direction_displacement.cpu()
            ),
            "left_q_final": left_q_final,
            "right_q_final": right_q_final,
            "object_mass_kg": torch.full(
                (self.batch_size,),
                float(self.object_force / 0.5),
            ),
        }
        if self.capture_contacts:
            result.update(
                {
                    "closure_contacts": closure_contacts,
                    "lifted_contacts": lifted_contacts,
                    "settled_contacts": settled_contacts,
                }
            )
        return result

    def destroy(self):
        for env in self.envs:
            self.gym.destroy_env(env)
        self.gym.destroy_sim(self.sim)
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        del self.gym
