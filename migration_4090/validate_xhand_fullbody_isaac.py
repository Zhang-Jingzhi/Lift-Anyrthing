#!/usr/bin/env python3
"""Isaac Gym physical gate for a Tianji full-body dual-XHand pose.

The robot is one fixed-base 51-DoF actor.  The object starts on a table,
both hands close simultaneously, the robot is lifted by 50 mm, gravity is
settled, six independent disturbance directions are applied, and two fresh
single-hand ablations are run.  This validator writes a machine-readable
record and never modifies source URDFs or meshes.
"""
import argparse
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi, gymtorch
import torch

URDF = Path(os.environ.get("XHAND_FULLBODY_URDF", ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf"))
IK_URDF = Path(os.environ.get("XHAND_FULLBODY_IK_URDF", ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"))


def torch_load(path):
    """Load with both the Isaac Gym torch 1.8 and modern torch APIs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def indent_xml(root):
    # xml.etree.ElementTree.indent was added in Python 3.9.
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")


def write_object_urdf(sample, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "object_runtime_density50.urdf"
    if not path.exists():
        mesh = Path(sample["object_mesh_path"]).resolve()
        root = ET.Element("robot", {"name": "xhand_runtime_object"})
        link = ET.SubElement(root, "link", {"name": "object"})
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", {"filename": str(mesh)})
        collision = ET.SubElement(link, "collision")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", {"filename": str(mesh)})
        indent_xml(root)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_ablation_urdf(side, out_dir):
    """Remove collision geometry on one side for a true single-hand test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"robot_{side}_only_fixed_ik.urdf"
    if path.exists():
        return path
    root = ET.parse(IK_URDF).getroot()
    disabled_prefix = "right_" if side == "left" else "left_"
    for link in root.findall("link"):
        name = link.get("name", "")
        if name.startswith(disabled_prefix + "j") or name.startswith(disabled_prefix + "hand"):
            for node in list(link.findall("collision")):
                link.remove(node)
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def object_pose(sample):
    t = np.asarray(sample["object_pose_world"], dtype=np.float32)
    tr = gymapi.Transform()
    tr.p = gymapi.Vec3(float(t[0, 3]), float(t[1, 3]), float(t[2, 3]))
    tr.r = gymapi.Quat(0, 0, 0, 1)
    return tr


def make_sim(gpu, gravity=9.8):
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = float(os.environ.get("XHAND_SIM_DT", "0.01"))
    params.substeps = int(os.environ.get("XHAND_SIM_SUBSTEPS", "2"))
    params.gravity = gymapi.Vec3(0.0, 0.0, -gravity)
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.physx.num_position_iterations = int(os.environ.get("XHAND_PHYSX_POS_ITERS", "8"))
    params.physx.num_velocity_iterations = int(os.environ.get("XHAND_PHYSX_VEL_ITERS", "1"))
    # Explicitly collect all contacts; Preview 4 otherwise may return an
    # empty get_env_rigid_contacts() list even while GPU PhysX resolves them.
    params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    params.physx.contact_offset = 0.002
    params.physx.rest_offset = 0.0
    sim = gym.create_sim(gpu, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create GPU PhysX simulation")
    return gym, sim


def load_robot(gym, sim, robot_file):
    options = gymapi.AssetOptions()
    # The waist/head/wheels are fixed in fixed_ik.urdf.  Keep the actor base
    # movable so the lift phase can translate the whole robot without relying
    # on undefined root-state writes to a fixed-base PhysX actor.
    options.fix_base_link = False
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset = gym.load_asset(sim, str(robot_file.parent), robot_file.name, options)
    if asset is None:
        raise RuntimeError(f"Isaac Gym failed to load robot {robot_file}")
    return asset


def load_object(gym, sim, object_urdf, density):
    options = gymapi.AssetOptions()
    options.override_com = True
    options.override_inertia = True
    options.density = float(density)
    options.vhacd_enabled = True
    options.vhacd_params.resolution = 1_000_000
    options.vhacd_params.max_convex_hulls = 128
    options.vhacd_params.max_num_vertices_per_ch = 64
    asset = gym.load_asset(sim, str(object_urdf.parent), object_urdf.name, options)
    if asset is None:
        raise RuntimeError(f"Isaac Gym failed to load object {object_urdf}")
    return asset


def set_asset_friction(gym, asset, friction):
    props = gym.get_asset_rigid_shape_properties(asset)
    for prop in props:
        prop.friction = float(friction)
        prop.restitution = 0.0
    gym.set_asset_rigid_shape_properties(asset, props)


def contact_records(gym, env, object_handle, robot_handle, body_names):
    mapping = {}
    for actor, handle, label in (("object", object_handle, "object"), ("robot", robot_handle, "robot")):
        for i, name in enumerate(gym.get_actor_rigid_body_names(env, handle)):
            # get_env_rigid_contacts() reports simulation-global rigid-body
            # indices in Preview 4. DOMAIN_ENV silently maps the object to 0,
            # so real object-hand contacts were previously discarded.
            idx = gym.get_actor_rigid_body_index(env, handle, i, gymapi.DOMAIN_SIM)
            mapping[int(idx)] = (label, name)
    rows = []
    raw_contacts = gym.get_env_rigid_contacts(env)
    for contact in raw_contacts:
        b0, b1 = int(contact["body0"]), int(contact["body1"])
        a0, n0 = mapping.get(b0, ("other", str(b0)))
        a1, n1 = mapping.get(b1, ("other", str(b1)))
        if "object" not in (a0, a1) or "robot" not in (a0, a1):
            continue
        robot_link = n1 if a1 == "robot" else n0
        if robot_link.startswith("left_hand"):
            side = "left"
        elif robot_link.startswith("right_hand"):
            side = "right"
        else:
            side = "arm_or_body"
        names = getattr(contact, "dtype", None)
        fields = set(getattr(names, "names", ()) or ())
        min_dist = float(contact["min_dist"]) if "min_dist" in fields else 0.0
        initial_overlap = float(contact["initial_overlap"]) if "initial_overlap" in fields else 0.0
        friction = float(contact["friction"]) if "friction" in fields else 0.0
        rows.append({
            "side": side,
            "robot_link": robot_link,
            "normal_impulse": float(contact["lambda"]),
            "min_dist_m": min_dist,
            "initial_overlap_m": initial_overlap,
            "friction": friction,
            "body0": b0,
            "body1": b1,
            "body0_label": a0,
            "body1_label": a1,
        })
    return rows


def simulate_sample(sample, mode, output_dir, gpu=0, density=50.0, friction=1.0,
                    lift_height=0.05, lift_steps=100, gravity_steps=500,
                    disturbance_steps=100, hand_close_scale=1.0,
                    left_hand_close_scale=None, right_hand_close_scale=None,
                    hand_stiffness=400.0, hand_damping=80.0,
                    hand_effort_override=None, closure_mode="position",
                    preclosure_settle_steps=50, approach_steps=0,
                    approach_fraction=1.0,
                    squeeze_steps=0, squeeze_fraction=None,
                    left_squeeze_fraction=None, right_squeeze_fraction=None,
                    closure_steps=120,
                    closure_settle_steps=0, hold_object_during_closure=False):
    robot_file = IK_URDF if mode == "both" else write_ablation_urdf(mode, output_dir / "runtime_assets")
    object_urdf = write_object_urdf(sample, output_dir / "runtime_assets" / sample["object_name"])
    gym, sim = make_sim(gpu)
    env = None
    try:
        robot_asset = load_robot(gym, sim, robot_file)
        object_asset = load_object(gym, sim, object_urdf, density)
        set_asset_friction(gym, robot_asset, friction)
        set_asset_friction(gym, object_asset, friction)
        # The tabletop must be a true fixed support.  A movable, gravity-free
        # box forms an ill-conditioned two-dynamic-body contact with a light
        # rounded object in Preview 4 and can launch it on the first step.
        # We leave the fixed table in place after lift: a passing grasp keeps
        # the object within 10 mm of its lifted pose, still at least 40 mm
        # above the tabletop.  A failed grasp that falls back to the table
        # therefore cannot satisfy the gravity-displacement gate.
        support_options = gymapi.AssetOptions(); support_options.fix_base_link = True; support_options.disable_gravity = True
        support_size_x = float(os.environ.get("XHAND_SUPPORT_SIZE_X", "1.0"))
        support_size_y = float(os.environ.get("XHAND_SUPPORT_SIZE_Y", "2.2"))
        support_asset = gym.create_box(sim, support_size_x, support_size_y, 0.02, support_options)
        env = gym.create_env(sim, gymapi.Vec3(-2, -2, -2), gymapi.Vec3(2, 2, 2), 1)
        object_handle = gym.create_actor(env, object_asset, object_pose(sample), "object", 0)
        # Put object/robot/table in the same collision group; the ablation
        # URDFs, rather than actor groups, control which hand is disabled.
        # The actor is created at the neutral world pose. The explicit frame
        # correction is applied below to the acquired root-state tensor after
        # PhysX has initialized the articulation; setting create_actor's
        # Transform alone is reset by Preview 4 for this asset.
        # A non-zero actor filter suppresses collisions among shapes that
        # share that bit.  Object/table actors keep filter 0, so robot-object
        # and robot-table contacts remain enabled.  This is useful for the
        # imported Tianji URDF whose adjacent fixed links otherwise overlap
        # and prevent the XHand joints from closing at native effort limits.
        robot_collision_filter = int(os.environ.get("XHAND_ROBOT_COLLISION_FILTER", "0"))
        robot_handle = gym.create_actor(
            env, robot_asset, gymapi.Transform(), "robot", 0,
            robot_collision_filter,
        )
        support = gymapi.Transform()
        support.p.x = float(os.environ.get("XHAND_SUPPORT_X", "0.50"))
        support.p.y = float(os.environ.get("XHAND_SUPPORT_Y", "0.0"))
        support.p.z = float(os.environ.get("XHAND_SUPPORT_Z", "0.71"))
        support_handle = gym.create_actor(env, support_asset, support, "table", 0)

        # Persist actor/shape diagnostics: an empty hand collision asset can
        # otherwise look like a failed grasp while all kinematics are valid.
        actor_shape_diagnostics = {}
        for label, handle in (("object", object_handle), ("robot", robot_handle), ("table", support_handle)):
            props = gym.get_actor_rigid_shape_properties(env, handle)
            actor_shape_diagnostics[label] = {
                "rigid_body_count": int(gym.get_actor_rigid_body_count(env, handle)),
                "rigid_shape_count": int(gym.get_actor_rigid_shape_count(env, handle)),
                "shape_filters": [int(p.filter) for p in props],
            }

        dof_names = list(gym.get_actor_dof_names(env, robot_handle))
        q_target = np.asarray(sample["full_body_q"], dtype=np.float32)
        full_names = list(sample["joint_names"])
        q_by_name = {n: float(v) for n, v in zip(full_names, q_target)}
        if not set(dof_names).issubset(set(full_names)):
            missing = sorted(set(dof_names) - set(full_names)); extra = sorted(set(full_names) - set(dof_names))
            raise RuntimeError(f"Isaac/pose DoF mismatch missing={missing} extra={extra}")
        q_target_asset = np.asarray([q_by_name[n] for n in dof_names], dtype=np.float32)
        if hand_close_scale != 1.0 or left_hand_close_scale is not None or right_hand_close_scale is not None:
            for i, n in enumerate(dof_names):
                if "_hand_" in n:
                    scale = float(hand_close_scale)
                    if left_hand_close_scale is not None and n.startswith("left_hand_"):
                        scale = float(left_hand_close_scale)
                    if right_hand_close_scale is not None and n.startswith("right_hand_"):
                        scale = float(right_hand_close_scale)
                    q_target_asset[i] *= scale
        arm_indices = np.asarray([i for i, n in enumerate(dof_names) if "_hand_" not in n], dtype=np.int64)
        props = gym.get_actor_dof_properties(env, robot_handle)
        # Close-scale sweeps may request values beyond the URDF limits.
        # Clamp commands explicitly instead of relying on backend-specific
        # drive saturation.
        q_target_asset = np.clip(
            q_target_asset,
            np.asarray(props["lower"], dtype=np.float32),
            np.asarray(props["upper"], dtype=np.float32),
        )
        native_efforts = np.asarray(props["effort"], dtype=np.float32).copy()
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(1500.0); props["damping"].fill(120.0)
        hand_mask = np.asarray(["_hand_" in n for n in dof_names], dtype=bool)
        props["stiffness"][hand_mask] = float(hand_stiffness)
        props["damping"][hand_mask] = float(hand_damping)
        if hand_effort_override is not None:
            # Keep the URDF effort limits in the report, but allow an explicit
            # controller override for diagnosing whether closure is control- or
            # geometry-limited. The default remains the asset's native limits.
            props["effort"][hand_mask] = float(hand_effort_override)
        gym.set_actor_dof_properties(env, robot_handle, props)
        states = gym.get_actor_dof_states(env, robot_handle, gymapi.STATE_ALL).copy()
        q_open = q_target_asset.copy()
        if "pregrasp_full_body_q" in sample:
            pregrasp_values = {
                n: float(v)
                for n, v in zip(full_names, sample["pregrasp_full_body_q"])
            }
            q_open = np.asarray([pregrasp_values[n] for n in dof_names], dtype=np.float32)
        for i, n in enumerate(dof_names):
            if "_hand_" in n:
                q_open[i] = 0.0
        active_arm_target = q_open[arm_indices].copy()
        states["pos"] = q_open; states["vel"] = 0.0
        gym.set_actor_dof_states(env, robot_handle, states, gymapi.STATE_ALL)
        # Start from the genuinely open hand pose.  The previous validator
        # immediately commanded q_target_asset and merely waited for
        # ``closure_steps`` while the object was frozen.  That accumulated a
        # large contact impulse and launched rounded objects as soon as they
        # were released.  Position mode now receives an interpolated target
        # below, so closure_steps really denotes a slow simultaneous close.
        gym.set_actor_dof_position_targets(env, robot_handle, q_open)
        gym.prepare_sim(sim)

        root_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        dof_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
        rigid_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)
        object_root_initial = root_tensor[0].clone()
        actor_root_initial = root_tensor.clone()
        root_alignment = torch.tensor([
            float(os.environ.get("XHAND_ROBOT_BASE_X", "0.0027")),
            float(os.environ.get("XHAND_ROBOT_BASE_Y", "0.00042")),
            float(os.environ.get("XHAND_ROBOT_BASE_Z", "0.0524")),
        ], dtype=root_tensor.dtype)
        root_tensor[1, :3] += root_alignment
        root_tensor[1, 7:13] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
        gym.refresh_actor_root_state_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)
        actor_root_initial = root_tensor.clone()
        robot_root_initial = root_tensor[1].clone()

        def ee_positions():
            """Read the two EE rigid-body positions at the current simulator state."""
            gym.refresh_rigid_body_state_tensor(sim)
            names = list(gym.get_actor_rigid_body_names(env, robot_handle))
            out = {}
            for name in ("left_hand_ee_link", "right_hand_ee_link"):
                if name in names:
                    body_index = gym.get_actor_rigid_body_index(env, robot_handle, names.index(name), gymapi.DOMAIN_ENV)
                    out[name] = rigid_tensor[body_index, :3].cpu().tolist()
            return out

        def hold_arm_pose():
            gym.refresh_dof_state_tensor(sim)
            dof_tensor[arm_indices, 0] = torch.as_tensor(active_arm_target, dtype=dof_tensor.dtype)
            dof_tensor[arm_indices, 1] = 0.0
            if closure_mode == "teleport":
                hand_indices = np.flatnonzero(hand_mask)
                dof_tensor[hand_indices, 0] = torch.as_tensor(q_target_asset[hand_indices], dtype=dof_tensor.dtype)
                dof_tensor[hand_indices, 1] = 0.0
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))

        closure_object_state = object_root_initial.clone()

        def step(count, hold_object=False, hold_robot_root=False, robot_root_state=None):
            for _ in range(count):
                gym.simulate(sim); gym.fetch_results(sim, True)
                # Refresh first: set_actor_root_state_tensor writes the whole
                # actor array. Without this refresh, the object's dynamic
                # state is stale and gets reset every time the robot root is
                # prescribed, making lift/gravity appear frozen.
                gym.refresh_actor_root_state_tensor(sim)
                hold_arm_pose()
                if hold_object:
                    # Freeze at the actual pre-closure settled pose, not the
                    # initial airborne actor pose. Resetting to the latter at
                    # closure introduced a large artificial jump and erased
                    # otherwise valid mesh-filtered contacts.
                    root_tensor[0].copy_(closure_object_state)
                if hold_robot_root:
                    # The robot actor is intentionally movable for the later
                    # prescribed lift. During closure, however, its root must
                    # be held fixed; otherwise contact impulses move the whole
                    # robot and contaminate the EE/frame comparison.
                    root_tensor[1].copy_(robot_root_initial if robot_root_state is None else robot_root_state)
                if hold_object or hold_robot_root:
                    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
                gym.refresh_actor_root_state_tensor(sim); gym.refresh_dof_state_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)

        ee_before_closure = ee_positions()
        # One synchronization step is required before comparing Isaac's
        # rigid-body frames with Pyroki FK. Before the first simulate/fetch
        # cycle Isaac may still expose the asset's default DOF configuration.
        # When closure hold is requested, freeze the object from the very
        # first synchronization step. Otherwise the already-closed hand can
        # push the actor before the hold state is captured, invalidating the
        # pose-to-mesh alignment used by the raw-triangle filter.
        step(1, hold_object=hold_object_during_closure, hold_robot_root=True)
        ee_after_sync = ee_positions()
        # Let the object and open hands settle naturally on/around the table
        # before making contact.  In particular, do not freeze the object:
        # the physical gate must expose an unstable initial pose immediately.
        step(max(0, int(preclosure_settle_steps)), hold_object=hold_object_during_closure, hold_robot_root=True)
        preclosure_pos = root_tensor[0, :3].clone()
        closure_object_state.copy_(root_tensor[0])
        closure_object_state[7:13] = 0.0
        approach_trace = []
        closure_trace = []
        squeeze_trace = []
        if closure_mode == "position":
            has_pregrasp = "pregrasp_full_body_q" in sample
            if has_pregrasp and int(approach_steps) > 0:
                approach_steps = max(1, int(approach_steps))
                approach_fraction = float(np.clip(approach_fraction, 0.0, 1.0))
                approach_arm_target = (
                    q_open[arm_indices]
                    + approach_fraction
                    * (q_target_asset[arm_indices] - q_open[arm_indices])
                )
                for i in range(1, approach_steps + 1):
                    alpha = float(i) / float(approach_steps)
                    q_command = q_open.copy()
                    q_command[arm_indices] = (
                        q_open[arm_indices]
                        + alpha * (approach_arm_target - q_open[arm_indices])
                    )
                    active_arm_target[:] = q_command[arm_indices]
                    gym.set_actor_dof_position_targets(env, robot_handle, q_command)
                    step(1, hold_robot_root=True)
                    if i == 1 or i % 10 == 0 or i == approach_steps:
                        trace_contacts = contact_records(
                            gym, env, object_handle, robot_handle, dof_names
                        )
                        approach_trace.append({
                            "step": int(i),
                            "object_position_world": root_tensor[0, :3].cpu().tolist(),
                            "object_displacement_from_pregrasp_m": float(
                                torch.norm(root_tensor[0, :3] - preclosure_pos)
                            ),
                            "contact_links": sorted({
                                row["robot_link"] for row in trace_contacts
                                if row["normal_impulse"] > 0
                            }),
                        })
                closure_steps = max(1, int(closure_steps))
            for i in range(1, closure_steps + 1):
                alpha = float(i) / float(closure_steps)
                if has_pregrasp and int(approach_steps) > 0:
                    q_command = q_target_asset.copy()
                    q_command[arm_indices] = approach_arm_target
                    q_command[hand_mask] = (
                        q_open[hand_mask]
                        + alpha * (q_target_asset[hand_mask] - q_open[hand_mask])
                    )
                else:
                    q_command = q_open + alpha * (q_target_asset - q_open)
                active_arm_target[:] = q_command[arm_indices]
                gym.set_actor_dof_position_targets(env, robot_handle, q_command)
                step(1, hold_object=hold_object_during_closure, hold_robot_root=True)
                if i == 1 or i % 10 == 0 or i == closure_steps:
                    trace_contacts = contact_records(
                        gym, env, object_handle, robot_handle, dof_names
                    )
                    closure_trace.append({
                        "step": int(i),
                        "object_position_world": root_tensor[0, :3].cpu().tolist(),
                        "object_displacement_from_pregrasp_m": float(
                            torch.norm(root_tensor[0, :3] - preclosure_pos)
                        ),
                        "contact_links": sorted({
                            row["robot_link"] for row in trace_contacts
                            if row["normal_impulse"] > 0
                        }),
                    })
            if has_pregrasp and int(squeeze_steps) > 0:
                squeeze_steps = max(1, int(squeeze_steps))
                final_squeeze_fraction = float(np.clip(
                    approach_fraction if squeeze_fraction is None else squeeze_fraction,
                    approach_fraction,
                    1.0,
                ))
                squeeze_arm_target = (
                    q_open[arm_indices]
                    + final_squeeze_fraction
                    * (q_target_asset[arm_indices] - q_open[arm_indices])
                )
                if left_squeeze_fraction is not None or right_squeeze_fraction is not None:
                    for local_i, dof_i in enumerate(arm_indices):
                        name = dof_names[int(dof_i)]
                        side_fraction = final_squeeze_fraction
                        if name.startswith("left_") and left_squeeze_fraction is not None:
                            side_fraction = float(np.clip(
                                left_squeeze_fraction, approach_fraction, 1.0
                            ))
                        elif name.startswith("right_") and right_squeeze_fraction is not None:
                            side_fraction = float(np.clip(
                                right_squeeze_fraction, approach_fraction, 1.0
                            ))
                        squeeze_arm_target[local_i] = (
                            q_open[int(dof_i)]
                            + side_fraction
                            * (q_target_asset[int(dof_i)] - q_open[int(dof_i)])
                        )
                squeeze_start = active_arm_target.copy()
                for i in range(1, squeeze_steps + 1):
                    alpha = float(i) / float(squeeze_steps)
                    q_command = q_target_asset.copy()
                    q_command[arm_indices] = (
                        squeeze_start
                        + alpha * (squeeze_arm_target - squeeze_start)
                    )
                    active_arm_target[:] = q_command[arm_indices]
                    gym.set_actor_dof_position_targets(env, robot_handle, q_command)
                    step(1, hold_object=hold_object_during_closure, hold_robot_root=True)
                    if i == 1 or i % 10 == 0 or i == squeeze_steps:
                        trace_contacts = contact_records(
                            gym, env, object_handle, robot_handle, dof_names
                        )
                        squeeze_trace.append({
                            "step": int(i),
                            "object_position_world": root_tensor[0, :3].cpu().tolist(),
                            "object_displacement_from_pregrasp_m": float(
                                torch.norm(root_tensor[0, :3] - preclosure_pos)
                            ),
                            "contact_links": sorted({
                                row["robot_link"] for row in trace_contacts
                                if row["normal_impulse"] > 0
                            }),
                        })
        else:
            # Teleport remains available only as an explicit diagnostic mode.
            active_arm_target[:] = q_target_asset[arm_indices]
            gym.set_actor_dof_position_targets(env, robot_handle, q_target_asset)
            step(max(1, int(closure_steps)), hold_object=hold_object_during_closure, hold_robot_root=True)
        if closure_settle_steps > 0:
            # Release the object onto the tabletop while the robot root stays
            # fixed. This dissipates controller/contact energy accumulated
            # while the object was held during closure. Without this phase a
            # curved object can pass a one-step gate and then be launched when
            # the prescribed lift first releases the frozen root state.
            step(int(closure_settle_steps), hold_robot_root=True)
        ee_after_closure = ee_positions()
        robot_root_after_closure = root_tensor[1].clone()
        closure_contacts = contact_records(gym, env, object_handle, robot_handle, dof_names)
        raw_closure_contacts = gym.get_env_rigid_contacts(env)
        body_names = list(gym.get_actor_rigid_body_names(env, robot_handle))
        body_index_map = {
            "object": int(gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_SIM)),
            "robot": {name: int(gym.get_actor_rigid_body_index(env, robot_handle, i, gymapi.DOMAIN_SIM)) for i, name in enumerate(body_names)},
        }
        body_state_positions = {}
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        actual_q_after_closure = dof_tensor[:, 0].cpu().numpy().tolist()
        q_error_order = np.argsort(np.abs(np.asarray(actual_q_after_closure) - q_target_asset))[::-1][:5]
        for name in ("left_hand_ee_link", "right_hand_ee_link"):
            if name in body_names:
                body_index = gym.get_actor_rigid_body_index(env, robot_handle, body_names.index(name), gymapi.DOMAIN_ENV)
                body_state_positions[name] = rigid_tensor[body_index, :3].cpu().tolist()
        for name in body_names:
            if name.startswith(("left_hand", "right_hand")):
                body_index = gym.get_actor_rigid_body_index(env, robot_handle, body_names.index(name), gymapi.DOMAIN_ENV)
                body_state_positions[name] = rigid_tensor[body_index, :3].cpu().tolist()
        closure_pos = root_tensor[0, :3].clone()
        closure_pose = root_tensor[0, :7].clone()
        closure_has_both = all(
            any(
                row["side"] == side and row["normal_impulse"] > 0
                for row in closure_contacts
            )
            for side in ("left", "right")
        )
        closure_displacement_m = float(torch.norm(closure_pos - preclosure_pos))
        closure_viable = bool(closure_has_both and closure_displacement_m <= 0.15)
        early_reject_stage = None
        if not closure_viable:
            early_reject_stage = "closure"
        if closure_viable and os.environ.get("XHAND_DROP_SUPPORT_BEFORE_LIFT", "0") == "1":
            # Diagnostic branch: remove the tabletop before the prescribed
            # lift so we can distinguish a failed grasp from a table contact.
            root_tensor[2, 2] = -2.0
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
            step(5, hold_object=True, hold_robot_root=True)
        # Translate the prescribed robot base upward; PhysX tests whether the
        # closed XHands can carry the object off the table.
        lift_root_final = robot_root_initial.clone()
        if closure_viable:
            for i in range(1, lift_steps + 1):
                lift_root = robot_root_initial.clone()
                lift_root[2] = robot_root_initial[2] + lift_height * i / lift_steps
                # The base is intentionally kinematically prescribed for this
                # pose-level lift test. Clear residual free-body velocity so the
                # actor does not acquire an unbounded impulse between updates.
                lift_root[7:13] = 0.0
                root_tensor[1].copy_(lift_root)
                gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
                step(1, hold_robot_root=True, robot_root_state=lift_root)
                lift_root_final = lift_root
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)
            lifted_pos = root_tensor[0, :3].clone()
            ee_after_lift = ee_positions()
            lifted_contacts = contact_records(gym, env, object_handle, robot_handle, dof_names)
        else:
            lifted_pos = closure_pos.clone()
            ee_after_lift = ee_after_closure
            lifted_contacts = []

        lift_has_both = all(
            any(
                row["side"] == side and row["normal_impulse"] > 0
                for row in lifted_contacts
            )
            for side in ("left", "right")
        )
        lift_displacement_m = float(lifted_pos[2] - closure_pos[2])
        lift_viable = bool(
            closure_viable and lift_has_both and lift_displacement_m >= 0.03
        )
        if closure_viable and not lift_viable:
            early_reject_stage = "lift"

        object_mass = sum(float(p.mass) for p in gym.get_actor_rigid_body_properties(env, object_handle))
        if lift_viable:
            # Drop the support well below the workspace before gravity settling.
            root_tensor[2, 2] = -2.0
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
            step(gravity_steps, hold_robot_root=True, robot_root_state=lift_root_final)
            gym.refresh_actor_root_state_tensor(sim); gym.refresh_rigid_body_state_tensor(sim)
            settled_pos = root_tensor[0, :3].clone()
            settled_contacts = contact_records(gym, env, object_handle, robot_handle, dof_names)

            object_body_idx = gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_ENV)
            settled_root = root_tensor.clone(); settled_dof = dof_tensor.clone()
            directions = []
            force = torch.zeros((1, gym.get_env_rigid_body_count(env), 3), dtype=torch.float32)
            for axis in range(3):
                for sign in (1.0, -1.0):
                    root_tensor.copy_(settled_root); dof_tensor.copy_(settled_dof)
                    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor)); gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))
                    force.zero_(); force[0, object_body_idx, axis] = sign * 0.5 * object_mass
                    for _ in range(disturbance_steps):
                        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(force), None, gymapi.ENV_SPACE)
                        gym.simulate(sim); gym.fetch_results(sim, True)
                        # Refresh the dynamic object state before prescribing the
                        # robot root.  set_actor_root_state_tensor writes every
                        # actor; without this refresh it also rewrites the object
                        # to the stale settled pose and makes disturbances appear
                        # to have exactly zero displacement.
                        gym.refresh_actor_root_state_tensor(sim)
                        root_tensor[1].copy_(lift_root_final)
                        root_tensor[1, 7:13] = 0.0
                        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_tensor))
                    gym.refresh_actor_root_state_tensor(sim)
                    directions.append(float(torch.norm(root_tensor[0, :3] - settled_root[0, :3])))
        else:
            settled_pos = lifted_pos.clone()
            settled_contacts = lifted_contacts
            # Preserve a definite failed value without spending gravity and
            # disturbance simulation on a candidate that already failed lift.
            directions = [1.0] * 6

        def side_metrics(rows):
            out = {}
            for side in ("left", "right"):
                sr = [r for r in rows if r["side"] == side and r["normal_impulse"] > 0]
                out[f"{side}_contact_links"] = sorted({r["robot_link"] for r in sr})
                out[f"{side}_contact_count"] = len(sr)
                # Preview 4 often exposes min_dist=0 for resolved GPU PhysX
                # contacts while initial_overlap retains the actual overlap.
                # Use both fields so deep initial interpenetration cannot be
                # misreported as zero penetration.
                min_dist_pen = max([max(0.0, -r["min_dist_m"]) * 1000 for r in sr] or [0.0])
                initial_overlap_pen = max([max(0.0, r["initial_overlap_m"]) * 1000 for r in sr] or [0.0])
                out[f"{side}_max_min_dist_penetration_mm"] = min_dist_pen
                out[f"{side}_max_initial_overlap_mm"] = initial_overlap_pen
                out[f"{side}_max_penetration_mm"] = max(min_dist_pen, initial_overlap_pen)
            return out

        expected_ee = np.asarray(sample.get("achieved_tcp_positions_world", []), dtype=np.float32)
        synced_ee = np.asarray([
            ee_after_closure.get("left_hand_ee_link", [np.nan, np.nan, np.nan]),
            ee_after_closure.get("right_hand_ee_link", [np.nan, np.nan, np.nan]),
        ], dtype=np.float32)
        if expected_ee.shape == (2, 3):
            ee_residual = synced_ee - expected_ee
            ee_residual_norm_mm = (np.linalg.norm(ee_residual, axis=1) * 1000.0).tolist()
        else:
            ee_residual = np.full((2, 3), np.nan, dtype=np.float32)
            ee_residual_norm_mm = [float("nan"), float("nan")]

        metrics = {
            "mode": mode,
            "closure": side_metrics(closure_contacts),
            "raw_closure_contact_count": int(len(raw_closure_contacts)),
            "raw_closure_contacts": [
                {"body0": int(c["body0"]), "body1": int(c["body1"]), "lambda": float(c["lambda"])}
                for c in raw_closure_contacts
            ],
            "body_index_map": body_index_map,
            "isaac_hand_ee_positions_world": body_state_positions,
            "isaac_hand_ee_positions_before_closure_world": ee_before_closure,
            "isaac_hand_ee_positions_after_sync_world": ee_after_sync,
            "isaac_hand_ee_positions_after_closure_world": ee_after_closure,
            "isaac_hand_ee_positions_after_lift_world": ee_after_lift,
            "pyroki_achieved_ee_positions_world": expected_ee.tolist(),
            "isaac_minus_pyroki_ee_residual_m": ee_residual.tolist(),
            "isaac_minus_pyroki_ee_residual_norm_mm": ee_residual_norm_mm,
            "robot_root_position_before_closure_world": robot_root_initial[:3].cpu().tolist(),
            "robot_root_position_after_closure_world": robot_root_after_closure[:3].cpu().tolist(),
            "robot_root_position_after_lift_world": lift_root_final[:3].cpu().tolist(),
            "max_dof_position_error_after_closure": float(np.max(np.abs(np.asarray(actual_q_after_closure) - q_target_asset))),
            "actual_dof_positions_after_closure": {
                n: float(v) for n, v in zip(dof_names, actual_q_after_closure)
            },
            "commanded_dof_positions": {
                n: float(v) for n, v in zip(dof_names, q_target_asset)
            },
            "largest_dof_errors": [{"name": dof_names[int(i)], "actual": float(actual_q_after_closure[int(i)]), "target": float(q_target_asset[int(i)])} for i in q_error_order],
            "lifted": side_metrics(lifted_contacts),
            "settled": side_metrics(settled_contacts),
            "closure_object_z_m": float(closure_pos[2]),
            "closure_displacement_from_pregrasp_m": closure_displacement_m,
            "early_reject_stage": early_reject_stage,
            "object_root_position_after_preclosure_settle_world": preclosure_pos.cpu().tolist(),
            "approach_trace": approach_trace,
            "closure_trace": closure_trace,
            "squeeze_trace": squeeze_trace,
            # Use the state captured immediately after closure.  Reading
            # root_tensor here reports the much later post-disturbance state
            # and previously made stable closure look like an instant launch.
            "object_root_position_after_closure_world": closure_pos.cpu().tolist(),
            "object_root_pose_after_closure_world": closure_pose.cpu().tolist(),
            "object_root_position_after_lift_world": lifted_pos.cpu().tolist(),
            "object_root_position_after_gravity_world": settled_pos.cpu().tolist(),
            "lift_object_displacement_m": float(lifted_pos[2] - closure_pos[2]),
            "gravity_displacement_m": float(torch.norm(settled_pos - lifted_pos)),
            "six_direction_displacements_m": directions,
            "object_mass_kg": object_mass,
            "gravity_m_s2": 9.8,
            "object_density_kg_m3": density,
            "robot_friction": friction,
            "object_friction": friction,
            "effort_limits": {n: float(v) for n, v in zip(dof_names, native_efforts) if "_hand_" in n},
            "controller": {
                "closure_mode": closure_mode,
                "preclosure_settle_steps": int(preclosure_settle_steps),
                "approach_steps": int(approach_steps),
                "approach_fraction": float(approach_fraction),
                "squeeze_steps": int(squeeze_steps),
                "squeeze_fraction": None if squeeze_fraction is None else float(squeeze_fraction),
                "left_squeeze_fraction": None if left_squeeze_fraction is None else float(left_squeeze_fraction),
                "right_squeeze_fraction": None if right_squeeze_fraction is None else float(right_squeeze_fraction),
                "closure_steps": int(closure_steps),
                "closure_settle_steps": int(closure_settle_steps),
                "hand_stiffness": float(hand_stiffness),
                "hand_damping": float(hand_damping),
                "hand_effort_override": None if hand_effort_override is None else float(hand_effort_override),
            },
            "isaac_gpu": int(gpu),
            "isaac_robot_root_alignment_offset_xyz": root_alignment.cpu().tolist(),
            "support_size_xy_m": [support_size_x, support_size_y],
            "support_position_xyz_m": [
                float(support.p.x), float(support.p.y), float(support.p.z)
            ],
            "robot_collision_filter": int(robot_collision_filter),
            "actor_shape_diagnostics": actor_shape_diagnostics,
        }
        left_closure = metrics["closure"]["left_contact_count"] > 0
        right_closure = metrics["closure"]["right_contact_count"] > 0
        left_lift = metrics["lifted"]["left_contact_count"] > 0
        right_lift = metrics["lifted"]["right_contact_count"] > 0
        max_penetration_mm = float(os.environ.get("XHAND_MAX_PENETRATION_MM", "2.0"))
        penetration_pass = all(
            metrics[phase][f"{side}_max_penetration_mm"] <= max_penetration_mm
            for phase in ("closure", "lifted", "settled")
            for side in ("left", "right")
        )
        metrics["max_allowed_penetration_mm"] = max_penetration_mm
        metrics["penetration_pass"] = penetration_pass
        metrics["physical_pass"] = bool(left_closure and right_closure and left_lift and right_lift and penetration_pass and metrics["lift_object_displacement_m"] >= 0.03 and metrics["gravity_displacement_m"] <= 0.01 and max(directions) <= 0.015)
        return metrics
    finally:
        if env is not None:
            gym.destroy_env(env)
        gym.destroy_sim(sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "migration_4090/results/xhand_fullbody_grasps_v1_final/baseline.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--modes", default="both,left,right", help="Comma-separated physical modes to run")
    parser.add_argument("--hand-close-scale", type=float, default=1.0)
    parser.add_argument("--left-hand-close-scale", type=float, default=None)
    parser.add_argument("--right-hand-close-scale", type=float, default=None)
    parser.add_argument("--hand-stiffness", type=float, default=400.0)
    parser.add_argument("--hand-damping", type=float, default=80.0)
    parser.add_argument("--hand-effort-override", type=float, default=None)
    parser.add_argument("--closure-mode", choices=("position", "teleport"), default="position")
    parser.add_argument("--preclosure-settle-steps", type=int, default=50)
    parser.add_argument("--approach-steps", type=int, default=0)
    parser.add_argument("--approach-fraction", type=float, default=1.0)
    parser.add_argument("--squeeze-steps", type=int, default=0)
    parser.add_argument("--squeeze-fraction", type=float, default=None)
    parser.add_argument("--left-squeeze-fraction", type=float, default=None)
    parser.add_argument("--right-squeeze-fraction", type=float, default=None)
    parser.add_argument("--closure-steps", type=int, default=120)
    parser.add_argument("--closure-settle-steps", type=int, default=0)
    parser.add_argument("--hold-object-during-closure", action="store_true")
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=50.0)
    parser.add_argument("--lift-steps", type=int, default=100)
    parser.add_argument("--gravity-steps", type=int, default=500)
    parser.add_argument("--disturbance-steps", type=int, default=100)
    args = parser.parse_args()
    payload = torch_load(args.dataset)
    sample = payload["samples"][args.sample_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "xhand_fullbody_isaac_physical_smoke_v1", "dataset": str(args.dataset), "sample_index": args.sample_index, "sample": {"method": sample["method"], "object_name": sample["object_name"], "schedule_index": sample.get("schedule_index")}, "runs": {}}
    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    for mode in modes:
        print(f"RUN mode={mode} object={sample['object_name']} index={args.sample_index}", flush=True)
        report["runs"][mode] = simulate_sample(
            sample, mode, args.output.parent, gpu=args.gpu,
            friction=args.friction,
            density=args.density,
            hand_close_scale=args.hand_close_scale,
            left_hand_close_scale=args.left_hand_close_scale,
            right_hand_close_scale=args.right_hand_close_scale,
            hand_stiffness=args.hand_stiffness,
            hand_damping=args.hand_damping,
            hand_effort_override=args.hand_effort_override,
            closure_mode=args.closure_mode,
            preclosure_settle_steps=args.preclosure_settle_steps,
            approach_steps=args.approach_steps,
            approach_fraction=args.approach_fraction,
            squeeze_steps=args.squeeze_steps,
            squeeze_fraction=args.squeeze_fraction,
            left_squeeze_fraction=args.left_squeeze_fraction,
            right_squeeze_fraction=args.right_squeeze_fraction,
            closure_steps=args.closure_steps,
            closure_settle_steps=args.closure_settle_steps,
            hold_object_during_closure=args.hold_object_during_closure,
            lift_steps=args.lift_steps,
            gravity_steps=args.gravity_steps,
            disturbance_steps=args.disturbance_steps,
        )
        print(json.dumps(report["runs"][mode], indent=2), flush=True)
    both = report["runs"].get("both", {"physical_pass": False})
    report["physical_pass"] = bool(
        both["physical_pass"]
        and ("left" not in report["runs"] or not report["runs"]["left"]["physical_pass"])
        and ("right" not in report["runs"] or not report["runs"]["right"]["physical_pass"])
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"physical_pass": report["physical_pass"], "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
