#!/usr/bin/env python3
"""Generate fixed-waist dual-XHand full-body kinematic grasp poses.

This generator uses the assembled approximate LinkHou+XHand URDF, fixes waist,
head, and wheel joints, and solves both TCP positions simultaneously. It is a
kinematic/mesh-filtered dataset generator; IsaacLab physical validation is
recorded as pending until a matching full-body IsaacLab asset is available.
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import jax_dataclasses as jdc
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.pyroki_ik import PyrokiRetarget


URDF = Path(os.environ.get("XHAND_FULLBODY_URDF", ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1.urdf"))
IK_URDF = Path(os.environ.get("XHAND_FULLBODY_IK_URDF", ROOT / "migration_4090/assets/xhand_fullbody_approx_v1/linkhou_xhand_fullbody_approx_v1_fixed_ik.urdf"))
CATALOG = ROOT / "migration_4090/results/large_random_6x8_v1/catalog.json"
SCHEDULE = ROOT / "migration_4090/results/large_random_6x8_v1/shared_schedule_6x100.json"
METHODS = ("baseline", "bidex_v3")
TARGET_LINKS = tuple(os.environ.get("XHAND_TARGET_LINKS", "L_tcp,R_tcp").split(","))
FIXED_NAMES = {
    "waist_extend2", "waist_yaw", "waist_pitch", "head_yaw", "head_pitch",
    "front_rightwheel_re", "front_rightwheel_go", "front_leftwheel_re",
    "front_leftwheel_go", "back_leftwheel_re", "back_leftwheel_go",
    "back_rightwheel_re", "back_rightwheel_go",
}
TABLE_Z = float(os.environ.get("XHAND_TABLE_Z", "0.72"))


def load_catalog():
    catalogue = json.loads(CATALOG.read_text())
    by_name = {row["object_name"]: row for row in catalogue["objects"]}
    schedule = json.loads(SCHEDULE.read_text())["entries"]
    return by_name, schedule


def mesh_for_object(row):
    path = Path(row["collision_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return path, mesh


def orient_and_scale_mesh(mesh, seed, target_y=(1.283, 1.287), scale_mode="long_axis"):
    """Make a non-destructive grasp mesh whose long axis spans the two wrists.

    The Tianji wrists remain far apart with the waist fixed.  The source meshes
    are kept untouched; this returns a transformed copy and the transform
    metadata written with every sample.  Existing size variants still provide
    the requested random size distribution.
    """
    source = mesh.copy()
    ext = np.asarray(source.extents, dtype=np.float64)
    long_axis = int(np.argmax(ext))
    remaining = [i for i in range(3) if i != long_axis]
    # Keep the smallest remaining dimension vertical.  This places the
    # tabletop-supported center in the fixed-waist arm workspace; using the
    # second-longest axis here made airplane/pitcher centers unreachable.
    vertical_axis = remaining[int(np.argmin(ext[remaining]))]
    x_axis = next(i for i in remaining if i != vertical_axis)
    # rows map old xyz to new xyz: long -> y, second-longest -> z.
    perm = np.zeros((3, 3), dtype=np.float64)
    perm[0, x_axis] = 1.0
    perm[1, long_axis] = 1.0
    perm[2, vertical_axis] = 1.0
    if np.linalg.det(perm) < 0:
        perm[0] *= -1.0
    centre = np.asarray(source.bounds.mean(axis=0), dtype=np.float64)
    vertices = (np.asarray(source.vertices) - centre) @ perm.T
    target = float(np.random.default_rng(seed).uniform(*target_y))
    scale = target / max(float(ext[long_axis]), 1e-6)
    if scale_mode == "uniform":
        # Preserve the source object's proportions.  This is used when the
        # robot is placed farther behind the object and the arms can converge
        # on a compact, box-like or round object without stretching one axis.
        vertices *= scale
        scale_xyz = [scale, scale, scale]
    else:
        scale_y = max(1.0, scale)
        vertices[:, 1] *= scale_y
        scale_xyz = [1.0, scale_y, 1.0]
    source.vertices = vertices
    source.apply_translation(-source.bounds.mean(axis=0))
    return source, {
        "axis_permutation_old_to_new": perm.tolist(),
        "source_extents_m": ext.tolist(),
        "grasp_scale_xyz": scale_xyz,
        "scale_mode": scale_mode,
        "target_y_extent_m": target,
    }


def ensure_fixed_ik_urdf():
    if IK_URDF.exists():
        return
    root = ET.parse(URDF).getroot()
    for joint in root.findall("joint"):
        if joint.get("name") in FIXED_NAMES:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in ("axis", "limit", "dynamics"):
                    joint.remove(child)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(IK_URDF, encoding="utf-8", xml_declaration=True)


def fixed_solver():
    ensure_fixed_ik_urdf()
    solver = PyrokiRetarget(str(IK_URDF), list(TARGET_LINKS))
    robot = solver.robot
    return solver, robot


def expand_ik_q(q, ik_names, full_names):
    expanded = np.zeros(len(full_names), dtype=np.float32)
    full_index = {name: index for index, name in enumerate(full_names)}
    for value, name in zip(q, ik_names):
        expanded[full_index[name]] = value
    return expanded


def hand_mesh_points(urdf_path: Path, count_per_link: int = 24):
    root = ET.parse(urdf_path).getroot()
    points = {}
    for link in root.findall("link"):
        name = link.get("name", "")
        if "hand" not in name:
            continue
        mesh_node = link.find("collision/geometry/mesh")
        if mesh_node is None:
            mesh_node = link.find("visual/geometry/mesh")
        if mesh_node is None:
            continue
        filename = mesh_node.get("filename")
        if not filename:
            continue
        path = Path(filename)
        if not path.is_absolute():
            path = urdf_path.parent / path
        if not path.is_file():
            continue
        mesh = trimesh.load_mesh(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        points[name] = np.asarray(mesh.sample(count_per_link), dtype=np.float32)
    return points


def link_points(robot, q, link_points, link_indices):
    poses = robot.forward_kinematics(jnp.asarray(q))
    points_world = []
    for name, points in link_points.items():
        pose = np.asarray(jaxlie.SE3(poses[link_indices[name]]).as_matrix())
        homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        )
        points_world.append((homogeneous @ pose.T)[:, :3])
    return np.concatenate(points_world, axis=0)


def target_batch(mesh, method, seeds, table_z, wrist_side=0.69, left_wrist_side=None, right_wrist_side=None, wrist_z_offset=0.0, wrist_x=None):
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    extents = bounds[1] - bounds[0]
    center_world = np.asarray((bounds[0] + bounds[1]) * 0.5, dtype=np.float32)
    # With the waist fixed, both wrists are reachable only in a narrow band
    # around +/-0.69 m.  The transformed meshes span that band; keeping the
    # TCP targets there avoids the previous unreachable +/-0.4 m requests.
    left_side = float(wrist_side if left_wrist_side is None else left_wrist_side)
    right_side = float(wrist_side if right_wrist_side is None else right_wrist_side)
    targets = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if method == "baseline":
            x = float(wrist_x) if wrist_x is not None else float(center_world[0] + rng.uniform(-0.008, 0.008))
            z = float(center_world[2] + float(wrist_z_offset) + rng.uniform(-0.006, 0.006))
            left_y = left_side + float(rng.uniform(-0.003, 0.003))
            right_y = -right_side + float(rng.uniform(-0.003, 0.003))
        else:
            # BiDex-style paired region initialization: symmetric, opposed,
            # and slightly elevated away from the tabletop.
            x = float(wrist_x) if wrist_x is not None else float(center_world[0] + rng.uniform(-0.006, 0.006))
            z = float(center_world[2] + float(wrist_z_offset) + rng.uniform(-0.004, 0.004))
            lateral = float(rng.uniform(-0.003, 0.003))
            left_y, right_y = left_side + lateral, -right_side + lateral
        targets.append([[x, left_y, z], [x, right_y, z]])
    return np.asarray(targets, dtype=np.float32), center_world, extents


def geometric_metrics(mesh, hand_query, left_points, right_points, contact_m=0.020):
    query = trimesh.proximity.ProximityQuery(mesh)
    all_points = np.concatenate([left_points, right_points], axis=0)
    try:
        _, distances, _ = query.on_surface(all_points)
        inside = mesh.contains(all_points)
    except Exception:
        distances = np.full(len(all_points), np.nan, dtype=np.float32)
        inside = np.zeros(len(all_points), dtype=bool)
    left_d, right_d = distances[: len(left_points)], distances[len(left_points):]
    left_inside, right_inside = inside[: len(left_points)], inside[len(left_points):]
    left_pen = float(left_d[left_inside].max()) if left_inside.any() else 0.0
    right_pen = float(right_d[right_inside].max()) if right_inside.any() else 0.0
    left_contact = int(np.count_nonzero(left_d <= contact_m)) if np.isfinite(left_d).all() else 0
    right_contact = int(np.count_nonzero(right_d <= contact_m)) if np.isfinite(right_d).all() else 0
    clearance = float(np.min(np.linalg.norm(left_points[:, None] - right_points[None, :], axis=2)))
    return {
        "left_penetration_mm": left_pen * 1000.0,
        "right_penetration_mm": right_pen * 1000.0,
        "left_contact_points": left_contact,
        "right_contact_points": right_contact,
        "hand_clearance_mm": clearance * 1000.0,
        "mesh_query_valid": bool(np.isfinite(distances).all()),
    }


def solve_group(solver, initial_q, target_positions):
    solved = np.asarray(
        solver.solve_retarget(
            jnp.asarray(initial_q), jnp.asarray(target_positions)
        )
    )
    robot = solver.robot
    poses = np.asarray(robot.forward_kinematics(jnp.asarray(solved)))
    target_indices = [robot.links.names.index(name) for name in TARGET_LINKS]
    actual = np.asarray(jaxlie.SE3(poses[:, target_indices]).translation())
    errors = np.linalg.norm(actual - target_positions, axis=2)
    return solved, poses, actual, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-per-object", type=int, default=100)
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scheduled", action="store_true", help="Generate one pose for each shared-schedule entry (600/method).")
    parser.add_argument("--schedule-start", type=int, default=0)
    parser.add_argument("--schedule-end", type=int, default=None)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--wrist-side", type=float, default=0.34,
                        help="Absolute +/-Y wrist TCP target in metres; compact default keeps the object near its natural proportions")
    parser.add_argument("--left-wrist-side", type=float)
    parser.add_argument("--right-wrist-side", type=float)
    parser.add_argument("--wrist-z-offset", type=float, default=0.03,
                        help="Offset both TCP targets vertically from the object center (m); used for physical candidate search.")
    parser.add_argument("--target-y-extent", type=float, default=None,
                        help="Override transformed object's Y extent in metres for physical candidate search.")
    parser.add_argument("--scale-mode", choices=("long_axis", "uniform"), default="uniform",
                        help="Stretch only the grasp axis (legacy) or preserve object proportions with uniform scaling.")
    parser.add_argument("--object-x", type=float, default=0.4,
                        help="Place the object forward/backward relative to the fixed robot base (m).")
    parser.add_argument("--wrist-x", type=float, default=None,
                        help="Fix both TCP x targets in metres; omit to keep the original random x sampling.")
    parser.add_argument(
        "--accept-ik-candidates",
        action="store_true",
        help="Optimization-only mode: keep finite, limit-valid IK poses even when mesh contact filtering fails; Isaac must validate them later.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not URDF.is_file():
        raise FileNotFoundError(URDF)
    by_name, schedule = load_catalog()
    methods = (args.method,) if args.method else METHODS
    scheduled_entries = None
    if args.scheduled:
        schedule = schedule[args.schedule_start:args.schedule_end]
        if args.smoke:
            scheduled_entries = schedule[: max(1, args.smoke_count)]
        else:
            scheduled_entries = schedule
    else:
        selected_names = [args.object_name] if args.object_name else [schedule[0]["object_name"]]
        if args.smoke:
            selected_names = selected_names[:1]
            target_count = args.smoke_count
        else:
            target_count = args.target_per_object
    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver, robot = fixed_solver()
    # Preserve the full URDF joint order without constructing a second JAX IK solver.
    full_root = ET.parse(URDF).getroot()
    full_names = [j.get("name") for j in full_root.findall("joint") if j.get("type") != "fixed"]
    ik_names = list(robot.joints.actuated_names)
    robot = solver.robot
    left_points = hand_mesh_points(URDF, count_per_link=64)
    right_points = hand_mesh_points(URDF, count_per_link=64)
    left_points = {k: v for k, v in left_points.items() if k.startswith("left_hand_")}
    right_points = {k: v for k, v in right_points.items() if k.startswith("right_hand_")}
    link_indices = {name: robot.links.names.index(name) for name in (*left_points, *right_points)}
    print(f"URDF={URDF} ik_urdf={IK_URDF} ik_actuated={robot.joints.num_actuated_joints} full_actuated={len(full_names)} fixed={len(FIXED_NAMES)} left_links={len(left_points)} right_links={len(right_points)}", flush=True)

    mesh_dir = args.output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    mesh_cache = {}
    warm_q_cache = {}

    def prepare_mesh(object_name, seed):
        if object_name not in mesh_cache:
            mesh_path, mesh_local = mesh_for_object(by_name[object_name])
            # Compact XHand generation preserves the source proportions and
            # randomizes the overall object size.  The legacy 1.28 m long-axis
            # range made boxes and rounded objects look like long bars merely
            # to bridge widely separated wrists.
            y_extent = (
                (float(args.target_y_extent), float(args.target_y_extent))
                if args.target_y_extent is not None
                else (0.50, 0.65)
            )
            transformed, transform_meta = orient_and_scale_mesh(
                mesh_local, seed, target_y=y_extent, scale_mode=args.scale_mode
            )
            if "toy_airplane" in object_name:
                # The airplane collision mesh has thin wing gaps at the two
                # wrist contact bands.  Its convex collision envelope retains
                # the source silhouette while providing a closed graspable
                # surface; the source asset remains unchanged.
                transformed = trimesh.convex.convex_hull(transformed)
                transform_meta["collision_envelope"] = "convex_hull"
            safe = object_name.replace("+", "__").replace("/", "_")
            out_mesh = mesh_dir / f"{safe}__xhand_grasp.obj"
            if not out_mesh.exists():
                transformed.export(out_mesh)
            mesh_cache[object_name] = (out_mesh, transformed, transform_meta)
        out_mesh, mesh_centered, transform_meta = mesh_cache[object_name]
        mesh = mesh_centered.copy()
        translation = np.array([
            float(args.object_x), 0.0, TABLE_Z - float(mesh.bounds[0, 2])
        ], dtype=np.float32)
        mesh.apply_translation(translation)
        return out_mesh, mesh, translation, transform_meta

    def make_sample(method, object_name, seed):
        mesh_path, mesh, object_translation, transform_meta = prepare_mesh(object_name, seed)
        targets, _, _ = target_batch(mesh, method, [seed], TABLE_Z, args.wrist_side, args.left_wrist_side, args.right_wrist_side, args.wrist_z_offset, args.wrist_x)
        if "toy_airplane" in object_name:
            targets[:, :, 1] *= 0.68 / 0.69
            targets[:, :, 2] -= 0.05
        elif "cracker_box" in object_name or "power_drill" in object_name:
            targets[:, :, 2] -= 0.03
        initial = np.zeros((1, robot.joints.num_actuated_joints), dtype=np.float32)
        for i, name in enumerate(robot.joints.actuated_names):
            if "_hand_" in name:
                initial[:, i] = 0.55
        # First solve a canonical +/-0.69 m target for this object and reuse it
        # as a warm start.  This removes the zero-initialization branch that
        # previously rejected otherwise reachable samples.
        if object_name not in warm_q_cache:
            centre = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float32)
            # Always build the warm start at the canonical symmetric wrist
            # separation. The requested left/right perturbations are applied
            # only to the actual target solve; using a perturbed target for
            # warm-start construction can make a nearby, otherwise reachable
            # search point fail before optimization begins.
            left_side = args.wrist_side
            right_side = args.wrist_side
            canonical = np.asarray([[[centre[0], left_side, centre[2]], [centre[0], -right_side, centre[2]]]], dtype=np.float32)
            if "toy_airplane" in object_name:
                canonical[:, :, 1] *= 0.68 / 0.69
                canonical[:, :, 2] -= 0.05
            elif "cracker_box" in object_name or "power_drill" in object_name:
                canonical[:, :, 2] -= 0.03
            canonical_q, _, _, canonical_err = solve_group(solver, initial, canonical)
            if float(canonical_err.max()) > 0.002:
                if args.accept_ik_candidates:
                    # In optimization mode a deliberately perturbed wrist
                    # target may make the canonical warm-start unreachable.
                    # Do not spend 100 retries re-solving the same failed
                    # warm start; fall back to the neutral initial state and
                    # let the target solve decide feasibility.
                    warm_q_cache[object_name] = initial[0].copy()
                else:
                    return None
            else:
                warm_q_cache[object_name] = canonical_q[0].copy()
        initial[0] = warm_q_cache[object_name]
        solved, poses, actual, errors = solve_group(solver, initial, targets)
        q = solved[0]
        finite = bool(np.isfinite(q).all() and np.isfinite(errors[0]).all())
        limit_ok = bool(np.all(q >= np.asarray(robot.joints.lower_limits) - 1e-4) and np.all(q <= np.asarray(robot.joints.upper_limits) + 1e-4))
        if not (finite and limit_ok and (float(errors[0].max()) <= 0.002 or args.accept_ik_candidates)):
            return None
        lp = link_points(robot, q, left_points, link_indices)
        rp = link_points(robot, q, right_points, link_indices)
        metrics = geometric_metrics(mesh, None, lp, rp)
        geometry_pass = bool(metrics["mesh_query_valid"] and metrics["left_contact_points"] > 0 and metrics["right_contact_points"] > 0 and metrics["left_penetration_mm"] <= 2.0 and metrics["right_penetration_mm"] <= 2.0 and metrics["hand_clearance_mm"] > 2.0)
        if not geometry_pass and not args.accept_ik_candidates:
            print(f"GEOMETRY_REJECT method={method} object={object_name} seed={seed} metrics={metrics}", flush=True)
            return None
        return {
            "method": method, "object_name": object_name, "object_mesh_path": str(mesh_path),
            "object_pose_world": [[1, 0, 0, float(object_translation[0])], [0, 1, 0, float(object_translation[1])], [0, 0, 1, float(object_translation[2])], [0, 0, 0, 1]],
            "object_transform": transform_meta, "target_tcp_positions_world": targets[0].tolist(), "achieved_tcp_positions_world": actual[0].tolist(),
            "full_body_q": expand_ik_q(q, ik_names, full_names).tolist(), "joint_names": full_names, "ik_position_error_m": errors[0].tolist(),
            "fixed_waist": True, "geometry_pass": geometry_pass, "isaaclab_physical_validated": False, "metrics": metrics, "seed": int(seed),
        }

    all_samples = []
    per_object = []
    if scheduled_entries is not None:
        for method in methods:
            method_samples = []
            failed = []
            for index, entry in enumerate(scheduled_entries):
                object_name = entry["object_name"]
                sample = None
                for retry in range(args.max_retries):
                    sample = make_sample(method, object_name, int(entry["sample_seed"] + retry * 100003 + (0 if method == "baseline" else 50000000)))
                    if sample is not None:
                        break
                if sample is None:
                    failed.append({"index": index, "object_name": object_name})
                else:
                    sample["schedule_index"] = args.schedule_start + index
                    sample["sample_index"] = entry["sample_index"]
                    method_samples.append(sample)
                if (index + 1) % 10 == 0:
                    print(f"{method}: accepted={len(method_samples)}/{index + 1} failed={len(failed)}", flush=True)
            if failed:
                raise RuntimeError(f"{method}: {len(failed)} scheduled entries failed; first={failed[:3]}")
            suffix = "" if (args.schedule_start == 0 and args.schedule_end is None) else f"__{args.schedule_start:04d}_{args.schedule_end:04d}"
            output = args.output_dir / f"{method}{suffix}.pt"
            if output.exists() and not args.force:
                raise FileExistsError(output)
            torch.save({"schema": "xhand_fullbody_grasp_pose_v1", "samples": method_samples}, output)
            all_samples.extend(method_samples)
            per_object.append({"method": method, "accepted": len(method_samples), "output": str(output)})
            print(f"DONE {method}: {len(method_samples)}", flush=True)
        summary = {"schema": "xhand_fullbody_grasp_pose_summary_v1", "urdf": str(URDF), "methods": list(methods), "schedule_entries": len(scheduled_entries), "total": len(all_samples), "per_method": per_object, "isaaclab_physical_validation": "pending"}
    else:
        for method in methods:
            for object_name in selected_names:
                accepted = []
                attempt = 0
                max_attempts = max(target_count * (10 if args.accept_ik_candidates else 30), 10 if args.accept_ik_candidates else 100)
                while len(accepted) < target_count and attempt < max_attempts:
                    for _ in range(min(args.batch_size, max_attempts - attempt)):
                        seed = args.seed + attempt + (0 if method == "baseline" else 50000000)
                        attempt += 1
                        sample = make_sample(method, object_name, seed)
                        if sample is not None:
                            accepted.append(sample)
                            if len(accepted) >= target_count:
                                break
                    if attempt % (args.batch_size * 4) == 0:
                        print(f"{method} {object_name}: accepted={len(accepted)}/{target_count} attempts={attempt}", flush=True)
                if len(accepted) < target_count:
                    raise RuntimeError(f"{method} {object_name}: only {len(accepted)}/{target_count} after {attempt} attempts")
                output = args.output_dir / f"{method}__{object_name.replace('+','__')}.pt"
                if output.exists() and not args.force:
                    raise FileExistsError(output)
                torch.save({"schema": "xhand_fullbody_grasp_pose_v1", "samples": accepted}, output)
                all_samples.extend(accepted)
                per_object.append({"method": method, "object_name": object_name, "accepted": len(accepted), "attempts": attempt, "output": str(output)})
                print(f"DONE {method} {object_name}: {len(accepted)}", flush=True)
        summary = {"schema": "xhand_fullbody_grasp_pose_summary_v1", "urdf": str(URDF), "methods": list(methods), "objects": selected_names, "target_per_object": target_count, "total": len(all_samples), "per_object": per_object, "isaaclab_physical_validation": "pending"}
    if args.scheduled and (args.schedule_start != 0 or args.schedule_end is not None):
        summary_name = f"summary__{args.schedule_start:04d}_{args.schedule_end:04d}.json"
    else:
        summary_name = "smoke_summary.json" if args.smoke else "summary.json"
    summary_path = args.output_dir / summary_name
    if summary_path.exists() and not args.force:
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
