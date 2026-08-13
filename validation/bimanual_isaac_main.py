"""Subprocess entry point for bimanual Isaac Gym validation."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.bimanual_isaac_validator import BimanualIsaacValidator
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch
import trimesh


@contextmanager
def vhacd_asset_lock(args, object_file):
    """Serialize first-time V-HACD cache creation for the same mesh."""
    if not args.object_vhacd:
        yield
        return
    identity = "|".join(
        [
            object_file,
            str(args.object_vhacd_resolution),
            str(args.object_vhacd_max_convex_hulls),
            str(args.object_vhacd_max_vertices),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    lock_root = Path.home() / ".isaacgym" / "vhacd_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{digest}.lock"
    with lock_path.open("a+") as handle:
        print(f"Waiting for V-HACD asset lock: {lock_path}", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print(f"Acquired V-HACD asset lock: {lock_path}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            print(f"Released V-HACD asset lock: {lock_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-name",
        help="Legacy mode: use one robot asset for both actors.",
    )
    parser.add_argument("--left-robot-name")
    parser.add_argument("--right-robot-name")
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--left-q-file", required=True)
    parser.add_argument("--right-q-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--gravity", type=float, default=0.0)
    parser.add_argument("--gravity-settle-step", type=int, default=100)
    parser.add_argument("--staged-gravity", action="store_true")
    parser.add_argument("--independent-directions", action="store_true")
    parser.add_argument(
        "--active-hands",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--gravity-only", action="store_true")
    parser.add_argument("--support-during-closure", action="store_true")
    parser.add_argument("--fixture-during-closure", action="store_true")
    parser.add_argument("--lift-height", type=float, default=0.0)
    parser.add_argument("--lift-step", type=int, default=100)
    parser.add_argument("--min-lift-height", type=float, default=0.03)
    parser.add_argument("--robot-friction", type=float, default=3.0)
    parser.add_argument("--object-friction", type=float, default=3.0)
    parser.add_argument("--finger-effort-limit", type=float)
    parser.add_argument("--contact-offset", type=float, default=0.01)
    parser.add_argument("--max-gravity-displacement", type=float, default=0.02)
    parser.add_argument("--max-direction-displacement", type=float, default=0.02)
    parser.add_argument("--object-density", type=float, default=500.0)
    parser.add_argument("--object-vhacd", action="store_true")
    parser.add_argument("--object-vhacd-resolution", type=int, default=300000)
    parser.add_argument(
        "--object-vhacd-max-convex-hulls", type=int, default=64
    )
    parser.add_argument("--object-vhacd-max-vertices", type=int, default=64)
    parser.add_argument("--object-multicollision", action="store_true")
    parser.add_argument("--object-vhacd-high-v1", action="store_true")
    parser.add_argument("--object-vhacd-visual-high-v2", action="store_true")
    parser.add_argument("--capture-contacts", action="store_true")
    args = parser.parse_args()

    left_q = torch.load(args.left_q_file, map_location="cpu")
    right_q = torch.load(args.right_q_file, map_location="cpu")
    if left_q.shape != right_q.shape:
        raise ValueError(
            f"left/right q shape mismatch: {left_q.shape} vs {right_q.shape}"
        )

    data_root = os.path.join(ROOT_DIR, "data/data_urdf")
    metadata = json.load(
        open(os.path.join(data_root, "robot/urdf_assets_meta.json"))
    )
    left_robot_name = args.left_robot_name or args.robot_name
    right_robot_name = args.right_robot_name or args.robot_name
    if left_robot_name is None or right_robot_name is None:
        parser.error(
            "Specify --robot-name for legacy same-hand mode, or both "
            "--left-robot-name and --right-robot-name."
        )
    robot_asset_root = os.path.join(data_root, "robot")

    def robot_file(robot_name):
        path = os.path.normpath(metadata["urdf_path"][robot_name])
        prefix = os.path.normpath("data/data_urdf/robot") + os.sep
        if not path.startswith(prefix):
            raise ValueError(f"Robot URDF is outside the asset root: {path}")
        return path[len(prefix) :]

    left_robot_file = robot_file(left_robot_name)
    right_robot_file = robot_file(right_robot_name)
    dataset, object_name = args.object_name.split("+")
    if args.object_vhacd_high_v1 and not args.object_vhacd:
        parser.error("--object-vhacd-high-v1 requires --object-vhacd")
    if args.object_vhacd_visual_high_v2 and not args.object_vhacd:
        parser.error("--object-vhacd-visual-high-v2 requires --object-vhacd")
    if args.object_vhacd_high_v1 and args.object_vhacd_visual_high_v2:
        parser.error("Choose exactly one versioned V-HACD asset")
    if args.object_vhacd_high_v1 and args.object_multicollision:
        parser.error(
            "--object-vhacd-high-v1 and --object-multicollision are mutually exclusive"
        )
    if args.object_vhacd_visual_high_v2 and args.object_multicollision:
        parser.error(
            "--object-vhacd-visual-high-v2 and --object-multicollision are "
            "mutually exclusive"
        )
    if args.object_vhacd_high_v1 and (
        args.object_vhacd_resolution != 1_000_000
        or args.object_vhacd_max_convex_hulls != 128
        or args.object_vhacd_max_vertices != 64
    ):
        parser.error(
            "high-v1 requires resolution=1000000, max-convex-hulls=128, "
            "max-vertices=64"
        )
    if args.object_vhacd_visual_high_v2 and (
        args.object_vhacd_resolution != 1_000_000
        or args.object_vhacd_max_convex_hulls != 128
        or args.object_vhacd_max_vertices != 64
    ):
        parser.error(
            "visual-high-v2 requires resolution=1000000, "
            "max-convex-hulls=128, max-vertices=64"
        )
    if args.object_vhacd_visual_high_v2:
        object_urdf = "vhacd_visual_high_v2_object_one_link.urdf"
    elif args.object_vhacd_high_v1:
        object_urdf = "vhacd_high_v1_object_one_link.urdf"
    elif args.object_multicollision:
        object_urdf = "coacd_decomposed_object_multicollision_v1.urdf"
    else:
        object_urdf = "coacd_decomposed_object_one_link.urdf"
    object_file = f"{dataset}/{object_name}/{object_urdf}"
    object_absolute = Path(data_root) / "object" / object_file
    if not object_absolute.is_file():
        raise FileNotFoundError(
            f"Object URDF is missing: {object_absolute}. "
            "Build high-v1 assets before validation."
        )
    support_height = None
    if args.support_during_closure:
        object_mesh = trimesh.load_mesh(
            os.path.join(
                data_root,
                "object",
                dataset,
                object_name,
                f"{object_name}.stl",
            )
        )
        support_height = float(object_mesh.bounds[0, 2])

    left_hand = create_hand_model(left_robot_name, torch.device("cpu"))
    right_hand = create_hand_model(right_robot_name, torch.device("cpu"))
    left_joint_orders = left_hand.get_joint_orders()
    right_joint_orders = right_hand.get_joint_orders()
    if left_q.shape[-1] != len(left_joint_orders):
        left_q = q_rot6d_to_q_euler(left_q)
    if right_q.shape[-1] != len(right_joint_orders):
        right_q = q_rot6d_to_q_euler(right_q)

    validator = BimanualIsaacValidator(
        robot_name=left_robot_name,
        joint_orders=left_joint_orders,
        right_robot_name=right_robot_name,
        right_joint_orders=right_joint_orders,
        batch_size=left_q.shape[0],
        gpu=args.gpu,
        use_gui=args.use_gui,
        gravity=args.gravity,
        gravity_settle_step=args.gravity_settle_step,
        staged_gravity=args.staged_gravity,
        independent_directions=args.independent_directions,
        active_hands=args.active_hands,
        support_height=support_height,
        fixture_during_closure=args.fixture_during_closure,
        lift_height=args.lift_height,
        lift_step=args.lift_step,
        min_lift_height=args.min_lift_height,
        robot_friction=args.robot_friction,
        object_friction=args.object_friction,
        finger_effort_limit=args.finger_effort_limit,
        contact_offset=args.contact_offset,
        max_gravity_displacement=args.max_gravity_displacement,
        max_direction_displacement=args.max_direction_displacement,
        object_density=args.object_density,
        object_vhacd=args.object_vhacd,
        object_vhacd_resolution=args.object_vhacd_resolution,
        object_vhacd_max_convex_hulls=args.object_vhacd_max_convex_hulls,
        object_vhacd_max_vertices=args.object_vhacd_max_vertices,
        capture_contacts=args.capture_contacts,
    )
    try:
        with vhacd_asset_lock(args, str(object_absolute.resolve())):
            validator.set_asset(
                robot_path=robot_asset_root,
                robot_file=left_robot_file,
                right_robot_path=robot_asset_root,
                right_robot_file=right_robot_file,
                object_path=os.path.join(data_root, "object"),
                object_file=object_file,
            )
        validator.create_envs()
        validator.set_actor_pose_dof(left_q, right_q)
        result = validator.run_sim(gravity_only=args.gravity_only)
        torch.save(result, args.output_file)
        print(
            f"success={int(result['success'].sum())}/"
            f"{len(result['success'])}"
        )
    finally:
        print("Destroying Isaac validator", flush=True)
        validator.destroy()
        print("Isaac validator destroyed", flush=True)


if __name__ == "__main__":
    main()
    # Preview 4 can segfault while Python unloads its static CUDA/PhysX
    # objects after a fully successful right-hand-only run.  The simulation,
    # result save, and explicit validator destruction have all completed at
    # this point.  Bypass only the faulty interpreter teardown; exceptions in
    # main() still propagate with a non-zero status before reaching here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
