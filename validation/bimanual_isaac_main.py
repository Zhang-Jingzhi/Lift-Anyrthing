"""Subprocess entry point for bimanual Isaac Gym validation."""

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.bimanual_isaac_validator import BimanualIsaacValidator
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch
import trimesh


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
    object_file = (
        f"{dataset}/{object_name}/coacd_decomposed_object_one_link.urdf"
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
    )
    try:
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
        validator.destroy()


if __name__ == "__main__":
    main()
