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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--left-q-file", required=True)
    parser.add_argument("--right-q-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-gui", action="store_true")
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
    robot_file = metadata["urdf_path"][args.robot_name][21:]
    dataset, object_name = args.object_name.split("+")
    object_file = (
        f"{dataset}/{object_name}/coacd_decomposed_object_one_link.urdf"
    )

    hand = create_hand_model(args.robot_name, torch.device("cpu"))
    joint_orders = hand.get_joint_orders()
    if left_q.shape[-1] != len(joint_orders):
        left_q = q_rot6d_to_q_euler(left_q)
        right_q = q_rot6d_to_q_euler(right_q)

    validator = BimanualIsaacValidator(
        robot_name=args.robot_name,
        joint_orders=joint_orders,
        batch_size=left_q.shape[0],
        gpu=args.gpu,
        use_gui=args.use_gui,
    )
    try:
        validator.set_asset(
            robot_path=os.path.join(data_root, "robot"),
            robot_file=robot_file,
            object_path=os.path.join(data_root, "object"),
            object_file=object_file,
        )
        validator.create_envs()
        validator.set_actor_pose_dof(left_q, right_q)
        result = validator.run_sim()
        torch.save(result, args.output_file)
        print(
            f"success={int(result['success'].sum())}/"
            f"{len(result['success'])}"
        )
    finally:
        validator.destroy()


if __name__ == "__main__":
    main()
