"""
Validation visualization results will be saved in the 'vis_info/' folder.
This code is used to visualize the saved information.
"""

import os
import sys
import time
import argparse
import viser
import trimesh
import torch
from utils.controller import controller
from utils.hand_model import create_hand_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


def main(file_name, host, port, initial_index):
    vis_info = torch.load(file_name, map_location='cpu')
    hands = {}
    status_markdown = None

    def on_update(idx):
        global_idx = idx
        invalid = True
        for info in vis_info:
            if idx >= info['predict_q'].shape[0]:
                idx -= info['predict_q'].shape[0]
            else:
                invalid = False
                break
        if invalid:
            print('Invalid index!')
            return

        print(info['robot_name'], info['object_name'], idx)
        print('result:', info['success'][idx])
        if status_markdown is not None:
            status_markdown.content = (
                f"**Global sample:** `{global_idx}` &nbsp; "
                f"**Object-local sample:** `{idx}` &nbsp; "
                f"**Robot:** `{info['robot_name']}` &nbsp; "
                f"**Object:** `{info['object_name']}` &nbsp; "
                f"**Isaac rollout:** "
                f"{'success' if bool(info['success'][idx]) else 'failure'}\n\n"
                "Scene tree 中 `model/input_initial` 只是模型/IK 的随机初值，"
                "不是物理仿真的起点；真正的仿真起点是 "
                "`rollout/initial_open`。建议一次只显示一个机械手节点。"
            )

        object_name = info['object_name'].split('+')
        object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{object_name[0]}/{object_name[1]}/{object_name[1]}.stl')
        object_trimesh = trimesh.load_mesh(object_path)
        server.scene.add_mesh_simple(
            'object',
            object_trimesh.vertices,
            object_trimesh.faces,
            color=(239, 132, 167),
            opacity=1.0
        )

        server.scene.add_point_cloud(
            'object_pc',
            info['object_pc'][idx].cpu().numpy(),
            point_size=0.0015,
            point_shape="circle",
            colors=(239, 132, 167),
            visible=False
        )

        if info['robot_name'] not in hands:
            hands[info['robot_name']] = create_hand_model(
                info['robot_name'],
                device=torch.device('cpu')
            )
        hand = hands[info['robot_name']]

        robot_transform_trimesh = hand.get_trimesh_se3(info['predict_transform'][0], idx)
        server.scene.add_mesh_trimesh(
            'model/diffusion_link_pose',
            robot_transform_trimesh,
            visible=False,
        )

        # The dataloader-provided initial_q is only the model/IK initialization.
        # It is not used as the initial state of the Isaac rollout.
        robot_trimesh = hand.get_trimesh_q(info['initial_q'][idx])['visual']
        server.scene.add_mesh_simple(
            'model/input_initial',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0,
            visible=False
        )

        # predict_q is the synthesized grasp returned by IK before the
        # hand-specific controller opens/closes the fingers for simulation.
        robot_trimesh = hand.get_trimesh_q(info['predict_q'][idx])['visual']
        server.scene.add_mesh_simple(
            'model/generated_grasp',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0,
            visible=False
        )

        rollout_initial_q, rollout_target_q = controller(
            info['robot_name'],
            info['predict_q'][idx],
        )
        robot_trimesh = hand.get_trimesh_q(rollout_initial_q)['visual']
        server.scene.add_mesh_simple(
            'rollout/initial_open',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0,
            visible=False
        )

        robot_trimesh = hand.get_trimesh_q(rollout_target_q)['visual']
        server.scene.add_mesh_simple(
            'rollout/target_closed',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0,
            visible=False
        )

        robot_trimesh = hand.get_trimesh_q(info['isaac_q'][idx])['visual']
        server.scene.add_mesh_simple(
            'rollout/final_after_disturbance',
            robot_trimesh.vertices,
            robot_trimesh.faces,
            color=(102, 192, 255),
            opacity=1.0
        )

    server = viser.ViserServer(host=host, port=port)
    status_markdown = server.gui.add_markdown(
        "选择一个 grasp 样本后，这里会显示各节点的含义。"
    )

    grasp_num = 0
    for info in vis_info:
        grasp_num += info['predict_q'].shape[0]
    if grasp_num == 0:
        raise ValueError(f"No grasps found in {file_name}")
    if not 0 <= initial_index < grasp_num:
        raise ValueError(
            f"Initial index must be in [0, {grasp_num - 1}], got {initial_index}"
        )

    slider = server.gui.add_slider(
        label='grasp_idx',
        min=0,
        max=grasp_num - 1,
        step=1,
        initial_value=initial_index
    )
    slider.on_update(lambda _: on_update(slider.value))
    on_update(initial_index)

    while True:
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to a generated vis.pt file')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8080, type=int)
    parser.add_argument('--index', default=0, type=int, help='Initial global grasp index')
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        parser.error(f"Visualization file does not exist: {args.file}")

    main(args.file, args.host, args.port, args.index)
