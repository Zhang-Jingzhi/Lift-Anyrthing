"""Dataset adapter for Isaac-filtered two-Allegro grasp pairs."""

import math
import random
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
from torch.utils.data import DataLoader, Dataset

from utils.hand_model import create_hand_model


class BimanualPairDataset(Dataset):
    def __init__(
        self,
        batch_size,
        pair_file,
        num_points=512,
        object_pc_type="random",
    ):
        if object_pc_type != "random":
            raise ValueError(
                "BimanualPairDataset currently supports object_pc_type=random"
            )
        self.batch_size = batch_size
        self.num_points = num_points
        self.hand = create_hand_model("allegro", torch.device("cpu"))
        self.samples = torch.load(pair_file, map_location="cpu")
        if not self.samples:
            raise ValueError(f"No filtered pairs found in {pair_file}")

        repo = Path(__file__).resolve().parent.parent
        self.object_pcs = {}
        self.object_normals = {}
        for object_name in sorted(
            {sample["object_name"] for sample in self.samples}
        ):
            dataset, name = object_name.split("+")
            mesh_path = (
                repo
                / "data/data_urdf/object"
                / dataset
                / name
                / f"{name}.stl"
            )
            mesh = trimesh.load_mesh(mesh_path)
            points, face_indices = mesh.sample(65536, return_index=True)
            self.object_pcs[object_name] = torch.as_tensor(
                points,
                dtype=torch.float32,
            )
            self.object_normals[object_name] = torch.as_tensor(
                mesh.face_normals[face_indices],
                dtype=torch.float32,
            )

    @staticmethod
    def _matrix_to_pose(matrices):
        poses = torch.zeros((len(matrices), 6), dtype=torch.float32)
        for index, matrix in enumerate(matrices):
            poses[index, :3] = matrix[:3, 3]
            poses[index, 3:] = torch.from_numpy(
                Rotation.from_matrix(
                    matrix[:3, :3].numpy()
                ).as_rotvec()
            ).float()
        return poses

    @staticmethod
    def _merge_link_dicts(left, right):
        merged = {}
        for prefix, source in (("left", left), ("right", right)):
            for link_name, value in source.items():
                merged[f"{prefix}/{link_name}"] = value
        return merged

    def _hand_pair_features(self, left_q, right_q):
        left_pc, left_se3 = self.hand.get_transformed_links_pc(left_q)
        right_pc, right_se3 = self.hand.get_transformed_links_pc(right_q)
        return (
            self._merge_link_dicts(left_pc, right_pc),
            torch.cat([left_se3, right_se3], dim=0),
        )

    def __getitem__(self, index):
        result = {
            "robot_name": [],
            "object_name": [],
            "robot_pc_initial": [],
            "robot_pc_target": [],
            "robot_links_pc": [],
            "object_pc": [],
            "object_pc_normal": [],
            "initial_q": [],
            "target_q": [],
            "initial_se3": [],
            "target_se3": [],
            "initial_vec": [],
            "target_vec": [],
        }
        duplicated_links = self._merge_link_dicts(
            self.hand.links_pc,
            self.hand.links_pc,
        )

        for _ in range(self.batch_size):
            sample = random.choice(self.samples)
            object_name = sample["object_name"]
            left_target = sample["left_q"].clone()
            right_target = sample["right_q"].clone()
            left_initial = self.hand.get_initial_q(left_target)
            right_initial = self.hand.get_initial_q(right_target)

            target_pc, target_se3 = self._hand_pair_features(
                left_target,
                right_target,
            )
            initial_pc, initial_se3 = self._hand_pair_features(
                left_initial,
                right_initial,
            )

            point_indices = torch.randperm(65536)[: self.num_points]
            object_pc = self.object_pcs[object_name][point_indices].clone()
            object_normal = self.object_normals[object_name][
                point_indices
            ].clone()
            object_pc += torch.randn_like(object_pc) * 0.002

            result["robot_name"].append("allegro_bimanual")
            result["object_name"].append(object_name)
            result["robot_pc_initial"].append(initial_pc)
            result["robot_pc_target"].append(target_pc)
            result["robot_links_pc"].append(duplicated_links)
            result["object_pc"].append(object_pc)
            result["object_pc_normal"].append(object_normal)
            result["initial_q"].append(
                torch.cat([left_initial, right_initial])
            )
            result["target_q"].append(
                torch.cat([left_target, right_target])
            )
            result["initial_se3"].append(initial_se3)
            result["target_se3"].append(target_se3)
            result["initial_vec"].append(
                self._matrix_to_pose(initial_se3)
            )
            result["target_vec"].append(
                self._matrix_to_pose(target_se3)
            )

        result["object_pc"] = torch.stack(result["object_pc"])
        result["object_pc_normal"] = torch.stack(
            result["object_pc_normal"]
        )
        return result

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)


def _collate(batch):
    return batch[0]


def create_bimanual_dataloader(cfg):
    dataset = BimanualPairDataset(
        batch_size=cfg.batch_size,
        pair_file=cfg.pair_file,
        num_points=cfg.get("num_points", 512),
        object_pc_type=cfg.get("object_pc_type", "random"),
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_collate,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
