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
        sampling_mode="object_balanced",
        swap_probability=0.0,
        objects=None,
        limit_per_object=None,
        object_pc_noise_std=0.002,
    ):
        if object_pc_type != "random":
            raise ValueError(
                "BimanualPairDataset currently supports object_pc_type=random"
            )
        self.batch_size = batch_size
        self.num_points = num_points
        self.object_pc_noise_std = float(object_pc_noise_std)
        if self.object_pc_noise_std < 0:
            raise ValueError("object_pc_noise_std must be non-negative")
        self.sampling_mode = sampling_mode
        self.swap_probability = float(swap_probability)
        if not 0.0 <= self.swap_probability <= 1.0:
            raise ValueError("swap_probability must be in [0, 1]")
        if sampling_mode not in {
            "object_balanced",
            "sample_uniform",
            "sample_cycle",
        }:
            raise ValueError(
                "sampling_mode must be object_balanced, sample_uniform, "
                "or sample_cycle"
            )
        self.hand = create_hand_model("allegro", torch.device("cpu"))
        loaded = torch.load(
            pair_file,
            map_location="cpu",
            weights_only=False,
        )
        self.samples = (
            loaded["samples"]
            if isinstance(loaded, dict) and "samples" in loaded
            else loaded
        )
        if objects:
            allowed_objects = set(objects)
            self.samples = [
                sample
                for sample in self.samples
                if sample["object_name"] in allowed_objects
            ]
        if not self.samples:
            raise ValueError(f"No filtered pairs found in {pair_file}")
        self.samples_by_object = {}
        for sample in self.samples:
            self.samples_by_object.setdefault(
                sample["object_name"],
                [],
            ).append(sample)
        if limit_per_object is not None:
            limit = int(limit_per_object)
            if limit <= 0:
                raise ValueError("limit_per_object must be positive")
            self.samples_by_object = {
                object_name: sorted(
                    samples,
                    key=lambda sample: int(sample["candidate_index"]),
                )[:limit]
                for object_name, samples in self.samples_by_object.items()
            }
            self.samples = [
                sample
                for object_name in sorted(self.samples_by_object)
                for sample in self.samples_by_object[object_name]
            ]
        self.object_names = sorted(self.samples_by_object)

        repo = Path(__file__).resolve().parent.parent
        self.object_pcs = {}
        self.object_normals = {}
        for object_name in self.object_names:
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

    def object_name_for_slot(self, index, slot):
        if self.sampling_mode == "object_balanced":
            global_slot = index * self.batch_size + slot
            return self.object_names[global_slot % len(self.object_names)]
        return None

    def sample_for_slot(self, index, slot):
        if self.sampling_mode == "sample_cycle":
            global_slot = index * self.batch_size + slot
            return self.samples[global_slot % len(self.samples)]
        object_name = self.object_name_for_slot(index, slot)
        if object_name is None:
            return random.choice(self.samples)
        return random.choice(self.samples_by_object[object_name])

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

        for slot in range(self.batch_size):
            sample = self.sample_for_slot(index, slot)
            object_name = sample["object_name"]
            left_target = sample["left_q"].clone()
            right_target = sample["right_q"].clone()
            swap_applied = random.random() < self.swap_probability
            if swap_applied:
                left_target, right_target = right_target, left_target
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
            if self.object_pc_noise_std:
                object_pc += (
                    torch.randn_like(object_pc)
                    * self.object_pc_noise_std
                )

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
        sampling_mode=cfg.get("sampling_mode", "object_balanced"),
        swap_probability=cfg.get("swap_probability", 0.0),
        objects=cfg.get("objects", None),
        limit_per_object=cfg.get("limit_per_object", None),
        object_pc_noise_std=cfg.get("object_pc_noise_std", 0.002),
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=_collate,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
