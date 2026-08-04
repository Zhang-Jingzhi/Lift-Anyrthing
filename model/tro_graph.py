import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import *
from model.denoiser import GraphDenoiser
from utils.hand_model import create_hand_model

class RobotGraph(nn.Module):

    def __init__(
        self,
        vqvae_cfg,
        vqvae_pretrain,
        object_patch,
        max_link_node,
        robot_links,
        inference_config,
        bps_config,
        N_t_training,
        diffusion_config,
        denoiser_config,
        embodiment,
        loss_config,
        link_embedding_repeats=None,
        role_anchor_scale=0.0,
        mode="train",
    ):

        super(RobotGraph, self).__init__()
        # vqvae encoder
        self.vqvae = VQVAE(vqvae_cfg)
        if vqvae_pretrain is not None:
            state_dict = torch.load(vqvae_pretrain, map_location='cpu')
            self.vqvae.load_state_dict(state_dict)
            print(f"Loaded pretrained VQVAE from {vqvae_pretrain}.")
        # vqvae fixed
        for param in self.vqvae.parameters():
            param.requires_grad = False

        # meta
        self.embodiment = embodiment
        self.hand_dict = {}
        for hand_name in self.embodiment:
            self.hand_dict[hand_name] = create_hand_model(hand_name)
        self.link_embedding_repeats = link_embedding_repeats or {}
        self.role_anchor_scale = float(role_anchor_scale)

        self.object_patch = object_patch
        self.max_link_node = max_link_node
        
        # link embedding
        self.robot_links = robot_links
        self.link_embed_dim = bps_config.n_bps_points + 4   # link bps, centroid, scale
        self.bps = bps_torch(**bps_config)
        self.link_token_encoder = nn.Sequential(
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.link_embed_dim, self.link_embed_dim)
        )
        self.hand_role_embeddings = nn.Parameter(
            torch.zeros(2, self.link_embed_dim)
        )
        nn.init.normal_(self.hand_role_embeddings, std=0.02)

        self.N_t_training = N_t_training
        self.init_diffusion(diffusion_config)
        self.denoiser = GraphDenoiser(
            M=diffusion_config["M"],
            object_patch=self.object_patch,
            max_link_node=self.max_link_node,
            **OmegaConf.to_container(denoiser_config, resolve=True)
        )
        self.mode = mode
        self.link_embeddings = self.construct_bps()
        self.loss_config = loss_config
        if self.mode == "train":
            pass
        elif self.mode == "test":
            inference_config = OmegaConf.to_container(inference_config, resolve=True)
            self.inference_mode = inference_config["inference_mode"]
            self.bimanual_separation_projection = bool(
                inference_config.get(
                    "bimanual_separation_projection",
                    False,
                )
            )
            self.separation_projection_start_t = int(
                inference_config.get(
                    "separation_projection_start_t",
                    100,
                )
            )
            self.root_direction_prior = inference_config.get(
                "root_direction_prior",
                [0.024, -0.272, 0.889],
            )
            self.root_direction_priors = inference_config.get(
                "root_direction_priors",
                {},
            )
            assert self.inference_mode in ["unconditioned", "palm_conditioned"]
            if self.inference_mode == "palm_conditioned":
                self.palm_names = inference_config["palm_names"]
                self.interpolation_clip = inference_config["interpolation_clip"] * math.pi / 180
                self.interpolation_rate = inference_config["interpolation_rate"]

                self.rotation_error = inference_config["rotation_error"] * math.pi / 180
                self.t_star = self.get_start_timestamp()

    def construct_bps(self):
        
        link_embedding_dict = {}
        for embodiment, hand_model in self.hand_dict.items():
            links_pc = hand_model.links_pc
            embodiment_bps = []
            for link_name, link_pc in links_pc.items():
                centroid, scale = self._unit_ball_(link_pc)
                link_pc = (link_pc - centroid) / scale
                link_bps = self.bps.encode(
                    link_pc,
                    feature_type=['dists'],
                    x_features=None,
                    custom_basis=None
                )['dists']
                link_bps = torch.cat([link_bps, centroid, scale.view(1, 1)], dim=-1)
                embodiment_bps.append(link_bps)
            link_embedding_dict[embodiment] = torch.cat(
                embodiment_bps, dim=0
            )
        for alias, repeat_config in self.link_embedding_repeats.items():
            source = repeat_config["source"]
            repeats = int(repeat_config["repeats"])
            if source not in link_embedding_dict:
                raise KeyError(
                    f"Unknown link-embedding source {source!r} for {alias!r}"
                )
            link_embedding_dict[alias] = link_embedding_dict[source].repeat(
                repeats,
                1,
            )
        return link_embedding_dict

    def _encode_link_embeddings(self, robot_name):
        link_embed = self.link_token_encoder(
            self.link_embeddings[robot_name]
        )
        repeat_config = self.link_embedding_repeats.get(robot_name)
        if repeat_config and int(repeat_config.get("repeats", 1)) == 2:
            half = link_embed.shape[0] // 2
            if 2 * half != link_embed.shape[0]:
                raise ValueError(
                    f"Bimanual link count must be even, got "
                    f"{link_embed.shape[0]}"
                )
            role_embed = torch.cat(
                (
                    self.hand_role_embeddings[0].expand(half, -1),
                    self.hand_role_embeddings[1].expand(half, -1),
                ),
                dim=0,
            )
            if self.role_anchor_scale:
                role_anchor = torch.zeros_like(role_embed)
                role_anchor[:half, 0] = -self.role_anchor_scale
                role_anchor[half:, 0] = self.role_anchor_scale
                role_embed = role_embed + role_anchor
            link_embed = link_embed + role_embed
        return link_embed
            
    def _unit_ball_(self, pc):

        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def _project_bimanual_root_separation(
        self,
        translations,
        normalized_object_pc,
        robot_name,
        object_names=None,
        eps=1e-8,
    ):
        if (
            not getattr(
                self,
                "bimanual_separation_projection",
                False,
            )
            or robot_name != "allegro_bimanual"
        ):
            return translations
        half = int(self.loss_config.get("hand_links_per_actor", 0))
        if half <= 0 or 2 * half > translations.shape[1]:
            return translations
        left_root = translations[:, 0]
        right_root = translations[:, half]
        relative = right_root - left_root
        separation = relative.norm(dim=-1, keepdim=True)
        object_radius = normalized_object_pc.norm(
            dim=-1
        ).amax(dim=-1, keepdim=True)
        required = (
            float(
                self.loss_config.get(
                    "root_separation_radius_ratio",
                    1.5,
                )
            )
            * object_radius
        )
        if object_names is None:
            object_names = [None] * len(translations)
        priors = []
        for object_name in object_names:
            priors.append(
                self.root_direction_priors.get(
                    object_name,
                    self.root_direction_prior,
                )
            )
        prior = torch.as_tensor(
            priors,
            device=translations.device,
            dtype=translations.dtype,
        )
        prior = prior / prior.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
        reliable_direction = relative / separation.clamp_min(eps)
        direction = torch.where(
            (separation >= 0.5 * required).expand_as(relative),
            reliable_direction,
            prior,
        )
        correction = (
            0.5 * torch.relu(required - separation) * direction
        )
        projected = translations.clone()
        projected[:, :half] -= correction.unsqueeze(1)
        projected[:, half : 2 * half] += correction.unsqueeze(1)
        return projected

    def _normalize_pc_(self, pc):

        # recenter
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids

        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale

        return pc, centroids, scale

    def _swap_equivariance_loss(
        self,
        predicted_noise,
        object_nodes,
        noisy_robot_nodes,
        noisy_robot_object_edges,
        noisy_robot_robot_edges,
        timestamps,
        node_mask,
        batch,
        eps,
    ):
        weight = float(
            self.loss_config.get("swap_equivariance_weight", 0.0)
        )
        half = int(self.loss_config.get("hand_links_per_actor", 0))
        if (
            weight <= 0
            or half <= 0
            or any(
                name != "allegro_bimanual"
                for name in batch["robot_name"]
            )
        ):
            return predicted_noise.sum() * 0.0
        total = 2 * half
        if total > self.max_link_node:
            raise ValueError(
                f"Cannot swap {total} links with max_link_node="
                f"{self.max_link_node}"
            )
        permutation = torch.cat(
            (
                torch.arange(
                    half,
                    total,
                    device=predicted_noise.device,
                ),
                torch.arange(
                    0,
                    half,
                    device=predicted_noise.device,
                ),
                torch.arange(
                    total,
                    self.max_link_node,
                    device=predicted_noise.device,
                ),
            )
        )
        swapped_nodes = noisy_robot_nodes[:, permutation].clone()
        # Exchange hand geometry while keeping the destination role token.
        # This prevents the role label itself from being used as a shortcut.
        swapped_nodes[:, :total, 6:] = noisy_robot_nodes[:, :total, 6:]
        swapped_prediction = self.denoiser(
            object_nodes,
            swapped_nodes,
            noisy_robot_object_edges[:, permutation],
            noisy_robot_robot_edges[:, permutation][
                :, :, permutation
            ],
            timestamps,
        )
        swapped_back = swapped_prediction[:, permutation]
        error = (predicted_noise - swapped_back).pow(2).mean(dim=-1)
        return (error * node_mask).sum() / (node_mask.sum() + eps)

    def _geometry_losses(
        self,
        predicted_clean_pose,
        batch,
        object_pc,
        object_pc_normal,
        geometry_weights,
        eps,
    ):
        zero = predicted_clean_pose.sum() * 0.0
        half = int(self.loss_config.get("hand_links_per_actor", 0))
        if half <= 0:
            return zero, zero, zero, zero, zero, zero, zero
        contact_threshold = float(
            self.loss_config.get("contact_threshold_m", 0.005)
        )
        clearance = float(
            self.loss_config.get("inter_hand_clearance_m", 0.002)
        )
        penetration_losses = []
        contact_losses = []
        inter_hand_losses = []
        opposition_losses = []
        root_relative_losses = []
        predicted_root_separations = []
        target_root_separations = []
        matrices = vector_to_matrix(predicted_clean_pose)

        for expanded_index in range(len(predicted_clean_pose)):
            batch_index = expanded_index // self.N_t_training
            if batch["robot_name"][batch_index] != "allegro_bimanual":
                continue
            local_link_points = list(
                batch["robot_links_pc"][batch_index].values()
            )
            if len(local_link_points) < 2 * half:
                raise ValueError(
                    "Bimanual geometry loss expected at least "
                    f"{2 * half} link point clouds, got "
                    f"{len(local_link_points)}"
                )
            transformed = []
            for link_index in range(2 * half):
                points = local_link_points[link_index].to(
                    predicted_clean_pose.device,
                    dtype=predicted_clean_pose.dtype,
                )
                transform = matrices[expanded_index, link_index]
                transformed.append(
                    points @ transform[:3, :3].transpose(0, 1)
                    + transform[:3, 3]
                )
            left_points = torch.cat(transformed[:half], dim=0)
            right_points = torch.cat(transformed[half : 2 * half], dim=0)
            surface = object_pc[batch_index]
            normals = object_pc_normal[batch_index]

            hand_contact_losses = []
            hand_penetration_losses = []
            for hand_points in (left_points, right_points):
                distances = torch.cdist(
                    hand_points.unsqueeze(0),
                    surface.unsqueeze(0),
                )[0]
                nearest_distance, nearest_index = distances.min(dim=1)
                offsets = hand_points - surface[nearest_index]
                signed_distance = (
                    offsets * normals[nearest_index]
                ).sum(dim=-1)
                normalized_penetration = torch.relu(
                    -signed_distance / max(contact_threshold, eps)
                )
                hand_penetration_losses.append(
                    F.smooth_l1_loss(
                        normalized_penetration,
                        torch.zeros_like(normalized_penetration),
                    )
                )
                contact_violation = torch.relu(
                    nearest_distance.min()
                    / max(contact_threshold, eps)
                    - 1.0
                )
                hand_contact_losses.append(
                    F.smooth_l1_loss(
                        contact_violation,
                        torch.zeros_like(contact_violation),
                    )
                )
            geometry_weight = geometry_weights[expanded_index]
            penetration_losses.append(
                torch.stack(hand_penetration_losses).mean()
                * geometry_weight
            )
            contact_losses.append(
                torch.stack(hand_contact_losses).mean()
                * geometry_weight
            )
            minimum_hand_distance = torch.cdist(
                left_points.unsqueeze(0),
                right_points.unsqueeze(0),
            )[0].min()
            inter_hand_violation = torch.relu(
                1.0
                - minimum_hand_distance / max(clearance, eps)
            )
            inter_hand_losses.append(
                F.smooth_l1_loss(
                    inter_hand_violation,
                    torch.zeros_like(inter_hand_violation),
                )
                * geometry_weight
            )
            object_center = surface.mean(dim=0)
            left_direction = (
                matrices[expanded_index, 0, :3, 3] - object_center
            )
            right_direction = (
                matrices[expanded_index, half, :3, 3] - object_center
            )
            root_cosine = F.cosine_similarity(
                left_direction.unsqueeze(0),
                right_direction.unsqueeze(0),
                dim=-1,
                eps=eps,
            )[0]
            opposition_violation = torch.relu(root_cosine + 0.5)
            object_radius = (
                surface - object_center
            ).norm(dim=-1).amax().clamp_min(eps)
            predicted_root_relative = (
                matrices[expanded_index, half, :3, 3]
                - matrices[expanded_index, 0, :3, 3]
            )
            target_pose = batch["target_vec"][batch_index]
            target_root_relative = (
                target_pose[half, :3] - target_pose[0, :3]
            )
            normalized_relative_error = (
                predicted_root_relative - target_root_relative
            ) / object_radius
            root_relative_losses.append(
                F.smooth_l1_loss(
                    normalized_relative_error,
                    torch.zeros_like(normalized_relative_error),
                )
                * geometry_weight
            )
            predicted_root_separations.append(
                predicted_root_relative.detach().norm() * 1000.0
            )
            target_root_separations.append(
                target_root_relative.detach().norm() * 1000.0
            )
            required_separation = (
                float(
                    self.loss_config.get(
                        "root_separation_radius_ratio",
                        1.5,
                    )
                )
                * object_radius
            )
            root_separation = torch.norm(
                matrices[expanded_index, 0, :3, 3]
                - matrices[expanded_index, half, :3, 3]
            )
            separation_violation = torch.relu(
                1.0 - root_separation / required_separation
            )
            opposition_losses.append(
                0.5
                * (
                    F.smooth_l1_loss(
                        opposition_violation,
                        torch.zeros_like(opposition_violation),
                    )
                    + F.smooth_l1_loss(
                        separation_violation,
                        torch.zeros_like(separation_violation),
                    )
                )
                * geometry_weight
            )

        if not penetration_losses:
            return zero, zero, zero, zero, zero, zero, zero
        return (
            torch.stack(penetration_losses).mean(),
            torch.stack(contact_losses).mean(),
            torch.stack(inter_hand_losses).mean(),
            torch.stack(opposition_losses).mean(),
            torch.stack(root_relative_losses).mean(),
            torch.stack(predicted_root_separations).mean(),
            torch.stack(target_root_separations).mean(),
        )

    def init_diffusion(self, cfg):

        self.M = cfg["M"]
        self.scheduling = cfg["scheduling"]
        if self.scheduling == "linear":
            self.beta_min, self.beta_max = cfg["beta_min"], cfg["beta_max"]
            betas = torch.linspace(self.beta_min, self.beta_max, self.M)
        else:
            raise NotImplementedError()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer(
            "alpha_bars",
            torch.tensor([torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]),
        )
        self.ddim_steps = cfg["ddim_steps"]
        self.eta = cfg["ddim_eta"]
        self.noise_lambda = cfg["lambda"]

    def expand_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return (
            x[:, None, :, :]
            .expand(-1, self.N_t_training, -1, -1)
            .reshape(B * self.N_t_training, N, -1)
        )

    def _expand_and_reshape_(self, x, name):

        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:  # Node
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:  # Edge
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:  # Node Mask
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    def forward(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        ## Graph Construction

        # Object Node
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)
            
        object_nodes = torch.cat([
            object_tokens["xyz"],
            scale.expand(-1, self.object_patch, -1),
            object_tokens["z_q"]
        ], dim=-1)  # [B, P, 3+1+64]

        # Target Link Node
        target_vec = batch["target_vec"]
        link_target_poses = torch.zeros(
            [B, self.max_link_node, 6], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )
        
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            target_pose_vec = target_vec[b]
            target_trans = target_pose_vec[:, :3]
            target_rot = target_pose_vec[:, 3:]

            object_center, object_scale = centroids[b], scale[b]
            target_trans = (target_trans - object_center) / object_scale
            target_pose_vec = torch.cat([target_trans, target_rot], dim=-1)
            link_target_poses[b, :num_link, :] = target_pose_vec

            robot_name = batch["robot_name"][b]
            link_embed = self._encode_link_embeddings(robot_name)
            link_robot_embeds[b, :num_link, :] = link_embed
            link_node_masks[b, :num_link] = True

        robot_nodes = torch.cat([
            link_target_poses, 
            link_robot_embeds
        ], dim=-1)  # [B, L, 6+64]

        # Robot-Robot Edge, Fixed Pose
        link_rel_poses = torch.zeros(
            [B, self.max_link_node, self.max_link_node, 6],
            device=device, dtype=dtype
        )
        link_rel_masks = torch.zeros(
            [B, self.max_link_node, self.max_link_node],
            device=device, dtype=torch.bool
        )
        norm_robot_vec = robot_nodes[:, :, :6]
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T = vector_to_matrix(norm_robot_vec[b])[:num_link]
            rel_pose = compute_relative_se3(T, T)
            link_rel_poses[b, :num_link, :num_link, :] = matrix_to_vector(rel_pose)
            link_rel_masks[b, :num_link, :num_link] = True
            link_rel_masks[b, torch.arange(num_link), torch.arange(num_link)] = False

        # Robot-Object Edge
        link_object_rel_poses = torch.zeros(
            [B, self.max_link_node, self.object_patch, 6],
            device=device, dtype=dtype
        )
        link_object_masks = torch.zeros(
            [B, self.max_link_node, self.object_patch],
            device=device, dtype=torch.bool
        )
        # Only consider object translation
        object_positions = object_nodes[:, :, :3]
        object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, self.object_patch, -1, -1).clone()
        object_se3[:, :, :3, 3] = object_positions   # [B, P, 4, 4]
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T_r = vector_to_matrix(norm_robot_vec[b])[:num_link]   # [L, 4, 4]
            T_o = object_se3[b]                                    # [P, 4, 4]
            T_rel = compute_relative_se3(T_r, T_o)
            T_rel = matrix_to_vector(T_rel)
            link_object_rel_poses[b, :num_link, :self.object_patch] = T_rel
            link_object_masks[b, :num_link, :self.object_patch] = True

        ## Forward Diffusion
        t = np.random.randint(0, self.M, (B * self.N_t_training))
        high_noise_fraction = float(
            self.loss_config.get("high_noise_fraction", 0.0)
        )
        if high_noise_fraction > 0:
            high_noise_start = int(
                self.M
                * float(
                    self.loss_config.get(
                        "high_noise_start_ratio",
                        0.75,
                    )
                )
            )
            high_noise_mask = (
                np.random.random(B * self.N_t_training)
                < high_noise_fraction
            )
            t[high_noise_mask] = np.random.randint(
                high_noise_start,
                self.M,
                int(high_noise_mask.sum()),
            )
        V_O = self._expand_and_reshape_(object_nodes, "V_O")             # [B*T, P, 68]
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")              # [B*T, L, 70]
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]  

        eta_V_R_trans = torch.randn_like(V_R_trans)
        eta_V_R_rot = torch.randn_like(V_R_rot)
        a_bar = self.alpha_bars[t][:, None, None]
        
        noisy_trans = a_bar.sqrt() * V_R_trans + (1 - a_bar).sqrt() * eta_V_R_trans
        noisy_rot = a_bar.sqrt() * V_R_rot + (1 - a_bar).sqrt() * eta_V_R_rot
        noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # update graph edges
        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])           # [B*T, L, 4, 4] 
        noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
            compute_batch_relative_se3(                       
                noisy_V_R_se3, noisy_V_R_se3
            )
        )
        object_positions = V_O[:, :, :3]
        B, P, _ = object_positions.shape
        object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
        object_se3[:, :, :3, 3] = object_positions                                                   
        noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
        noisy_E_OR = matrix_to_vector(noisy_E_OR)

        ## Backward Denoising  
        pred_link_noise = self.denoiser(
            V_O,
            noisy_V_R,
            noisy_E_OR,
            noisy_E_RR,
            t
        )

        # noise loss
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_trans_noise = pred_link_noise[:, :, :3]
        pred_rot_noise = pred_link_noise[:, :, 3:]

        error_trans_noise = (eta_V_R_trans - pred_trans_noise) ** 2
        error_trans_noise = error_trans_noise.mean(dim=-1)
        loss_trans_noise = (error_trans_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot_noise = (eta_V_R_rot - pred_rot_noise) ** 2
        error_rot_noise = error_rot_noise.mean(dim=-1)
        loss_rot_noise = (error_rot_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        sqrt_alpha_bar = a_bar.sqrt().clamp_min(eps)
        sqrt_one_minus_alpha_bar = (1 - a_bar).sqrt()
        predicted_clean_trans = (
            noisy_trans
            - sqrt_one_minus_alpha_bar * pred_trans_noise
        ) / sqrt_alpha_bar
        predicted_clean_rot = (
            noisy_rot
            - sqrt_one_minus_alpha_bar * pred_rot_noise
        ) / sqrt_alpha_bar
        expanded_centroids = centroids.repeat_interleave(
            self.N_t_training,
            dim=0,
        )
        expanded_scale = scale.repeat_interleave(
            self.N_t_training,
            dim=0,
        )
        predicted_world_trans = (
            predicted_clean_trans * expanded_scale
            + expanded_centroids
        )
        predicted_clean_pose = torch.cat(
            (predicted_world_trans, predicted_clean_rot),
            dim=-1,
        )
        clean_trans_error = F.smooth_l1_loss(
            predicted_clean_trans,
            V_R_trans,
            reduction="none",
        ).mean(dim=-1)
        clean_rot_error = F.smooth_l1_loss(
            predicted_clean_rot,
            V_R_rot,
            reduction="none",
        ).mean(dim=-1)
        minimum_x0_weight = float(
            self.loss_config.get("minimum_x0_weight", 0.0)
        )
        x0_weights = a_bar[:, 0, 0].clamp_min(minimum_x0_weight)
        clean_pose_weights = x0_weights.unsqueeze(-1)
        loss_clean_pose = (
            ((clean_trans_error + clean_rot_error) * M_V_R)
            * clean_pose_weights
        ).sum() / (M_V_R.sum() + eps)
        (
            loss_penetration,
            loss_contact,
            loss_inter_hand,
            loss_opposition,
            loss_root_relative,
            predicted_root_separation_mm,
            target_root_separation_mm,
        ) = (
            self._geometry_losses(
                predicted_clean_pose,
                batch,
                object_pc,
                batch["object_pc_normal"],
                x0_weights,
                eps,
            )
        )
        loss_swap = self._swap_equivariance_loss(
            pred_link_noise,
            V_O,
            noisy_V_R,
            noisy_E_OR,
            noisy_E_RR,
            t,
            M_V_R,
            batch,
            eps,
        )

        total_loss = (
            self.loss_config["trans_weight"] * loss_trans_noise
            + self.loss_config["rot_weight"] * loss_rot_noise
            + float(self.loss_config.get("swap_equivariance_weight", 0.0))
            * loss_swap
            + float(self.loss_config.get("penetration_weight", 0.0))
            * loss_penetration
            + float(self.loss_config.get("contact_weight", 0.0))
            * loss_contact
            + float(self.loss_config.get("inter_hand_weight", 0.0))
            * loss_inter_hand
            + float(self.loss_config.get("opposition_weight", 0.0))
            * loss_opposition
            + float(self.loss_config.get("root_relative_weight", 0.0))
            * loss_root_relative
            + float(self.loss_config.get("clean_pose_weight", 0.0))
            * loss_clean_pose
        )
        loss_dict = {
            "loss_rot": loss_rot_noise,
            "loss_trans": loss_trans_noise,
            "loss_swap": loss_swap,
            "loss_penetration": loss_penetration,
            "loss_contact": loss_contact,
            "loss_inter_hand": loss_inter_hand,
            "loss_opposition": loss_opposition,
            "loss_root_relative": loss_root_relative,
            "loss_clean_pose": loss_clean_pose,
            "predicted_root_separation_mm": predicted_root_separation_mm,
            "target_root_separation_mm": target_root_separation_mm,
            "loss_total": total_loss
        }
        return loss_dict


    def get_start_timestamp(self, mu=1.596, eps=1e-8):
    
        target = torch.tensor(self.rotation_error / mu).clamp_min(eps) ** 2
        idx = torch.argmin(torch.abs((1.0 - self.alpha_bars) - target)).item()
        return int(idx)

    @torch.no_grad()
    def drop_half_halfspace(
        points: torch.Tensor,
        generator: torch.Generator | None = None,
        eps=1e-8
    ):
      
        B, N, _ = points.shape
        device = points.device
        k_remove = N // 2

        normals = torch.randn(B, 3, device=device, generator=generator)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + eps)

        proj = (points * normals[:, None, :]).sum(-1)
        idx_remove = proj.topk(k_remove, dim=1, largest=True, sorted=False).indices
        keep_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        keep_mask[torch.arange(B, device=device)[:, None], idx_remove] = False

        new_points = points[keep_mask].view(B, N - k_remove, 3)
        return new_points

    def inference(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        robot_names = batch["robot_name"]
        if isinstance(robot_names, str):
            robot_name = robot_names
        else:
            unique_robot_names = set(robot_names)
            if len(unique_robot_names) != 1:
                raise ValueError(
                    "Inference currently requires one embodiment per batch"
                )
            robot_name = next(iter(unique_robot_names))
        link_names = list(batch["robot_links_pc"][0].keys())
        valid_links = self.robot_links[robot_name]
        if len(link_names) != valid_links:
            raise ValueError(
                f"{robot_name} expects {valid_links} links, got "
                f"{len(link_names)}"
            )

        # Object Node
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)
            
        object_nodes = torch.cat([
            object_tokens["xyz"],
            scale.expand(-1, self.object_patch, -1),
            object_tokens["z_q"]
        ], dim=-1)  # [B, P, 3+1+64]

      
        if self.inference_mode == "unconditioned":
            object_names = batch.get("object_name", None)
            if isinstance(object_names, str):
                object_names = [object_names] * B

            ## Standard DDIM inference
            step = self.M // self.ddim_steps
            ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
            all_diffuse_step_poses_dict = {}

            ## Start from complete noise of link node (pose)
            noisy_V_R_trans = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            noisy_V_R_rot = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            link_robot_embeds = torch.zeros(
                [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
            )
            for b in range(B):
                num_link = self.robot_links[robot_name]
                link_embed = self._encode_link_embeddings(robot_name)
                link_robot_embeds[b, :num_link, :] = link_embed
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)

            ## Formulate edges
            noisy_V_R_pose = noisy_V_R[:, :, :6]
            noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
            noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                compute_batch_relative_se3(                       
                    noisy_V_R_se3, noisy_V_R_se3
                )
            )
            object_positions = object_nodes[:, :, :3]
            B, P, _ = object_positions.shape
            object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
            object_se3[:, :, :3, 3] = object_positions                                                   
            noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
            noisy_E_OR = matrix_to_vector(noisy_E_OR)

            for i, diffuse_step in enumerate(ddim_t):
                
                diffuse_step = diffuse_step.item()
                # predict noise
                pred_link_pose_noise = self.denoiser(
                    object_nodes,
                    noisy_V_R,
                    noisy_E_OR,
                    noisy_E_RR,
                    t=torch.full(
                        (object_nodes.shape[0],),
                        diffuse_step,
                        dtype=torch.long,
                        device=object_nodes.device
                    )
                )
                pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
                pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]
                
                # predict x_0
                a_bar_t = self.alpha_bars[diffuse_step]
                if i == len(ddim_t) - 1:
                    a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
                else:
                    a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

                x_t_trans = noisy_V_R_trans
                x_t_rot = noisy_V_R_rot
                x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise) / a_bar_t.sqrt()
                x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()

                sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
                ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

                if i == len(ddim_t) - 1:
                    z_trans = torch.zeros_like(x_0_trans)
                    z_rot = torch.zeros_like(x_0_rot)
                else:
                    z_trans = torch.randn_like(x_0_trans)
                    z_rot = torch.randn_like(x_0_rot)

                x_prev_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeffient * pred_link_trans_noise + sigma_t * z_trans * self.noise_lambda
                x_prev_rot = a_bar_prev.sqrt() * x_0_rot + ddim_coeffient * pred_link_rot_noise + sigma_t * z_rot * self.noise_lambda
                if diffuse_step <= self.separation_projection_start_t:
                    x_prev_trans = self._project_bimanual_root_separation(
                        x_prev_trans,
                        normal_pc,
                        robot_name,
                        object_names,
                        eps,
                    )

                noisy_V_R_trans = x_prev_trans
                noisy_V_R_rot = x_prev_rot

                # update node and edge
                noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)
                noisy_V_R_pose = noisy_V_R[:, :, :6]
                noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
                noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                    compute_batch_relative_se3(                       
                        noisy_V_R_se3, noisy_V_R_se3
                    )
                )
                object_positions = object_nodes[:, :, :3]
                B, P, _ = object_positions.shape
                object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
                object_se3[:, :, :3, 3] = object_positions                                                   
                noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
                noisy_E_OR = matrix_to_vector(noisy_E_OR)

                # save snapshot
                pred_trans = noisy_V_R_trans * scale + centroids
                pred_rot = noisy_V_R_rot
                pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

                predict_link_pose_dict = {}
                denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
                predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])       # [B, L, 4, 4]
                for link_id, link_name in enumerate(link_names):
                    predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
                all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict
        
        elif self.inference_mode == "palm_conditioned":
            palm_name = self.palm_names[robot_name]
            palm_index = link_names.index(palm_name)
            palm_r3 = batch["initial_se3"][:, palm_index][:, :3, :3]
            initial_pose = matrix_to_vector(batch["initial_se3"])

            step = self.M // self.ddim_steps
            ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
            start_idx = int(torch.argmin(torch.abs(ddim_t - self.t_star)).item())
            t_start = int(ddim_t[start_idx].item())
            a_bar_s = self.alpha_bars[t_start]
            
            # initial links
            link_robot_rots = torch.zeros(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            link_robot_embeds = torch.zeros(
                [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
            )
            for b in range(B):
                num_link = self.robot_links[robot_name]
                link_embed = self._encode_link_embeddings(robot_name)
                link_robot_embeds[b, :num_link, :] = link_embed
                link_robot_rots[b, :num_link] = initial_pose[b, :, 3:]

            noisy_V_R_rot = a_bar_s.sqrt() * link_robot_rots + (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
            noisy_V_R_trans = (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)

            ## Formulate edges
            noisy_V_R_pose = noisy_V_R[:, :, :6]
            noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
            noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                compute_batch_relative_se3(                       
                    noisy_V_R_se3, noisy_V_R_se3
                )
            )
            object_positions = object_nodes[:, :, :3]
            B, P, _ = object_positions.shape
            object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
            object_se3[:, :, :3, 3] = object_positions                                                   
            noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
            noisy_E_OR = matrix_to_vector(noisy_E_OR)

            all_diffuse_step_poses_dict = {}
            for i in range(start_idx, len(ddim_t)):
                
                diffuse_step = int(ddim_t[i].item())
                # predict noise
                pred_link_pose_noise = self.denoiser(
                    object_nodes,
                    noisy_V_R,
                    noisy_E_OR,
                    noisy_E_RR,
                    t=torch.full(
                        (object_nodes.shape[0],),
                        diffuse_step,
                        dtype=torch.long,
                        device=object_nodes.device
                    )
                )
                pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
                pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]
                
                # predict x_0
                a_bar_t = self.alpha_bars[diffuse_step]
                x_t_trans = noisy_V_R_trans
                x_t_rot = noisy_V_R_rot
                x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise) / a_bar_t.sqrt()
                x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()

                #########Add Palm Rotation Guidance#############
                progress = (i + 1) / self.ddim_steps
                s_t = self.interpolation_rate * math.sin(0.5 * progress * math.pi)                                                  # timestamp ratio
          
                with torch.enable_grad():
                    pred_link_rot_noise_with_grad = pred_link_rot_noise.detach().clone().requires_grad_(True)
                    x_0_rot_with_grad = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise_with_grad) / a_bar_t.sqrt()

                    B, L, _ = x_0_rot_with_grad.shape
                    R_cur_all = SO3.exp_map(x_0_rot_with_grad.reshape(-1, 3)).to_matrix().reshape(B, L, 3, 3)

                    r_cur = x_0_rot_with_grad[:, palm_index]
                    R_cur = SO3.exp_map(r_cur).to_matrix()                 
                    R_init = palm_r3.detach().expand(B, 3, 3)
    
                    ###### Interpolation ############
                    R_err = R_init @ torch.linalg.inv(R_cur)
                    r_err = SO3(tensor=R_err).log_map()

                    theta = r_err.norm(dim=-1, keepdim=True).clamp_min(eps)
                    step_angle = torch.clamp(theta, max=self.interpolation_clip)
                    # delta_rate = s_t * (step_angle / theta)
                    delta_rate = s_t
                    
                    r_delta = delta_rate * r_err
                    R_delta = SO3.exp_map(r_delta).to_matrix()[:, None, :, :]
                    R_intered = torch.matmul(R_delta, R_cur_all)
                    r_intered = SO3(tensor=R_intered.reshape(-1, 3, 3)).log_map().reshape(B, L, 3)
                    pred_link_rot_noise = (x_t_rot - a_bar_t.sqrt() * r_intered) / (1 - a_bar_t).sqrt()

                    palm_rot_loss = rotation_matrix_geodesic_loss(R_cur, palm_r3)
                    x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()
      
                    # print(f"DDIM {diffuse_step} | Palm Loss: {palm_rot_loss.item():.4f} | Delta Rate: {delta_rate}")
                ################################################

                if i == len(ddim_t) - 1:
                    a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
                else:
                    a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

                sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
                ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

                if i == len(ddim_t) - 1:
                    z_trans = torch.zeros_like(x_0_trans)
                    z_rot = torch.zeros_like(x_0_rot)
                else:
                    z_trans = torch.randn_like(x_0_trans)
                    z_rot = torch.randn_like(x_0_rot)

                x_prev_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeffient * pred_link_trans_noise + sigma_t * z_trans * self.noise_lambda
                x_prev_rot = a_bar_prev.sqrt() * x_0_rot + ddim_coeffient * pred_link_rot_noise + sigma_t * z_rot * self.noise_lambda

                noisy_V_R_trans = x_prev_trans
                noisy_V_R_rot = x_prev_rot

                # update node and edge
                noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)
                noisy_V_R_pose = noisy_V_R[:, :, :6]
                noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
                noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                    compute_batch_relative_se3(                       
                        noisy_V_R_se3, noisy_V_R_se3
                    )
                )
                object_positions = object_nodes[:, :, :3]
                B, P, _ = object_positions.shape
                object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
                object_se3[:, :, :3, 3] = object_positions                                                   
                noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
                noisy_E_OR = matrix_to_vector(noisy_E_OR)

                # save snapshot
                pred_trans = noisy_V_R_trans * scale + centroids
                pred_rot = noisy_V_R_rot
                pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

                predict_link_pose_dict = {}
                denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
                predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])       # [B, L, 4, 4]
                for link_id, link_name in enumerate(link_names):
                    predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
                all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict
        return all_diffuse_step_poses_dict
