#!/usr/bin/env python3
"""Render a compact, meeting-ready gallery of ObjectFlow meshes and XHand TCP IK targets."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "external_assets/objectflow_20260812/ObjectFlow_3D_sim_assets_20260812"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--episodes", nargs="+", default=["episode_000002", "episode_000003", "episode_000233", "episode_000234"])
    args = ap.parse_args()
    fig = plt.figure(figsize=(15, 11), dpi=160)
    for k, ep in enumerate(args.episodes):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        mesh = trimesh.load_mesh(ASSET_ROOT / ep / "model.obj", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh): mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        bank = torch.load(args.pose_dir / f"{ep}_ik.pt", map_location="cpu", weights_only=False)
        s = bank["samples"]
        # Show the first/middle/last trajectory frames, with faint meshes.
        ids = np.linspace(0, len(s)-1, min(3, len(s)), dtype=int)
        colors = ["#4C78A8", "#72B7B2", "#F58518"]
        allpts = []
        for ii, idx in enumerate(ids):
            sample = s[int(idx)]
            T = np.asarray(sample["object_pose_robot_preview"], dtype=float)
            m = mesh.copy(); m.apply_transform(T)
            v = np.asarray(m.vertices)
            # downsample for a responsive static figure
            if len(v) > 2500: v = v[np.linspace(0, len(v)-1, 2500, dtype=int)]
            ax.scatter(v[:,0], v[:,1], v[:,2], s=1.0, alpha=0.13, color=colors[ii])
            tcp = np.asarray(sample["target_tcp_positions_world"], dtype=float)
            ax.scatter(tcp[:,0], tcp[:,1], tcp[:,2], s=38, marker="o", color=["#D62728", "#2CA02C"], edgecolors="k", linewidths=.35)
            allpts.extend(v.tolist()); allpts.extend(tcp.tolist())
        p = np.asarray(allpts)
        lo, hi = p.min(0), p.max(0); ctr=(lo+hi)/2; span=max((hi-lo).max(), 0.12)*1.25
        ax.set_xlim(ctr[0]-span/2, ctr[0]+span/2); ax.set_ylim(ctr[1]-span/2, ctr[1]+span/2); ax.set_zlim(max(0, ctr[2]-span/2), ctr[2]+span/2)
        ax.set_title(f"{ep}  |  {len(s)} IK frames", fontsize=11)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=22, azim=-58)
        ax.text2D(0.02, 0.03, "red=L TCP, green=R TCP\nIK only; Isaac pending", transform=ax.transAxes, fontsize=8)
    fig.suptitle("ObjectFlow scanned objects — dual XHand kinematic grasp candidates", fontsize=16)
    fig.tight_layout(rect=[0,0,1,.96])
    args.output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.output, bbox_inches="tight"); print(args.output.resolve())

if __name__ == "__main__": main()
