#!/usr/bin/env python3
"""Transfer precomputed bimanual wrist seeds to a uniformly scaled object.

The output remains a normal one-entry source-vis payload.  It is intended as
an initialization pool only: every transferred pose must still pass the full
high-v1 geometry, Isaac, ablation, realized-geometry, and repeat gates.
"""

import argparse
import json
from pathlib import Path

import torch
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-vis", type=Path, required=True)
    parser.add_argument("--source-object", required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--root-scale-ratio", type=float, required=True)
    parser.add_argument("--source-mesh", type=Path)
    parser.add_argument("--preserve-root-surface-offset", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.audit_json.exists():
        raise FileExistsError("refusing to overwrite output or audit JSON")
    if args.root_scale_ratio <= 0:
        raise ValueError("--root-scale-ratio must be positive")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("--input must contain one source-vis entry")
    source_entry = payload[0]
    if source_entry.get("object_name") != args.source_object:
        raise ValueError(
            f"source object mismatch: {source_entry.get('object_name')}"
        )
    candidates = source_entry.get("precomputed_bimanual_candidates")
    if not isinstance(candidates, dict):
        raise RuntimeError("input has no precomputed_bimanual_candidates")
    left = candidates["left_q"].detach().cpu().clone()
    right = candidates["right_q"].detach().cpu().clone()
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 22:
        raise RuntimeError(f"invalid candidate tensors: {left.shape}, {right.shape}")
    if args.preserve_root_surface_offset:
        if args.source_mesh is None:
            raise ValueError(
                "--source-mesh is required with --preserve-root-surface-offset"
            )
        mesh = trimesh.load_mesh(args.source_mesh, process=False)
        roots = torch.cat((left[:, :3], right[:, :3])).numpy()
        closest, _, _ = trimesh.proximity.closest_point(mesh, roots)
        transferred = closest * float(args.root_scale_ratio) + (roots - closest)
        transferred = torch.as_tensor(transferred, dtype=left.dtype)
        left[:, :3] = transferred[: len(left)]
        right[:, :3] = transferred[len(left) :]
    else:
        left[:, :3] *= float(args.root_scale_ratio)
        right[:, :3] *= float(args.root_scale_ratio)

    entries = torch.load(
        args.source_vis, map_location="cpu", weights_only=False
    )
    matches = [
        entry for entry in entries if entry.get("object_name") == args.target_object
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one target source-vis entry, found {len(matches)}"
        )
    target_entry = dict(matches[0])
    source_metadata = candidates.get("metadata", [])
    metadata = []
    for index in range(len(left)):
        row = dict(source_metadata[index]) if index < len(source_metadata) else {}
        row.update(
            {
                "source_pair_index": index,
                "transfer_source_object": args.source_object,
                "transfer_target_object": args.target_object,
                "transfer_root_scale_ratio": float(args.root_scale_ratio),
                "transfer_source_candidate_index": index,
            }
        )
        metadata.append(row)
    target_entry["precomputed_bimanual_candidates"] = {
        "left_q": left,
        "right_q": right,
        "metadata": metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save([target_entry], args.output)
    audit = {
        "schema": "tro_grasp_precomputed_uniform_transfer_v1",
        "input": str(args.input.resolve()),
        "source_vis": str(args.source_vis.resolve()),
        "source_object": args.source_object,
        "target_object": args.target_object,
        "root_scale_ratio": args.root_scale_ratio,
        "preserve_root_surface_offset": args.preserve_root_surface_offset,
        "source_mesh": (
            str(args.source_mesh.resolve()) if args.source_mesh else None
        ),
        "num_candidates": len(left),
        "output": str(args.output.resolve()),
    }
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"transferred {len(left)} candidates -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
