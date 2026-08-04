#!/usr/bin/env python3
"""Fail fast when a minimal server reconstruction is incomplete."""

import json
import os
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "data/CMapDataset_filtered/cmap_dataset.pt",
    "data/data_urdf/robot/allegro/allegro_hand_left_extended.urdf",
    "data/data_urdf/robot/allegro_right/allegro_hand_right_extended.urdf",
    "data/PointCloud/robot/allegro_left.pt",
    "data/PointCloud/robot/allegro_right.pt",
    "data/data_urdf/object/contactdb/cylinder_xlarge/"
    "coacd_decomposed_object_one_link.urdf",
)


def main():
    missing = [path for path in REQUIRED_PATHS if not (REPO / path).exists()]
    metadata_path = REPO / "data/data_urdf/robot/urdf_assets_meta.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        for name in ("allegro_left", "allegro_right"):
            if name not in metadata.get("urdf_path", {}):
                missing.append(f"metadata urdf_path[{name!r}]")
    else:
        missing.append(str(metadata_path.relative_to(REPO)))

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}, cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"ISAAC_PYTHON={os.environ.get('ISAAC_PYTHON', '<not set>')}")
    dataset = REPO / "data/bimanual/bimanual_dataset.pt"
    print(f"bimanual_dataset={'present' if dataset.exists() else 'not downloaded'}")

    if missing:
        print("Missing required runtime assets:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("PyTorch cannot see a CUDA GPU.", file=sys.stderr)
        return 1
    print("Minimal TRO environment and runtime assets: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
