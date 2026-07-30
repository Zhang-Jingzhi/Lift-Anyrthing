import argparse
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from test import test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--split-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument(
        "--embodiment",
        choices=("allegro", "barrett", "shadowhand"),
        help=(
            "Override dataset/test embodiment. This is useful for running the "
            "same inference mode across multiple hands."
        ),
    )
    penetration_group = parser.add_mutually_exclusive_group()
    penetration_group.add_argument(
        "--penetration-exact",
        dest="penetration_mode",
        action="store_const",
        const="mesh",
        help="Use object-mesh containment (default; slower but more reliable).",
    )
    penetration_group.add_argument(
        "--penetration-approx",
        dest="penetration_mode",
        action="store_const",
        const="point_cloud",
        help="Use the faster nearest-point/normal approximation.",
    )
    parser.set_defaults(penetration_mode="mesh")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    config = OmegaConf.load(args.base_config)
    if args.embodiment is not None:
        config.dataset.robot_names = [args.embodiment]
        config.test.embodiment = args.embodiment
    config.dataset.debug_object_names = list(args.objects)
    config.dataset.batch_size = args.batch_size
    config.dataset.num_workers = 0
    config.dataset.object_pc_type = "fixed"
    config.test.ckpt = str(checkpoint)
    config.test.save_dir = str(Path(args.save_dir).resolve())
    config.test.split_batch_size = args.split_batch_size
    config.test.evaluate_penetration = True
    config.test.penetration_threshold = 0.005
    config.test.penetration_exact = args.penetration_mode == "mesh"
    config.test.penetration_method = args.penetration_mode
    test(config)


if __name__ == "__main__":
    main()
