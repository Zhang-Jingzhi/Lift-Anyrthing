"""Browse strict bimanual samples with Viser."""

import argparse
import time
from pathlib import Path

import torch
import trimesh
import viser

from utils.hand_model import create_hand_model


def object_mesh(repo, object_name):
    dataset, name = object_name.split("+", 1)
    return trimesh.load_mesh(
        repo
        / "data/data_urdf/object"
        / dataset
        / name
        / f"{name}.stl",
        force="mesh",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/pilot_v4_realized_strict/"
            "visual_qc/selected_samples.pt"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    file_path = args.file if args.file.is_absolute() else repo / args.file
    records = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,
    )
    if not records:
        raise ValueError(f"No samples found in {file_path}")
    if not 0 <= args.index < len(records):
        raise ValueError(
            f"index must be in [0, {len(records) - 1}]"
        )

    hand = create_hand_model("allegro", torch.device("cpu"))
    server = viser.ViserServer(host=args.host, port=args.port)
    status = server.gui.add_markdown("")

    def update(index):
        record = records[index]
        sample = record["sample"]
        metrics = sample["metrics"]
        object_name = sample["object_name"]
        mesh = object_mesh(repo, object_name)
        left = hand.get_trimesh_q(sample["left_q"])["visual"]
        right = hand.get_trimesh_q(sample["right_q"])["visual"]
        server.scene.add_mesh_trimesh(
            "/object",
            mesh,
        )
        server.scene.add_mesh_simple(
            "/left_hand",
            left.vertices,
            left.faces,
            color=(46, 130, 210),
        )
        server.scene.add_mesh_simple(
            "/right_hand",
            right.vertices,
            right.faces,
            color=(51, 171, 97),
        )
        penetration = max(
            metrics["left_realized_penetration_mm"],
            metrics["right_realized_penetration_mm"],
        )
        status.content = (
            f"**Sample:** {index}/{len(records) - 1}  \n"
            f"**Category:** `{record['category']}`  \n"
            f"**Object:** `{object_name}`  \n"
            f"**Candidate:** `{sample['candidate_index']}`  \n"
            f"**Max disturbance:** "
            f"{metrics['max_direction_displacement_mm']:.3f} mm  \n"
            f"**Max penetration:** {penetration:.3f} mm  \n"
            f"**Inter-hand clearance:** "
            f"{metrics['realized_hand_clearance_mm']:.3f} mm"
        )

    slider = server.gui.add_slider(
        "sample_index",
        min=0,
        max=len(records) - 1,
        step=1,
        initial_value=args.index,
    )
    slider.on_update(lambda _: update(slider.value))
    update(args.index)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
