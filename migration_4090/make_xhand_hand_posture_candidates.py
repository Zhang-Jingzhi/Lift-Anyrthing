#!/usr/bin/env python3
"""Create XHand finger-posture candidates for the compact tilted grasp."""
import copy
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "migration_4090/results/xhand_compact_cracker_tilted_palm_v1.pt"
SOURCE_INDEX = int(os.environ.get("XHAND_POSTURE_SOURCE_INDEX", "10"))
OUTPUT_TAG = os.environ.get("XHAND_POSTURE_OUTPUT_TAG", "v1")
OUTPUT = ROOT / f"migration_4090/results/xhand_compact_cracker_hand_postures_{OUTPUT_TAG}.pt"
SUMMARY = ROOT / f"migration_4090/results/xhand_compact_cracker_hand_postures_{OUTPUT_TAG}.json"

# (proximal finger flexion, distal finger flexion)
FINGER_FLEX = ((0.65, 0.85), (0.90, 1.10), (1.15, 1.35))
# (thumb bend, thumb rotation joint 1, thumb rotation joint 2)
THUMB_POSES = (
    (0.80, -0.50, 1.00),
    (1.00, -0.30, 1.20),
    (0.80, 0.00, 1.00),
    (0.80, 0.30, 0.80),
)


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError(OUTPUT if OUTPUT.exists() else SUMMARY)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)["samples"][SOURCE_INDEX]
    samples, rows = [], []
    for proximal, distal in FINGER_FLEX:
        for thumb_bend, thumb_joint1, thumb_joint2 in THUMB_POSES:
            sample = copy.deepcopy(source)
            values = dict(zip(sample["joint_names"], sample["full_body_q"]))
            for side in ("left", "right"):
                values[f"{side}_hand_index_bend_joint"] = 0.10
                for finger in ("index", "mid", "ring", "pinky"):
                    values[f"{side}_hand_{finger}_joint1"] = proximal
                    values[f"{side}_hand_{finger}_joint2"] = distal
                values[f"{side}_hand_thumb_bend_joint"] = thumb_bend
                values[f"{side}_hand_thumb_rota_joint1"] = thumb_joint1
                values[f"{side}_hand_thumb_rota_joint2"] = thumb_joint2
            sample["full_body_q"] = [float(values[name]) for name in sample["joint_names"]]
            sample["xhand_hand_posture"] = {
                "finger_proximal_rad": proximal,
                "finger_distal_rad": distal,
                "thumb_bend_rad": thumb_bend,
                "thumb_joint1_rad": thumb_joint1,
                "thumb_joint2_rad": thumb_joint2,
            }
            samples.append(sample)
            rows.append({"index": len(samples) - 1, **sample["xhand_hand_posture"]})
    torch.save(
        {"schema": "xhand_compact_cracker_hand_postures_v1", "samples": samples},
        OUTPUT,
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "schema": "xhand_compact_cracker_hand_postures_v1",
                "source": str(SOURCE),
                "source_index": SOURCE_INDEX,
                "output": str(OUTPUT),
                "candidates": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(OUTPUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
