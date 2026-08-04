"""Report raw and object-balanced bimanual training distributions."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset.BimanualPairDataset import BimanualPairDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-file",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/pilot_v4_realized_strict/"
            "bimanual_dataset.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "graph_exp/bimanual_data/pilot_v4_realized_strict/"
            "distribution_audit"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    pair_file = (
        args.pair_file
        if args.pair_file.is_absolute()
        else repo / args.pair_file
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(
        pair_file,
        map_location="cpu",
        weights_only=False,
    )
    samples = payload["samples"] if isinstance(payload, dict) else payload
    counts = Counter(sample["object_name"] for sample in samples)
    total = len(samples)
    object_names = sorted(counts)
    balanced_probability = 1 / len(object_names)

    dataset = BimanualPairDataset(
        batch_size=1,
        pair_file=pair_file,
        num_points=32,
        sampling_mode="object_balanced",
    )
    scheduled = Counter(
        dataset.object_name_for_slot(index, 0)
        for index in range(len(dataset))
    )
    rows = [
        {
            "object_name": name,
            "raw_samples": counts[name],
            "raw_probability": counts[name] / total,
            "balanced_probability": balanced_probability,
            "scheduled_samples_per_epoch": scheduled[name],
        }
        for name in object_names
    ]
    with (output_dir / "distribution.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [name.split("+", 1)[1] for name in object_names]
    raw = [100 * row["raw_probability"] for row in rows]
    balanced = [100 * row["balanced_probability"] for row in rows]
    x = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [value - 0.2 for value in x],
        raw,
        width=0.4,
        label="Uniform over samples",
        color="#cf5b57",
    )
    axis.bar(
        [value + 0.2 for value in x],
        balanced,
        width=0.4,
        label="Object-balanced",
        color="#3978b9",
    )
    axis.set_xticks(x, labels, rotation=25)
    axis.set_ylabel("Expected sampling probability (%)")
    axis.set_title("Bimanual training distribution before and after balancing")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "distribution_balance.png",
        dpi=180,
    )
    plt.close(figure)

    minimum = min(scheduled.values())
    maximum = max(scheduled.values())
    lines = [
        "# Bimanual sampling distribution",
        "",
        f"- Strict samples: {total}.",
        f"- Objects: {len(object_names)}.",
        "- Sampling policy: choose objects in a deterministic balanced cycle,",
        "  then choose a random strict grasp from that object's bucket.",
        f"- Scheduled samples per epoch range: {minimum}–{maximum}.",
        f"- Maximum per-object epoch-count difference: {maximum - minimum}.",
        "",
        "| Object | Raw samples | Raw probability | Balanced probability |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['object_name']}` | {row['raw_samples']} | "
            f"{row['raw_probability']:.1%} | "
            f"{row['balanced_probability']:.1%} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(output_dir)
    print(f"epoch_balance_range={minimum}:{maximum}")


if __name__ == "__main__":
    main()
