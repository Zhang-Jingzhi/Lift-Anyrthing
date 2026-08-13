#!/usr/bin/env bash
set -euo pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
py=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
render_root="$repo/migration_4090/renders/expanded_multi_object_comparison_v2"
log_root="$repo/migration_4090/logs/large_random_6x100_vhacd_high_v1"
mkdir -p "$render_root" "$log_root"
cd "$repo"

"$py" - "$repo" "$render_root" <<'PY'
from pathlib import Path
import json
import subprocess
import sys
import torch

repo = Path(sys.argv[1])
out_root = Path(sys.argv[2])
graph_root = repo / "graph_exp" / "bimanual_data" / "large_random_6x100_v1"

objects = [
    ("ycb_bleach_cleanser", "Bleach cleanser"),
    ("ycb_cracker_box", "Cracker box"),
    ("ycb_pitcher_base", "Pitcher base"),
    ("contactdb_piggy_bank", "Piggy bank"),
    ("ycb_power_drill", "Power drill"),
    ("ycb_toy_airplane", "Toy airplane"),
]
methods = ("baseline", "bidex_v3")

def strict_count(path: Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return 0
    n = 0
    for sample in payload.get("samples", []):
        metrics = sample.get("metrics", {})
        if metrics.get("strict_success") is True and metrics.get("repeat_success_rate", 0.0) == 1.0:
            n += 1
    return n

def select_two(method: str, slug: str):
    base = graph_root / method / slug
    candidates = []
    for p in sorted(base.glob("size_*/seed_*/repeat_verified/verified_dataset.pt")):
        count = strict_count(p)
        if count:
            size = int(p.parents[2].name.split("_")[1])
            candidates.append((size, -count, str(p), count))
    if not candidates:
        raise RuntimeError(f"no strict verified dataset for {method}/{slug}")
    # Prefer two different formal size indices; use the most populated file per size.
    chosen = []
    for size in sorted({x[0] for x in candidates}):
        group = [x for x in candidates if x[0] == size]
        group.sort(key=lambda x: (x[1], x[2]))
        chosen.append(group[0])
        if len(chosen) == 2:
            break
    # If only one size exists, use a second seed from the same size.
    if len(chosen) < 2:
        for item in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            if item[2] != chosen[0][2]:
                chosen.append(item)
                break
    if len(chosen) < 2:
        chosen = chosen * 2
    return chosen[:2]

selection = {method: {} for method in methods}
for method in methods:
    for slug, label in objects:
        selected = select_two(method, slug)
        selection[method][slug] = []
        for variant, (size, neg_count, dataset, count) in enumerate(selected, start=1):
            dataset_path = Path(dataset)
            output = out_root / method / f"{slug}_variant_{variant:02d}_three_views.png"
            contacts = out_root / method / f"{slug}_variant_{variant:02d}_contacts.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.is_file() or output.stat().st_size == 0:
                cmd = [
                    "/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python",
                    str(repo / "scripts" / "render_verified_bimanual.py"),
                    "--dataset", str(dataset_path),
                    "--sample-index", "0",
                    "--output", str(output),
                    "--show-table", "--annotate-contacts",
                    "--contact-csv", str(contacts),
                ]
                subprocess.run(cmd, check=True)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"missing render: {output}")
            selection[method][slug].append({
                "variant": variant,
                "size_index": size,
                "strict_samples_in_source": count,
                "dataset": str(dataset_path.resolve()),
                "render": str(output.resolve()),
                "contacts": str(contacts.resolve()),
            })

(out_root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 6, figsize=(27, 16.5), squeeze=False)
for row, method in enumerate(("baseline", "bidex_v3")):
    for variant in (1, 2):
        r = row * 2 + (variant - 1)
        axes[r, 0].text(-0.22, 0.5,
                        ("Baseline" if method == "baseline" else "BiDex-v3") + f"\nvariant {variant}",
                        rotation=90, va="center", ha="center", fontsize=15,
                        fontweight="bold", transform=axes[r, 0].transAxes)
        for c, (slug, label) in enumerate(objects):
            info = selection[method][slug][variant - 1]
            axes[r, c].imshow(plt.imread(info["render"]))
            axes[r, c].set_axis_off()
            axes[r, c].set_title(label, fontsize=12, fontweight="bold")

fig.suptitle(
    "TRO-Grasp expanded PPT gallery — 24 strict verified bimanual examples\n"
    "six irregular object categories × two independent large-size variants per method",
    fontsize=21, fontweight="bold"
)
fig.subplots_adjust(left=0.045, right=0.995, bottom=0.02, top=0.88,
                    wspace=0.018, hspace=0.09)
out = out_root / "baseline_vs_bidex_v3_24_examples_comparison.png"
if out.exists():
    raise FileExistsError(out)
fig.savefig(out, dpi=180, facecolor="white")
plt.close(fig)

print(json.dumps({"output": str(out.resolve()), "selection": str((out_root / 'selection.json').resolve())}, indent=2))
PY

echo "expanded render complete"
