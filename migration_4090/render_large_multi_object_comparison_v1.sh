#!/usr/bin/env bash
set -euo pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
py=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
render_root="$repo/migration_4090/renders/large_random_multi_object_comparison_v1"
log_root="$repo/migration_4090/logs/large_random_6x100_vhacd_high_v1"
mkdir -p "$render_root/baseline" "$render_root/bidex_v3" "$log_root"
cd "$repo"

declare -a rows=(
  "baseline|bleach_cleanser|ycb+bleach_cleanser|graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_bleach_cleanser/size_000/seed_222869057/repeat_verified/verified_dataset.pt"
  "baseline|cracker_box|ycb+cracker_box|graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_cracker_box/size_000/seed_422880063/repeat_verified/verified_dataset.pt"
  "baseline|pitcher_base|ycb+pitcher_base|graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_pitcher_base/size_000/seed_20263000/repeat_verified/verified_dataset.pt"
  "baseline|piggy_bank|contactdb+piggy_bank|graph_exp/bimanual_data/large_random_6x100_v1/baseline/contactdb_piggy_bank/size_000/seed_622872052/repeat_verified/verified_dataset.pt"
  "baseline|power_drill|ycb+power_drill|graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_power_drill/size_000/seed_322913050/repeat_verified/verified_dataset.pt"
  "baseline|toy_airplane|ycb+toy_airplane|graph_exp/bimanual_data/large_random_6x100_v1/baseline/ycb_toy_airplane/size_000/seed_323024055/repeat_verified/verified_dataset.pt"
  "bidex_v3|bleach_cleanser|ycb+bleach_cleanser|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/ycb_bleach_cleanser/size_000/seed_222889057/repeat_verified/verified_dataset.pt"
  "bidex_v3|cracker_box|ycb+cracker_box|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/ycb_cracker_box/size_000/seed_422870063/repeat_verified/verified_dataset.pt"
  "bidex_v3|pitcher_base|ycb+pitcher_base|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/ycb_pitcher_base/size_000/seed_222881050/repeat_verified/verified_dataset.pt"
  "bidex_v3|piggy_bank|contactdb+piggy_bank|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/contactdb_piggy_bank/size_000/seed_622872052/repeat_verified/verified_dataset.pt"
  "bidex_v3|power_drill|ycb+power_drill|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/ycb_power_drill/size_000/seed_322873050/repeat_verified/verified_dataset.pt"
  "bidex_v3|toy_airplane|ycb+toy_airplane|graph_exp/bimanual_data/large_random_6x100_v1/bidex_v3/ycb_toy_airplane/size_000/seed_422914055/repeat_verified/verified_dataset.pt"
)

for row in "${rows[@]}"; do
  IFS='|' read -r method slug object dataset <<<"$row"
  output="$render_root/$method/${slug}_three_views.png"
  if [[ -s "$output" ]]; then
    echo "keep existing $output"
    continue
  fi
  test -s "$dataset"
  "$py" scripts/render_verified_bimanual.py \
    --dataset "$dataset" --sample-index 0 --output "$output" \
    --show-table --annotate-contacts
  test -s "$output"
done

"$py" - "$render_root" <<'PY'
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__import__("sys").argv[1])
objects = [
    ("bleach_cleanser", "Bleach"),
    ("cracker_box", "Cracker"),
    ("pitcher_base", "Pitcher"),
    ("piggy_bank", "Piggy bank"),
    ("power_drill", "Power drill"),
    ("toy_airplane", "Toy airplane"),
]
paths = {}
for method in ("baseline", "bidex_v3"):
    paths[method] = [root / method / f"{slug}_three_views.png" for slug, _ in objects]
    for p in paths[method]:
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(p)

out = root / "baseline_vs_bidex_six_objects_comparison.png"
if out.exists():
    raise FileExistsError(out)
fig, axes = plt.subplots(2, 6, figsize=(25, 8.8), squeeze=False)
for r, method in enumerate(("baseline", "bidex_v3")):
    axes[r, 0].text(-0.22, 0.5, "Baseline" if method == "baseline" else "BiDex-v3",
                    rotation=90, va="center", ha="center", fontsize=18,
                    fontweight="bold", transform=axes[r, 0].transAxes)
    for c, (slug, label) in enumerate(objects):
        axes[r, c].imshow(plt.imread(paths[method][c]))
        axes[r, c].set_axis_off()
        axes[r, c].set_title(label, fontsize=12, fontweight="bold")
fig.suptitle(
    "TRO-Grasp large-random formal assets — six irregular objects, strict 3/3 verified examples",
    fontsize=19, fontweight="bold"
)
fig.subplots_adjust(left=0.035, right=0.995, bottom=0.02, top=0.87, wspace=0.015, hspace=0.08)
fig.savefig(out, dpi=180, facecolor="white")
plt.close(fig)

for method, label in (("baseline", "baseline"), ("bidex_v3", "bidex_v3")):
    out_method = root / f"{label}_six_objects_comparison.png"
    if out_method.exists():
        raise FileExistsError(out_method)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for ax, (slug, title) in zip(axes.flat, objects):
        ax.imshow(plt.imread(root / method / f"{slug}_three_views.png"))
        ax.set_axis_off(); ax.set_title(title, fontsize=13, fontweight="bold")
    fig.suptitle(f"{label.upper()} — six large irregular objects", fontsize=19, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9, wspace=0.02, hspace=0.12)
    fig.savefig(out_method, dpi=180, facecolor="white")
    plt.close(fig)

print(json.dumps({"combined": str(out.resolve()), "baseline": str((root/'baseline_six_objects_comparison.png').resolve()), "bidex_v3": str((root/'bidex_v3_six_objects_comparison.png').resolve())}, indent=2))
PY

echo "render-and-compose complete"
