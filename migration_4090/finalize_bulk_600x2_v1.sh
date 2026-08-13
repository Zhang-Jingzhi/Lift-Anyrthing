#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
python=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3/envs/tro/bin/python
log="$repo/migration_4090/logs/bulk_finalize_600x2_v1.log"
cd "$repo"
exec > >(tee "$log") 2>&1

echo "Waiting for all per-GPU postprocessing queues"
while screen -ls 2>/dev/null | grep -Eq '[.]bulk_post_g[0-7][[:space:]]'; do
  sleep 30
done

baseline_root=graph_exp/bimanual_data/bulk_irregular_baseline_v1/final_100_each
bidex_root=graph_exp/bimanual_data/bulk_irregular_bidex_v1/final_100_each
baseline_out=graph_exp/bimanual_data/bulk_irregular_baseline_v1/final_600/bimanual_dataset.pt
bidex_out=graph_exp/bimanual_data/bulk_irregular_bidex_v1/final_600/bimanual_dataset.pt
test ! -e "$baseline_out"
test ! -e "$bidex_out"

"$python" migration_4090/assemble_bulk_method.py \
  --method-root "$baseline_root" --method baseline --output "$baseline_out"
"$python" migration_4090/assemble_bulk_method.py \
  --method-root "$bidex_root" --method bidex_style --output "$bidex_out"
"$python" migration_4090/summarize_bulk_600.py \
  --baseline "$baseline_out" \
  --bidex "$bidex_out" \
  --output-dir migration_4090/results/bulk_irregular_600x2_v1
echo "BULK 600x2 COMPLETE"
