#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
log="$repo/migration_4090/logs/bidexgrasp_bleach_candidate_generation_joint_v4.log"

export CUDA_VISIBLE_DEVICES=0
cd "$repo"
exec > >(tee "$log") 2>&1
"$conda_root/bin/conda" run --no-capture-output -n tro \
  python migration_4090/generate_bidexgrasp_candidates.py \
  --source-vis migration_4090/derived/source_vis_bleach_sources0_4.pt \
  --object ycb+bleach_cleanser \
  --surface-points 4096 \
  --anchors 200 \
  --region-points 256 \
  --gws-contacts 5 \
  --region-pairs 2 \
  --maximum-source-seeds 5 \
  --hand-variants 5 \
  --minimum-pair-clearance-mm 10 \
  --minimum-pair-separation-fraction 0.65 \
  --minimum-vertical-separation-mm 0 \
  --maximum-vertical-separation-mm 10 \
  --contact-mm 2 \
  --penetration-mm 2 \
  --minimum-contact-links 3 \
  --support-clearance-mm 1 \
  --friction 1 \
  --seed 20260805 \
  --output migration_4090/derived/source_vis_bleach_bidexgrasp_candidates_joint_v4.pt \
  --audit-json migration_4090/results/bidexgrasp_bleach_candidate_audit_joint_v4.json
