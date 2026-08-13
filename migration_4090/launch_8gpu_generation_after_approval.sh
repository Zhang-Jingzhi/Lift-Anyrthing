#!/usr/bin/env bash
# Prepared only; this file was NOT executed during the smoke test.
set -o errexit -o nounset -o pipefail

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
export ISAAC_PYTHON="$conda_root/envs/isaac/bin/python"
export PATH="$conda_root/envs/isaac/bin:$PATH"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Keep held-out test objects untouched. GPUs 0-5 split the six development
# objects. GPU 6 searches the validated xlarge protocol. GPU 7 extends the
# cylinder_large search with a different seed and a non-overlapping roll grid.
objects=(
  'ycb+bleach_cleanser'
  'ycb+wood_block'
  'contactdb+piggy_bank'
  'ycb+power_drill'
  'contactdb+cube_large'
  'contactdb+cylinder_large'
  'contactdb+cylinder_xlarge'
  'contactdb+cylinder_large'
)
seeds=(20260810 20260811 20260812 20260813 20260814 20260815 20260816 20260817)
roll_offsets=(0 0 0 0 0 0 0 11.25)

cd "$repo"
for gpu in $(seq 0 7); do
  object=${objects[$gpu]}
  seed=${seeds[$gpu]}
  safe_object=${object//+/_}
  output="$repo/graph_exp/bimanual_data/full_gpu${gpu}_${safe_object}_seed${seed}"
  log="$repo/migration_4090/logs/full_gpu${gpu}_${safe_object}_seed${seed}.log"
  session="tro_bimanual_g${gpu}"
  test ! -e "$output"
  test ! -e "$log"
  if screen -list | grep -q "[.]${session}"; then
    echo "screen session already exists: $session" >&2
    exit 1
  fi

  search=(
    --pairs-per-object 80
    --roll-count 8
    --roll-offset-degrees "${roll_offsets[$gpu]}"
  )
  if [[ $gpu -eq 6 ]]; then
    search=(
      --pairs-per-object 20
      --roll-count 16
      --symmetric-source
      --tabletop-left-roll-degrees 0
      --tabletop-root-height-mm 80
      --lateral-max-root-z-mm 100
      --lateral-max-height-diff-mm 40
    )
  fi

  screen -L -Logfile "$log" -dmS "$session" \
    env CUDA_VISIBLE_DEVICES="$gpu" \
    "$conda_root/bin/conda" run -n tro python generate_bimanual_pilot.py \
      --source-vis data/bimanual/source_vis.pt \
      --objects "$object" \
      "${search[@]}" \
      --isaac-batch-size 24 \
      --realized-batch-size 8 \
      --left-robot-name allegro_left \
      --right-robot-name allegro_right \
      --gravity 9.8 \
      --gravity-settle-step 500 \
      --support-during-closure \
      --no-fixture-during-closure \
      --opposition-mode tabletop \
      --independent-directions \
      --right-max-outward-mm 120 \
      --right-outward-step-mm 5 \
      --penetration-mm 2 \
      --contact-mm 2 \
      --min-contact-links 3 \
      --robot-friction 1 \
      --object-friction 1 \
      --finger-effort-limit 0.7 \
      --contact-offset 0.002 \
      --object-density 50 \
      --max-gravity-displacement 0.01 \
      --max-direction-displacement 0.015 \
      --seed "$seed" \
      --gpu 0 \
      --output-dir "$output"
  echo "started $session: GPU $gpu, $object, seed $seed, output $output"
done

screen -list
