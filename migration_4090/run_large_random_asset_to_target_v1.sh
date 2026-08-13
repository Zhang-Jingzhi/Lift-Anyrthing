#!/usr/bin/env bash
set -o errexit -o nounset -o pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU baseline|bidex_v3 OBJECT DENSITY TARGET BASE_SEED SIZE_ROOT" >&2
  exit 2
fi
gpu=$1
method=$2
object=$3
density=$4
target=$5
base_seed=$6
size_root=$7
finger_effort=${FORMAL_FINGER_EFFORT:-1.2}
repeat_oversample_factor=${FORMAL_REPEAT_OVERSAMPLE_FACTOR:-3}
repeat_selection_min=${FORMAL_REPEAT_SELECTION_MIN:-12}
repeat_selection_cap=${FORMAL_REPEAT_SELECTION_CAP:-48}
piggy_visual_high_v2_density_scale=${PIGGY_VISUAL_HIGH_V2_DENSITY_SCALE:-1.3013855074356893}
if [[ "$method" != baseline && "$method" != bidex_v3 ]]; then
  echo "invalid method=$method" >&2
  exit 2
fi

repo=/media/home/zhangjingzhi/TRO-Grasp-Reproduction
conda_root=/media/home/zhangjingzhi/.tro_grasp_tools/miniforge3
tro_python="$conda_root/envs/tro/bin/python"
isaac_python="$conda_root/envs/isaac/bin/python"
source_vis="$repo/migration_4090/derived/large_random_6x8_v1/source_vis_large_random_6x8_v1.pt"
bidex_one_shot_source=${BIDEX_ONE_SHOT_PRECOMPUTED_SOURCE:-}
bidex_one_shot_final_source=${BIDEX_ONE_SHOT_FINAL_SOURCE:-}
baseline_one_shot_source=${BASELINE_ONE_SHOT_PRECOMPUTED_SOURCE:-}
baseline_reuse_precomputed_source=${BASELINE_REUSE_PRECOMPUTED_SOURCE:-0}
bidex_reuse_final_source=${BIDEX_REUSE_FINAL_SOURCE:-0}
assembled="$size_root/assembled_target_$(printf '%02d' "$target")"
protocol_log_root="$repo/migration_4090/logs/large_random_6x100_vhacd_high_v1"
mkdir -p "$size_root" "$protocol_log_root"

simulation_density=$density
# Realized visual-mesh geometry is still audited sequentially inside an
# isolated subprocess.  Sixteen poses amortize the hand-model/point-cloud
# setup cost while staying below the entry point's default batch size of 24.
realized_batch_size=16
collision_args=(
  --object-vhacd --object-vhacd-high-v1
  --object-vhacd-resolution 1000000
  --object-vhacd-max-convex-hulls 128
  --object-vhacd-max-vertices 64
)
if [[ "$object" == contactdb+piggy_bank_* ]]; then
  simulation_density=$("$tro_python" -c \
    "print(float('$density') * float('$piggy_visual_high_v2_density_scale'))")
  collision_args=(
    --object-vhacd --object-vhacd-visual-high-v2
    --object-vhacd-resolution 1000000
    --object-vhacd-max-convex-hulls 128
    --object-vhacd-max-vertices 64
  )
fi
echo "collision_object=$object catalog_density=$density simulation_density=$simulation_density collision_args=${collision_args[*]}"

export PATH="$conda_root/envs/isaac/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu"
export ISAAC_PYTHON="$isaac_python"
export LD_LIBRARY_PATH="$conda_root/envs/isaac/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$repo"

count_verified() {
  "$tro_python" migration_4090/count_bulk_verified.py --input-root "$size_root" 2>/dev/null
}

if [[ -s "$assembled/bimanual_dataset.pt" ]]; then
  echo "already assembled: $assembled"
  exit 0
fi

attempt=0
disable_generation_dataset_reuse=false
while (( attempt < 100 )); do
  count=$(count_verified)
  echo "method=$method object=$object verified_unique=$count target=$target attempt=$attempt"
  if (( count >= target )); then
    "$tro_python" migration_4090/assemble_bulk_object.py \
      --input-root "$size_root" \
      --object-name "$object" \
      --method "$method" \
      --target "$target" \
      --output-dir "$assembled"
    exit 0
  fi

  seed=$((base_seed + attempt * 10000))
  seed_root="$size_root/seed_$seed"
  while [[ -e "$seed_root" ]]; do
    attempt=$((attempt + 1))
    seed=$((base_seed + attempt * 10000))
    seed_root="$size_root/seed_$seed"
  done
  selection="$seed_root/repeat_selection.pt"
  verified="$seed_root/repeat_verified"
  pre_repeat_quality="$seed_root/pre_repeat_local_quality.json"
  generation_dataset="$seed_root/bimanual_dataset.pt"
  log="$protocol_log_root/${method}_${object//+/_}_seed${seed}_gpu${gpu}.log"
  mkdir -p "$seed_root"

  reusable_generation_dataset=
  reusable_source=
  reusable_pointer_name=
  if [[ "$method" == baseline && -n "$baseline_one_shot_source" && "$baseline_reuse_precomputed_source" == 1 ]]; then
    reusable_source="$baseline_one_shot_source"
    reusable_pointer_name=reused_baseline_parent_source.txt
  elif [[ "$method" == bidex_v3 && -n "$bidex_one_shot_final_source" && "$bidex_reuse_final_source" == 1 ]]; then
    reusable_source="$bidex_one_shot_final_source"
    reusable_pointer_name=reused_bidex_final_source.txt
  fi
  if [[ -n "$reusable_source" && "$disable_generation_dataset_reuse" == false ]]; then
    for pointer in "$size_root"/seed_*/"$reusable_pointer_name"; do
      [[ -f "$pointer" ]] || continue
      [[ "$(<"$pointer")" == "$reusable_source" ]] || continue
      candidate_root=$(dirname "$pointer")
      [[ "$candidate_root" != "$seed_root" ]] || continue
      if [[ -s "$candidate_root/bimanual_dataset.pt" && -s "$candidate_root/sample_results.csv" && -s "$candidate_root/manifest.json" ]]; then
        reusable_sample_count=$("$tro_python" -c \
          "import torch; p=torch.load('$candidate_root/bimanual_dataset.pt', map_location='cpu', weights_only=False); print(len(p.get('samples', [])) if isinstance(p, dict) else 0)")
        if (( reusable_sample_count > 0 )); then
          reusable_generation_dataset="$candidate_root/bimanual_dataset.pt"
          break
        fi
      fi
    done
  fi

  run_generation() {
    local active_source="$source_vis"
    local pairs=8
    local rolls=16
    local baseline_targeted_retry=false
    if [[ "$method" == baseline && -n "$baseline_one_shot_source" ]]; then
      test -s "$baseline_one_shot_source"
      active_source="$baseline_one_shot_source"
      baseline_targeted_retry=true
      printf '%s\n' "$active_source" > "$seed_root/reused_baseline_parent_source.txt"
      if [[ "$baseline_reuse_precomputed_source" != 1 ]]; then
        baseline_one_shot_source=
      fi
    fi
    if [[ "$method" == baseline && "$object" == ycb+power_drill_* && "$baseline_targeted_retry" == false ]]; then
      # A mirrored source pair repeatedly lifted the enlarged drill while
      # losing both realized contacts.  Sample independent source grasps so
      # the two hands can occupy distinct handle/body regions instead.
      pairs=32
      rolls=16
    fi
    if [[ "$method" == bidex_v3 && -n "$bidex_one_shot_final_source" ]]; then
      test -s "$bidex_one_shot_final_source"
      active_source="$bidex_one_shot_final_source"
      printf '%s\n' "$active_source" > "$seed_root/reused_bidex_final_source.txt"
      if [[ "$bidex_reuse_final_source" != 1 ]]; then
        bidex_one_shot_final_source=
      fi
      pairs=17
      rolls=1
    elif [[ "$method" == bidex_v3 ]]; then
      local candidate="$seed_root/precomputed_bidex_v3.pt"
      local audit="$seed_root/precomputed_bidex_v3_audit.json"
      local high_vhacd_candidate="$seed_root/precomputed_bidex_v3_vhacd_high_neighborhood.pt"
      local targeted_contact_retry=false
      if [[ -n "$bidex_one_shot_source" ]]; then
        test -s "$bidex_one_shot_source"
        candidate="$bidex_one_shot_source"
        targeted_contact_retry=true
        printf '%s\n' "$candidate" > "$seed_root/reused_bidex_parent_source.txt"
        # Consume this source exactly once in the current queue process.
        bidex_one_shot_source=
      else
        local maximum_pair_clearance_mm=500
        local target_pair_clearance_mm=220
        local optimizer_minimum_contact_links=2
        local optimizer_minimum_contact_points=3
        if [[ "$object" == ycb+power_drill_* ]]; then
          # The enlarged drill is almost 0.60 m across but remains thin.  Its
          # feasible Allegro contact regions can exceed the generic 0.50 m
          # pair-clearance gate and often expose only one link during the
          # visual-mesh optimizer.  Keep the strict Isaac/realized gate
          # unchanged while allowing these candidates to reach it.
          maximum_pair_clearance_mm=700
          target_pair_clearance_mm=450
          optimizer_minimum_contact_links=1
          optimizer_minimum_contact_points=2
        fi
        if ! "$tro_python" migration_4090/generate_bidexgrasp_candidates.py \
          --source-vis "$source_vis" --object "$object" \
          --surface-points 2048 --anchors 128 --region-points 192 --gws-contacts 8 \
          --region-pairs 24 --region-scale 0.22 --enhanced-bidex-v3 \
          --v2-refinement-levels 2 --v2-joint-steps-degrees 4 \
          --v2-aperture-scales -0.75 -0.5 -0.25 0.5 1.0 1.5 2.0 2.5 \
          --v3-parent-candidate-cap 12 --v3-finger-perturbations 16 \
          --v3-finger-perturb-degrees 8 --v3-side-variants 6 \
          --v3-quality-keep-fraction 0.6 --v3-final-candidates 36 \
          --maximum-normal-opposition-cosine -0.80 \
          --minimum-radial-normal-alignment 0.30 \
          --minimum-height-fraction 0.22 --maximum-height-fraction 0.68 \
          --minimum-pair-separation-fraction 0.70 \
          --minimum-vertical-separation-mm 0 --maximum-vertical-separation-mm 40 \
          --minimum-standoff-mm 35 --maximum-standoff-mm 155 --standoff-step-mm 20 \
          --left-rolls 0 45 90 135 --right-rolls 0 45 90 135 \
          --maximum-source-seeds 2 --hand-variants 4 --preserve-standoff-diversity \
          --minimum-pair-clearance-mm 40 \
          --maximum-pair-clearance-mm "$maximum_pair_clearance_mm" \
          --target-pair-clearance-mm "$target_pair_clearance_mm" \
          --maximum-horizontal-root-cosine -0.80 \
          --maximum-root-z-mm 280 --maximum-root-height-difference-mm 55 \
          --contact-mm 2 --penetration-mm 2 \
          --minimum-contact-links "$optimizer_minimum_contact_links" \
          --minimum-contact-points "$optimizer_minimum_contact_points" \
          --minimum-contact-digits 1 \
          --target-contact-links 3 --target-contact-points 14 \
          --target-contact-digits 2 --target-contact-spread-mm 20 \
          --support-clearance-mm 1 --friction 1.2 \
          --stop-after-candidates 12 --seed "$seed" \
          --output "$candidate" --audit-json "$audit"; then
          return 1
        fi
      fi
      # BiDex optimizes against the visual mesh.  The high-resolution convex
      # collision decomposition shifts the feasible contact boundary by a few
      # millimetres, so expand the top ranked parents in a deterministic local
      # wrist/joint neighborhood before the strict physics gate.  This exact
      # strategy passed the high-v1 smoke test without relaxing any threshold.
      if [[ "$targeted_contact_retry" == true ]]; then
        if ! "$tro_python" migration_4090/expand_bidex_high_vhacd_neighborhood.py \
          --input "$candidate" --output "$high_vhacd_candidate" --top-k 3 \
          --left-radial-mm -6 -4 -2 0 \
          --right-radial-mm -1.5 -1 -0.5 0 \
          --joint-delta-pair-rad 0.04 0.04 \
          --joint-delta-pair-rad 0.04 0.08 \
          --joint-delta-pair-rad 0.04 0.12 \
          --joint-delta-pair-rad 0.04 0.16 \
          --joint-delta-pair-rad 0.04 0.20; then
          return 1
        fi
      else
        if ! "$tro_python" migration_4090/expand_bidex_high_vhacd_neighborhood.py \
          --input "$candidate" --output "$high_vhacd_candidate" --top-k 3 \
          --radial-mm -6 -4 -2 0 \
          --joint-delta-pair-rad -0.04 -0.04 \
          --joint-delta-pair-rad 0 0 \
          --joint-delta-pair-rad 0.04 0.04 \
          --joint-delta-pair-rad 0 0.08 \
          --joint-delta-pair-rad 0.04 0.08 \
          --joint-delta-pair-rad 0.04 0.12; then
          return 1
        fi
      fi
      active_source="$high_vhacd_candidate"
      pairs=240
      if [[ "$targeted_contact_retry" == false ]]; then
        pairs=288
      fi
      rolls=1
    fi

    args=(
      --source-vis "$active_source" --objects "$object"
      --pairs-per-object "$pairs" --roll-count "$rolls"
      --isaac-batch-size 16 --realized-batch-size "$realized_batch_size"
      --left-robot-name allegro_left --right-robot-name allegro_right
      --gravity 9.8 --gravity-settle-step 500
      --lift-height 0.10 --lift-step 150 --min-lift-height 0.02
      --support-during-closure --no-fixture-during-closure
      --opposition-mode tabletop --tabletop-left-roll-degrees 0
      --tabletop-root-height-mm 60 --max-root-height-fraction 0.80
      --lateral-max-root-z-mm 280 --lateral-max-height-diff-mm 55
      --independent-directions --penetration-mm 2 --contact-mm 2
      --min-contact-links 1 --robot-friction 1.2 --object-friction 1.2
      --finger-effort-limit "$finger_effort" --contact-offset 0.001
      "${collision_args[@]}"
      --object-density "$simulation_density" --max-gravity-displacement 0.0125
      --max-direction-displacement 0.015 --seed "$seed" --gpu 0
      --output-dir "$seed_root" --force
    )
    if [[ "$method" == baseline ]]; then
      if [[ "$baseline_targeted_retry" == false ]]; then
        if [[ "$object" != ycb+power_drill_* ]]; then
          args+=(--symmetric-source)
        fi
        args+=(--right-max-outward-mm 300 --right-outward-step-mm 30
          --radial-fine-step-mm 3 --radial-min-offset-mm -24)
      fi
      if [[ "$object" == ycb+toy_airplane_* && "$baseline_targeted_retry" == false ]]; then
        args+=(--joint-seed-offset-rad 0.04)
      fi
    fi
    if [[ "$object" == contactdb+piggy_bank_* ]]; then
      args+=(--realized-lateral-max-height-fraction 0.35)
    fi
    "$tro_python" generate_bimanual_pilot.py "${args[@]}"
  }

  generation_ok=false
  if [[ -n "$reusable_generation_dataset" ]]; then
    generation_dataset="$reusable_generation_dataset"
    printf '%s\n' "$reusable_generation_dataset" > "$seed_root/reused_generation_dataset.txt"
    echo "reusing deterministic strict-generation dataset: $reusable_generation_dataset" | tee "$log"
    generation_ok=true
  elif run_generation > >(tee "$log") 2>&1; then
    generation_ok=true
  fi
  if [[ "$generation_ok" == true ]]; then
    quality_selection_args=()
    if [[ "$method" == bidex_v3 ]]; then
      if [[ -n "$reusable_generation_dataset" ]]; then
        reusable_quality="$(dirname "$reusable_generation_dataset")/pre_repeat_local_quality.json"
        if [[ -s "$reusable_quality" && ! -e "$pre_repeat_quality" ]]; then
          cp "$reusable_quality" "$pre_repeat_quality"
          echo "reusing deterministic local-quality audit: $reusable_quality" >> "$log"
        fi
      fi
      if [[ ! -s "$pre_repeat_quality" ]]; then
        if ! "$tro_python" scripts/audit_decoupled_force_closure.py \
          --dataset "$generation_dataset" \
          --output "$pre_repeat_quality" --friction 1 \
          --contact-mm 2 --points-per-link 8 --min-contact-links 1 \
          --max-wrench-residual 0.35 --residual-statistic mean \
          >> "$log" 2>&1; then
          echo "pre-repeat local quality audit failed: $seed_root" >> "$log"
          attempt=$((attempt + 1))
          continue
        fi
      fi
      quality_selection_args=(--local-quality-json "$pre_repeat_quality")
    fi
    current_count=$(count_verified)
    remaining=$((target - current_count))
    if (( remaining < 1 )); then
      remaining=1
    fi
    # A strict first-pass pose still has to survive two additional Isaac
    # rollouts.  In the formal campaign only about 20--35% of selected poses
    # typically remain valid for all 3/3 rollouts.  Selecting exactly
    # `remaining` therefore causes several small serial repeat jobs over the
    # same deterministic first-pass dataset.  Validate a bounded oversample
    # in one batch instead; assemble_bulk_object.py still writes exactly the
    # scheduled target and every selected pose passes the unchanged gates.
    repeat_selection_count=$((remaining * repeat_oversample_factor))
    if (( repeat_selection_count < repeat_selection_min )); then
      repeat_selection_count=$repeat_selection_min
    fi
    if (( repeat_selection_count > repeat_selection_cap )); then
      repeat_selection_count=$repeat_selection_cap
    fi
    if (( repeat_selection_count < remaining )); then
      repeat_selection_count=$remaining
    fi
    echo "repeat_selection remaining=$remaining selected_cap=$repeat_selection_count oversample_factor=$repeat_oversample_factor minimum=$repeat_selection_min absolute_cap=$repeat_selection_cap" >> "$log"
    "$tro_python" migration_4090/select_bulk_repeat_candidates.py \
      --source-dataset "$generation_dataset" \
      --output "$selection" --max-samples "$repeat_selection_count" \
      --exclude-root "$size_root" \
      "${quality_selection_args[@]}" >> "$log" 2>&1
    selected_count=$("$tro_python" -c \
      "import torch; print(len(torch.load('$selection', map_location='cpu', weights_only=False)['samples']))")
    if (( selected_count > 0 )); then
      "$tro_python" scripts/repeat_verify_xlarge_batch.py \
        --repo "$repo" --source-dataset "$selection" --output-dir "$verified" \
        --isaac-python "$isaac_python" --additional-repeats 2 \
        --batch-size 16 --geometry-batch-size 4 --gpu 0 \
        --object-name "$object" --object-density "$simulation_density" \
        --finger-effort-limit "$finger_effort" --lift-height 0.10 --lift-step 150 \
        --min-lift-height 0.02 \
        --robot-friction 1.2 --object-friction 1.2 --contact-offset 0.001 \
        "${collision_args[@]}" \
        --max-gravity-displacement 0.0125 --max-direction-displacement 0.015 \
        --min-contact-links 1 --geometry-contact-mm 2 >> "$log" 2>&1 || true
    elif [[ -n "$reusable_generation_dataset" ]]; then
      # Every pose in the deterministic first-pass dataset has either been
      # verified or has consumed its allowed stochastic repeat attempts.
      # Re-run first-pass physics on the same adaptive source with the next
      # seed instead of spinning through empty selections.
      disable_generation_dataset_reuse=true
      echo "deterministic generation dataset exhausted; regenerating with new seeds" >> "$log"
    fi
  fi
  attempt=$((attempt + 1))
done

echo "target not reached: method=$method object=$object target=$target" >&2
exit 1
