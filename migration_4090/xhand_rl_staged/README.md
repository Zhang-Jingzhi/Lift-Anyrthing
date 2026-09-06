# BODex bimanual staged RL

This namespace is intentionally separate from `xhand_rl_embedded` and
`xhand_rl_curriculum`. Existing v2 checkpoints and accepted data are not
rewritten.

The policy ABI is fixed for every stage:

- action: 38 joint residuals; inactive joints are masked, never removed;
- observation: existing 145D + integrated residual 38D + BODex target error
  38D + two palm poses in object coordinates 18D + grouped contact
  force/flags 24D + stage one-hot 6D + 4 gate-memory values (bilateral streak,
  two historical group-count progresses, stable-hold streak) + 16
  candidate-gated gate-memory interactions + four-way BODex candidate one-hot
  4D = 293D.  Both memory blocks are inserted before the final four-way
  candidate one-hot so the candidate-context ABI remains stable.  The gated
  block gives each genuine joint-bimanual BODex candidate an independent
  state-dependent adapter without modifying the shared actor.

Stages are promoted only after deterministic evaluation reaches at least 80%
success for three consecutive windows. `orchestrate.py` stops if a stage does
not converge and never starts the next stage in that case.

Production training accepts only `xhand_bodex_bimanual_bank_v1`. The synthetic
smoke fixture has a different schema/backend and is rejected by `train.py` and
`orchestrate.py`.

The production data path is deliberately two-step:

1. `migration_4090.xhand_bodex_bimanual.run_solver` transforms object-local
   paired surface seeds into the locked Isaac object pose, runs one joint
   BODex optimization over both hands, and attaches a collision-aware two-palm
   lift IK. It writes `xhand_bodex_bimanual_solver_results_v1`.
2. `migration_4090.xhand_bodex_bimanual.export_bank` applies descriptor-space
   near-duplicate filtering and writes the only schema accepted by training.

The orchestrator always trains and evaluates one stage at a time. A following
stage is not launched unless the current stage has three consecutive fixed
evaluation windows with at least 80% success and no earlier-gate regression.

## Formation bridge retry

`formation_5mm_close_wrist_v1` is a non-promotional diagnostic profile for
weak BODex candidates. It preserves the controller-generated BODex finger
closure, keeps the 293D observation/38D action ABI, and exposes only
`left/right_j5-j7` residual rows. Those six rows are integrated only during
the last half of `PHASE_CLOSE`, capped at `0.035` rad, and then held fixed
through lift/hold. This is intentionally separate from Stage 2, whose default
action mask remains hand-only.

`formation_5mm_close_wrist_retry_v2` is the conservative follow-up for PPO
drift diagnostics. It has the same ABI and activation timing, but caps the
distal-wrist residual at `0.020` rad and integrates it at `0.001` per control
step. It must be evaluated independently; it does not replace the v1 profile
or the historical champion checkpoint.

For a resumed checkpoint, zero the newly exposed rows and update only those
rows, for example:

```text
--profile formation_5mm_close_wrist_v1 \
--trainable-action-names left_j5,left_j6,left_j7,right_j5,right_j6,right_j7 \
--zero-output-action-names-after-load left_j5,left_j6,left_j7,right_j5,right_j6,right_j7
```

## Fixed-pose palm-alignment ablation

`formation_5mm_fixed_pose_palm_align_v1` is an independent diagnostic task. It
always selects BODex Candidate 2, disables XY/yaw/joint reset noise through
`nominal_pose_lock=true`, and keeps the full 293D/38D policy ABI. The phrase
"no random seed" refers to the initial pose: PPO still receives a numeric seed
so optimizer behavior and comparisons remain reproducible. A symmetric
left/right palm-center cosine reward (weight `3.0`) is active only in approach
and close, so it cannot pull an already stable lift apart. The strict contact,
penetration, lift, and stability gates are unchanged.

For a weak-candidate retry, `bridge_train.py` also supports
`--actor-update-scope candidate_gated_action_rows`. This jointly adapts the
selected candidate-gated memory blocks and the active wrist output rows while
freezing the shared actor backbone and all other action rows. Use a fresh
optimizer state and evaluate all four candidates after every short training
chunk; this diagnostic scope does not bypass the normal promotion gates.

## Hold-only wrist settling

`stable_5mm_hold_wrist_guard_v1` is the conservative successor to the
close-active wrist guard. It leaves approach, finger closure, and the complete
paired-palm IK lift on the nominal BODex controller. The policy may integrate
at most 8 mrad of distal-wrist correction during the first half of hold; the
correction is then frozen for the rest of the episode. This profile is intended
for learning post-lift damping without sacrificing nominal contact formation.
New wrist output rows must be zero-initialized when branching from a checkpoint,
and every chunk remains non-promotional until a fixed four-candidate evaluation
improves controlled lift, stable hold, and sustained lift without a contact
regression.

## Adaptive Stage 3 and production collection

The sphere diagnosis showed that the original Stage 3 intersection can be empty
even when its individual gates are nonzero. In particular, requiring three
semantic contact groups on both hands, terminal stability for 32 consecutive
steps, and a historical maximum penetration of 5 mm at the same time removes
most otherwise useful grasp families. The following opt-in profiles preserve
the formal Stage 3 definition while exposing a learnable curriculum:

- `stage3a_diverse`: at least two groups per hand and five total, current
  penetration at most 25 mm, and at least eight historically stable steps with
  a 2 mm height tolerance;
- `stage3b_quality`: at least three groups per hand, historical penetration at
  most 10 mm, and at least eight historically stable steps without height
  tolerance;
- `stage3_strict`: the unchanged formal Stage 3 gates (three groups per hand,
  32 terminal stable steps, and historical penetration at most 5 mm).

Stage 3A is the diversity tier, Stage 3B is the quality tier, and strict Stage 3
is a high-quality subset. Stage 3A results must not be reported as strict
successes. Training and evaluation accept explicit gate overrides, and training
also accepts `--save-interval`; use short intervals because deterministic
success frequently peaks before the final PPO iteration.

`evaluate.py --successful-state-output <shard.pt>` writes every passing final
38D robot state together with the 13D object state, integrated residual,
candidate identity, and complete physics audit. Random reset noise remains
enabled unless `--nominal-pose-lock` is supplied. The resume-safe multi-GPU
collector repeatedly creates such shards, then performs quantized joint-state
deduplication and round-robin candidate balancing:

```bash
python -m migration_4090.xhand_rl_staged.collect_success_shards \
  --evaluator-python /path/to/objectflow_xhand_isaaclab/bin/python \
  --manifest migration_4090/config/xhand_locked_six_rl_embedded_v2.json \
  --object sphere \
  --bodex-bank /path/to/sphere_controller_bank.pt \
  --checkpoint /path/to/model_25.pt \
  --output-root /path/to/sphere_stage3a_100k \
  --target-count 100000 \
  --raw-oversample 1.5 \
  --episodes-per-shard 4096 \
  --num-envs 256 \
  --gpus 1,2,3,4 \
  --candidate-repeat-factors 8,2,3,4 \
  --profile stage3a_diverse
```

The collector can be stopped and restarted with the same output root; completed
shards are hash-audited and reused. Partial multi-GPU waves retain every
successful shard, allocate later shards after the highest existing index, and
stop only after three consecutive fully failed waves. On this server, 256 environments used about
3.6 GB on one RTX 4090 and measured 6,534 episodes/hour and 689 accepted
Stage-3A states/hour for the tested sphere checkpoint. A 512-environment launch
failed in the Isaac/PhysX allocator and is not a supported production setting.
Run separate object-specific collectors when building a multi-object dataset,
then allocate the global target count explicitly across objects.
Candidate repeat factors weight rollout allocation before collection; the
merger still performs quantized deduplication and round-robin candidate
balancing. For the measured sphere checkpoint, `8,2,3,4` approximately
equalizes accepted yields across its four candidate types.
