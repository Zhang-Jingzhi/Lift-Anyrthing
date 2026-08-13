# RTX 4090 / XHand migration tools

This directory contains the reproducible code path used to generate and
validate Tianji + dual-XHand grasp poses.  It intentionally does not contain
the downloaded meshes, Isaac Gym installation, candidate banks, checkpoints,
or experiment outputs.  Those files stay local to the machine or are obtained
separately.

## Current recommended smoke flow

The current compact smoke protocol is defined by
`xhand_compact_formal_1200_v1_protocol.json`.  The six object categories are
`sphere`, `cube`, `cracker`, `bleach`, `pitcher`, and `drill`.

The main components are:

1. `generate_xhand_compact_candidate_bank_v2.py` — generate IK candidates.
2. `objectflow_pose_preview/materialize_isaac_smoke.py` — materialize one
   candidate into the Isaac validator schema.
3. `validate_xhand_fullbody_isaac.py` — run the GPU PhysX closure, lift,
   gravity, disturbance, penetration, and single-hand ablation gates.
4. `xhand_visual_mesh_gate.py` — optional high-resolution triangle-mesh gate.
5. `save_xhand_compact_formal_success.py` — save only samples that pass the
   strict repeated-rollout gate.

Run the six-object smoke search in a detached screen session with:

```bash
TRO_REPO=/path/to/TRO-Grasp-Reproduction \
TRO_PYTHON=/path/to/tro/bin/python \
ISAAC_PYTHON=/path/to/isaac/bin/python \
XHAND_BANK_ROOT=/path/to/xhand_six_object_smoke_v1_v3/banks \
XHAND_SMOKE_ROOT=/path/to/xhand_six_object_smoke_v4 \
screen -dmS xhand_six_smoke bash -lc \
  'cd "$TRO_REPO" && migration_4090/run_current_xhand_smoke.sh'
```

`run_current_xhand_smoke.sh` is a thin, configurable wrapper around
`search_six_object_smoke_v4.sh`.  Set `XHAND_FULLBODY_URDF` and
`XHAND_FULLBODY_IK_URDF` to the local assembled XHand URDFs.  Set
`ISAAC_GYM_PYTHON` or `PYTHONPATH` to the separately installed Isaac Gym
Preview 4 Python package.

## What is deliberately excluded

Do not commit `*.pt`, `*.ckpt`, `*.pth`, meshes, Isaac Gym binaries,
`graph_exp/`, WandB logs, rendered galleries, or validation logs.  The root
`.gitignore` contains the migration-output rules.  Use a separate artifact
store or a handoff directory for large datasets and reports.

## Historical scripts

The remaining `run_*_vN.sh`, `*_probe.sh`, `*_retry*.sh`, and old object-set
scripts are retained for auditability.  They are not current entry points.
Use the files listed above unless reproducing a named historical experiment.
