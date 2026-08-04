# Reproduction snapshot

This repository is a lightweight code snapshot of the local TRO-Grasp
reproduction and bimanual-data work. It is based on upstream commit
`8e9719777a787b347dc60ac998c63900d057229e` from
[Barrybarry-Smith/TRO-Grasp](https://github.com/Barrybarry-Smith/TRO-Grasp).

## Included

- Upstream model, dataset, training, inference, visualization, and validation
  code.
- Ubuntu compatibility changes and a portable RTX 4090 starter config.
- Checkpoint evaluation and training-monitoring utilities.
- Large-object analysis and bimanual candidate generation/validation code.
- RTX 3070 history plus a conservative RTX 4090 training configuration.
- Portable environment descriptions under `environment/`.
- A text-only reproduction report.

## Excluded

- Model checkpoints and checkpoint archives.
- Training/evaluation datasets.
- Generated bimanual samples and all `graph_exp/` outputs.
- WandB logs, Python caches, package caches, vendored dependency clones, and
  Conda environment archives.

These exclusions keep the repository small and prevent machine-specific or
licensed binary content from entering Git history. Official data and
checkpoint download links remain in `README.md`.

## Bimanual status

The current bimanual pipeline uses a geometrically and Isaac-validated Allegro
left/right pair. It generates opposed lateral tabletop candidates under
gravity, runs simultaneous closure and lift, applies six independent
disturbances, audits realized-mesh penetration/contact/hand clearance, rejects
single-hand-supported solutions, and supports repeated-validation gates. The
small custom runtime overlay is tracked under `runtime_assets/`; generated
datasets and rollouts remain external.

See [MIGRATE_TO_4090.md](MIGRATE_TO_4090.md) for the minimal server migration
and [environment/README.md](environment/README.md) for environment rebuilding.
