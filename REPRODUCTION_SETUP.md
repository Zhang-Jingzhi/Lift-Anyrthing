# Reproduction snapshot

This repository is a lightweight code snapshot of the local TRO-Grasp
reproduction and bimanual-data work. It is based on upstream commit
`8e9719777a787b347dc60ac998c63900d057229e` from
[Barrybarry-Smith/TRO-Grasp](https://github.com/Barrybarry-Smith/TRO-Grasp).

## Included

- Upstream model, dataset, training, inference, visualization, and validation
  code.
- Ubuntu 22.04 / RTX 3070 compatibility changes.
- Checkpoint evaluation and training-monitoring utilities.
- Large-object analysis and bimanual candidate generation/validation code.
- RTX 3070 smoke-test and long-training configurations.
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

The current bimanual pipeline generates opposed two-hand candidates, runs
Isaac six-direction perturbation validation, and performs realized-mesh
penetration auditing. The current pilot uses two identical Allegro left-hand
actors because a verified mirrored right-hand URDF is not yet available.
Treat it as a pipeline-validation dataset rather than hardware-faithful
left/right-hand training data.

See [environment/README.md](environment/README.md) for server reconstruction.

