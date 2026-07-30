# TRO-Grasp Reproduction Report

## Scope

This report records a checkpoint-based reproduction of:

- ShadowHand unconditioned grasp synthesis.
- Allegro initial-pose-conditioned grasp synthesis.
- Isaac Gym GPU validation.
- Interactive visualization.
- A minimal offline training smoke test.

The paper values are taken from Table I and the experimental settings in the
[TRO-Grasp paper](https://arxiv.org/abs/2510.12724).

## System

- Repository commit: `8e9719777a787b347dc60ac998c63900d057229e`
- Operating system: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB VRAM
- NVIDIA driver: 580.126.09
- Main environment: Python 3.10.14, PyTorch 2.4.1, CUDA 12.1, JAX 0.6.2
- Isaac environment: Python 3.8.20, PyTorch 1.8.1, CUDA 11.1
- Isaac Gym: Preview 4 / `isaacgym 1.0rc4`
- Evaluation checkpoint: `ckpt/multi_hand.pth`
- VQ-VAE checkpoint: `ckpt/vqvae.ckpt`

## Evaluation Protocol

- Ten unseen validation objects.
- 100 generated grasps per object.
- 1,000 grasps per evaluation mode.
- DDIM inference steps: 20.
- Isaac Gym success validation.
- Pyroki IK executed on CPU to preserve GPU memory.
- Model inference split batch size: 4.

The split batch size changes peak memory usage but does not change the number
of evaluated grasps.

## Results

| Experiment | Metric | Paper | Local | Difference |
|---|---|---:|---:|---:|
| ShadowHand unconditioned | Success rate | 98.60% | 98.30% | -0.30 pp |
| ShadowHand unconditioned | Diversity | 0.292 | 0.295753 | +0.003753 |
| ShadowHand unconditioned | Time per grasp | 0.25 s | 0.129380 s | -0.120620 s |
| Allegro pose-conditioned | Success rate | 93.70% | 92.00% | -1.70 pp |
| Allegro pose-conditioned | Diversity | 0.430 | 0.406868 | -0.023132 |
| Allegro pose-conditioned | Time per grasp | 0.21 s | 0.085124 s | -0.124876 s |

The success-rate and diversity results are close to the reported values for a
single stochastic run. Timing values should not be interpreted as a direct
hardware comparison: the paper reports NVIDIA A100 40 GB experiments, while
this run used an RTX 3070 Laptop GPU, split batches of four, CPU Pyroki, and
excluded the first warm-up split following the repository code.

## Local Result Files

### ShadowHand unconditioned

- Config: `config/test_palm_unconditioned_rtx3070.yaml`
- Metrics: `graph_exp/reproduction/shadowhand-unconditioned/res.txt`
- Visualization: `graph_exp/reproduction/shadowhand-unconditioned/vis.pt`
- Successful grasps: 983 / 1,000

### Allegro pose-conditioned

- Config: `config/test_palm_conditioned_rtx3070.yaml`
- Metrics: `graph_exp/reproduction/allegro-conditioned/res.txt`
- Visualization: `graph_exp/reproduction/allegro-conditioned/vis.pt`
- Successful grasps: 920 / 1,000

## Visualization

- ShadowHand unconditioned: `http://127.0.0.1:8080`
- Allegro conditioned: `http://127.0.0.1:8081`

Both services load tensors and hand models on CPU.

## Compatibility Changes

- Removed author-specific Conda and GPU paths from `validation/validate_utils.py`.
- Made training-only geometry imports optional in `utils/rotation.py`.
- Replaced the newer PyTorch-only `.mT` use in the Isaac validation path.
- Removed unused visualization imports from `utils/controller.py`.
- Fixed one-sample metric reporting in `test.py`.
- Added CLI file/host/port/index options and CPU loading to `vis.py`.
- Added optional offline and batch-limited smoke-test controls to `train.py`.

## Training Smoke Test

The training entry point was validated without external logging:

- Config: `config/train_smoke_rtx3070.yaml`
- Robot: Allegro
- Training object: `contactdb+alarm_clock`
- Batch size: 1
- Epochs: 1
- Batches per epoch: 1
- WandB mode: offline
- Trainable parameters: 28,059,544
- Checkpoint: `graph_exp/smoke/train-allegro/ckpt/1.pth`

The smoke test completed data preparation, forward propagation, loss
calculation, backward propagation, an Adam update, scheduler update, and
checkpoint serialization. The 301 MB checkpoint was read back successfully
and contains:

- Epoch: 1
- 362 model-state tensors
- 258 optimizer-state entries
- Scheduler state at epoch 1

## Interpretation

The official checkpoint reproduction is successful:

- The complete model-to-IK-to-Isaac pipeline runs on Ubuntu 22.04 despite that
  OS not being officially listed for Isaac Gym Preview 4.
- Both evaluated modes produce success rates close to Table I.
- All 2,000 generated grasps and their Isaac states are present in the saved
  visualization files.

The paper trained for 300 epochs on an NVIDIA A100 40 GB GPU. On an 8 GB
RTX 3070 Laptop GPU, the batch size must be reduced and the same training run
takes substantially longer. The local long-running job is recorded below.

## Full 300-Epoch Training Run

The full multi-hand training run was started in a detached GNU Screen session:

- Screen session: `tro_train_300`
- Config: `config/train_multi_hand_rtx3070.yaml`
- Epochs: 300
- Training samples: 14,011
- Batch size: 3
- Batches per epoch: 4,671
- Total optimizer steps: 1,401,300
- Data-loader workers: 2
- WandB mode: offline
- Log: `graph_exp/reproduction/train-multi-hand-rtx3070.screen.log`
- Output: `graph_exp/reproduction/train-multi-hand-rtx3070`
- Rolling checkpoint: `graph_exp/reproduction/train-multi-hand-rtx3070/ckpt/latest.pth`
- Numbered checkpoint interval: 10 epochs

The run was observed through batch 200 of epoch 1. GPU memory remained stable
at approximately 6.3 GB with no out-of-memory error, and the loss decreased
from 2.269836 on the first batch to 0.623303 at batch 200.

Training-curve monitoring runs in a second detached Screen session:

- Monitor session: `tro_train_monitor`
- Curve HTTP session: `tro_curve_http`
- Live curve: `http://127.0.0.1:8082/training_curve.png`
- Curve image: `graph_exp/reproduction/train-multi-hand-rtx3070/monitor/training_curve.png`
- Batch CSV: `graph_exp/reproduction/train-multi-hand-rtx3070/monitor/training_loss.csv`
- Epoch CSV: `graph_exp/reproduction/train-multi-hand-rtx3070/monitor/epoch_loss.csv`
- Status: `graph_exp/reproduction/train-multi-hand-rtx3070/monitor/status.txt`

The monitor refreshes every 60 seconds. Conservative automatic early stopping
is enabled only after epoch 80. It stops the training Screen when epoch-average
training loss has failed to improve by at least 0.1% for 20 consecutive epochs,
and only if `ckpt/latest.pth` exists.
