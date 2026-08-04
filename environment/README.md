# Rebuilding the two environments

The project deliberately uses two environments:

- `tro`: training, generation, optimization, analysis, and visualization.
- `isaac`: Isaac Gym Preview 4 physics validation.

The files in this directory contain no machine-specific prefixes or local file
URLs. They describe the known-working reproduction environment and are the
recommended minimal starting point for the RTX 4090 server. These manifests,
rather than multi-gigabyte Conda archives, are versioned because they are
portable, inspectable, and much smaller.

## 1. Main TRO-Grasp environment

```bash
conda env create -f environment/tro.yml
conda activate tro
python -m pip install -r environment/requirements-tro.txt
```

Verify the main CUDA stack:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

The working environment uses Python 3.10, PyTorch 2.4.1, torchvision 0.19.1,
and CUDA 12.1. `requirements-tro.txt` pins the remaining Python packages and
the exact Git commits used for source-installed dependencies.

## 2. Isaac Gym validation environment

```bash
conda env create -f environment/isaac.yml
conda activate isaac
python -m pip install -r environment/requirements-isaac.txt
```

Download **Isaac Gym Preview 4** from NVIDIA separately. Its licensed package
and installed files are not committed to this repository:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
python -m pip install -e isaacgym/python
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify it before launching a large evaluation:

```bash
python -c "from isaacgym import gymapi; print('Isaac Gym import OK')"
cd isaacgym/python/examples
python joint_monkey.py
```

The working validation environment uses Python 3.8, PyTorch 1.8.1,
torchvision 0.9.1, and CUDA Toolkit 11.1. A modern NVIDIA driver is
backward-compatible with this CUDA runtime, but Isaac Gym Preview 4 is old
software; run a one-GPU smoke test on the RTX 4090 server before distributing
jobs across all eight GPUs.

The bimanual orchestration scripts run generation in `tro` and launch physics
validation as a subprocess. Pass the second environment explicitly:

```bash
conda activate tro
python generate_bimanual_pilot.py \
  --isaac-python "$ISAAC_PYTHON" \
  [other arguments]
```

## 3. Data and checkpoints

Neither datasets nor checkpoints are stored in Git. Download the official
archives using the links in the root `README.md`. Place the official data and
the separately downloaded bimanual data in:

```text
TRO-Grasp-Reproduction/
├── data/
│   └── bimanual/
│       ├── bimanual_dataset.pt
│       └── source_vis.pt       # only needed to generate more pairs
└── ckpt/
```

Only `vqvae.ckpt` is required for bimanual training. Other checkpoints are
needed only for resume/evaluation. The reproduction-specific experiment
outputs are intentionally omitted.

## 4. Working environment reference

Main environment:

- Python 3.10.14
- PyTorch 2.4.1 / torchvision 0.19.1
- CUDA runtime 12.1
- NumPy 1.26.4
- SciPy 1.15.3
- trimesh 4.7.4
- JAX 0.6.2
- viser 1.0.6

Isaac environment:

- Python 3.8.20
- PyTorch 1.8.1 / torchvision 0.9.1
- CUDA Toolkit 11.1
- NumPy 1.24.4
- SciPy 1.10.1
- Isaac Gym 1.0rc4 / Preview 4
