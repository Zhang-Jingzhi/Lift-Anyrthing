# Minimal migration to the RTX 4090 server

This repository contains the smallest practical, reconstructable code bundle.
Large or licensed artifacts are deliberately kept outside Git.

## What must be moved

| Item | Where it comes from | Required? |
|---|---|---|
| This private Git repository | GitHub | Yes |
| Official TRO-Grasp `data.zip` | Project Google Drive link | Yes |
| Current bimanual dataset | Your Google Drive, saved as `data/bimanual/bimanual_dataset.pt` | Yes for training |
| Single-hand source `vis.pt` | Your Google Drive, saved as `data/bimanual/source_vis.pt` | Only when generating more pairs |
| Isaac Gym Preview 4 | NVIDIA download | Yes for physics validation/generation |
| Official `vqvae.ckpt` | Official checkpoint archive | Only for training/evaluation |
| Other model checkpoints | Google Drive | No for data generation; optional for resume/evaluation |

Do **not** move `graph_exp/`, WandB runs, caches, screenshots, old pilot data,
failed candidates, Conda directories, or Isaac Gym binaries from the old PC.

The small custom overlay under `runtime_assets/` is tracked in Git. It contains
the verified mirrored Allegro right hand, the left/right robot point clouds,
the robot metadata entries, and the two generated extra-large tabletop object
assets. Its size is about 1.8 MB.

## Server setup

```bash
git clone https://github.com/Zhang-Jingzhi/TRO-Grasp-Reproduction.git
cd TRO-Grasp-Reproduction
```

Download and extract the official dataset so that `data/CMapDataset*`,
`data/data_urdf`, and `data/PointCloud` exist. Then apply the tracked overlay:

```bash
bash scripts/install_runtime_assets.sh
mkdir -p data/bimanual
# Copy/download bimanual_dataset.pt and, when needed, source_vis.pt here.
```

Build the two environments:

```bash
conda env create -f environment/tro.yml
conda activate tro
python -m pip install -r environment/requirements-tro.txt

conda env create -f environment/isaac.yml
conda activate isaac
python -m pip install -r environment/requirements-isaac.txt
```

Install Isaac Gym Preview 4 separately in the `isaac` environment, following
`environment/README.md`. Do not upload the Isaac archive or installed package
to GitHub. Define the interpreter once per shell/session:

```bash
export ISAAC_PYTHON="$(conda run -n isaac which python)"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$ISAAC_PYTHON")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Check the reconstruction before a long job:

```bash
conda run -n tro python scripts/verify_4090_setup.py
"$ISAAC_PYTHON" -c "from isaacgym import gymapi; print('Isaac Gym import OK')"
```

## First smoke tests

Run one GPU first; the data-generation pipeline is parallelized by launching
independent object/job shards on different GPUs rather than by distributed
training inside one process.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n tro python train.py \
  --config config/train_bimanual_4090.yaml \
  --max-batches-per-epoch 1 \
  --save-dir graph_exp/bimanual_training/4090_smoke
```

For additional data generation, the portable defaults now use
`data/bimanual/source_vis.pt`, `config/bimanual_object_split.json`, and the
`ISAAC_PYTHON` environment variable:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n tro python generate_bimanual_pilot.py \
  --gpu 0 \
  --output-dir graph_exp/bimanual_data/server_run_0
```

The 4090 training configuration starts at batch size 8. Increase it only after
the smoke test and GPU-memory observation; this avoids assuming that model
memory scales exactly with the card's nominal VRAM.
