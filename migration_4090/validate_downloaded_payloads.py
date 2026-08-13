"""Validate that the downloaded training data and VQ-VAE checkpoint load."""

import torch


dataset = torch.load(
    "data/CMapDataset_filtered/cmap_dataset.pt",
    map_location="cpu",
)
print(f"CMAP_TYPE {type(dataset).__name__}")
print(
    "CMAP_KEYS "
    + (" ".join(sorted(dataset)) if isinstance(dataset, dict) else "n/a")
)
print(
    "CMAP_METADATA_COUNT "
    + str(len(dataset.get("metadata", [])) if isinstance(dataset, dict) else "n/a")
)

checkpoint = torch.load("ckpt/vqvae.ckpt", map_location="cpu")
print(f"VQVAE_TYPE {type(checkpoint).__name__}")
print(
    "VQVAE_KEYS "
    + (" ".join(sorted(checkpoint)) if isinstance(checkpoint, dict) else "n/a")
)

tensors = []
stack = [checkpoint]
seen = set()
while stack:
    value = stack.pop()
    if id(value) in seen:
        continue
    seen.add(id(value))
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, dict):
        stack.extend(value.values())
    elif isinstance(value, (list, tuple)):
        stack.extend(value)

floating_tensors = [tensor for tensor in tensors if tensor.is_floating_point()]
all_finite = all(bool(torch.isfinite(tensor).all()) for tensor in floating_tensors)
print(f"VQVAE_TENSORS {len(tensors)}")
print(f"VQVAE_FLOATING_TENSORS {len(floating_tensors)}")
print(f"VQVAE_ALL_FINITE {all_finite}")
if not tensors or not all_finite:
    raise RuntimeError("VQ-VAE checkpoint tensor validation failed")

print("DATA_CHECK_OK")
