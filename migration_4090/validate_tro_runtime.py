import importlib
import math

import jax
import jax.numpy as jnp
import torch
import nvidia.cudnn

print(f"CUDNN_PYTHON_PATH {list(nvidia.cudnn.__path__)[0]}")


modules = [
    "trimesh",
    "pytorch_kinematics",
    "pytorch3d",
    "pyroki",
]
for module in modules:
    imported = importlib.import_module(module)
    print(f"IMPORT_OK {module} {getattr(imported, '__version__', 'unknown')}")

print(f"TORCH_VERSION {torch.__version__}")
print(f"TORCH_CUDA_BUILD {torch.version.cuda}")
print(f"TORCH_CUDA_AVAILABLE {torch.cuda.is_available()}")
assert torch.cuda.is_available()
print(f"TORCH_DEVICE_COUNT {torch.cuda.device_count()}")
print(f"TORCH_GPU_NAME {torch.cuda.get_device_name(0)}")
print(f"TORCH_GPU_CAPABILITY {torch.cuda.get_device_capability(0)}")

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b
torch.cuda.synchronize()
assert torch.isfinite(c).all()
print(f"TORCH_MATMUL_MEAN {c.mean().item():.9f}")
print(f"TORCH_MEMORY_ALLOCATED_MIB {torch.cuda.memory_allocated() / 2**20:.2f}")
print(f"TORCH_MEMORY_RESERVED_MIB {torch.cuda.memory_reserved() / 2**20:.2f}")

from pytorch3d.ops import knn_points

x = torch.rand(1, 128, 3, device="cuda")
y = torch.rand(1, 96, 3, device="cuda")
knn = knn_points(x, y, K=3)
torch.cuda.synchronize()
assert torch.isfinite(knn.dists).all()
print(f"PYTORCH3D_KNN_MEAN {knn.dists.mean().item():.9f}")

jax_devices = jax.devices()
print("JAX_DEVICES " + ", ".join(str(device) for device in jax_devices))
assert any(device.platform == "gpu" for device in jax_devices)
jax_result = jnp.sin(jnp.arange(1024.0)).sum()
jax_value = float(jax_result.block_until_ready())
assert math.isfinite(jax_value)
print(f"JAX_GPU_SUM {jax_value:.9f}")

conv = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1).cuda()
conv_result = conv(torch.rand(2, 8, 32, 32, device="cuda"))
torch.cuda.synchronize()
assert torch.isfinite(conv_result).all()
print(f"TORCH_CUDNN_VERSION {torch.backends.cudnn.version()}")
print(f"TORCH_CUDNN_CONV_MEAN {conv_result.mean().item():.9f}")
print("TRO_RUNTIME_VALIDATION_OK")
