import importlib

import torch


for module in (
    "trimesh",
    "pytorch_kinematics",
    "embreex",
    "yourdfpy",
):
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

conv = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1).cuda()
conv_result = conv(torch.rand(2, 8, 32, 32, device="cuda"))
torch.cuda.synchronize()
assert torch.isfinite(conv_result).all()
print(f"TORCH_CUDNN_VERSION {torch.backends.cudnn.version()}")
print(f"TORCH_CUDNN_CONV_MEAN {conv_result.mean().item():.9f}")
print(f"TORCH_MEMORY_ALLOCATED_MIB {torch.cuda.memory_allocated() / 2**20:.2f}")
print(f"TORCH_MEMORY_RESERVED_MIB {torch.cuda.memory_reserved() / 2**20:.2f}")
print("ISAAC_BASE_RUNTIME_VALIDATION_OK")
