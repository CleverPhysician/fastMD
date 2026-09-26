"""ASE-first inference for CUDA Graph MLIPs."""
from .calculator import FastMDCalculator
from .config import CUDAGraphConfig
from .models import ModelBackend, ModelCapabilities, available_models, register_model
from .relaxation import GPUFireConfig, GPUFireDeviceResult, GPUFireResult, run_gpu_fire, run_gpu_fire_device, run_gpu_fire_graph

__version__ = "0.1.0"
__all__ = [
    "FastMDCalculator",
    "CUDAGraphConfig",
    "GPUFireConfig",
    "GPUFireDeviceResult",
    "GPUFireResult",
    "run_gpu_fire",
    "run_gpu_fire_device",
    "run_gpu_fire_graph",
    "ModelBackend",
    "ModelCapabilities",
    "available_models",
    "register_model",
]
