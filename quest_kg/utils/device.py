"""GPU/CUDA helpers. Defaults to CUDA when available per user preference."""
from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the requested device if available, else fall back to CPU.

    User preference: always use CUDA on GPU when available.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_info() -> dict:
    """Print and return CUDA device info; used at the top of every Colab session."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    print(info)
    return info
