"""Hardware introspection and device selection."""
from __future__ import annotations

import platform
import subprocess
from typing import Optional


def hardware_info() -> dict:
    """Return a dict with platform, chip, RAM, and torch/mps/cuda version info."""
    import torch

    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    # RAM (cross-platform)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:
            # macOS fallback
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            info["ram_gb"] = round(int(out) / 1e9, 1)
        except Exception:
            info["ram_gb"] = None

    # Apple chip name
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        info["chip"] = chip
    except Exception:
        info["chip"] = platform.processor() or "unknown"

    # GPU device names (CUDA)
    if torch.cuda.is_available():
        info["cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    else:
        info["cuda_devices"] = []

    return info


def pick_device(requested: Optional[str] = None) -> str:
    """Auto-pick best device; honour an explicit request if valid."""
    import torch

    if requested is not None and requested not in ("auto", ""):
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
