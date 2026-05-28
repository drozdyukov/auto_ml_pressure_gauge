from __future__ import annotations

import subprocess


def get_nvidia_smi_summary() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    first_line = completed.stdout.strip().splitlines()[0]
    name, memory_mb, driver_version = [part.strip() for part in first_line.split(",", maxsplit=2)]
    return {"name": name, "memory_mb": memory_mb, "driver_version": driver_version}


def assert_cuda_ready(expected_gpu: str | None = None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install NVIDIA driver, CUDA-compatible PyTorch, "
            "and run this on the RTX 3060 machine."
        )
    gpu_name = torch.cuda.get_device_name(0)
    if expected_gpu and expected_gpu.lower() not in gpu_name.lower():
        raise RuntimeError(f"Expected GPU similar to '{expected_gpu}', but found '{gpu_name}'.")
