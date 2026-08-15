"""
Memory tracking and reporting utilities per §26.
"""

import os
import sys
import torch
from typing import Dict, Any

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_info(device: torch.device) -> Dict[str, Any]:
    """
    Get current memory usage and system headroom.
    """
    info = {
        "system_ram_gb": 0.0,
        "available_ram_gb": 0.0,
        "process_rss_mb": 0.0,
        "device_allocated_mb": 0.0,
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        info["system_ram_gb"] = mem.total / (1024 ** 3)
        info["available_ram_gb"] = mem.available / (1024 ** 3)
        info["process_rss_mb"] = proc.memory_info().rss / (1024 ** 2)

    if device.type == "cuda":
        info["device_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
    elif device.type == "mps":
        # MPS doesn't expose a simple torch memory query API in all versions; RSS process memory is the primary metric
        info["device_allocated_mb"] = info["process_rss_mb"]

    return info


def print_memory_report(device: torch.device, param_count: int) -> None:
    """Print system memory report before training per §26."""
    info = get_memory_info(device)

    print("\n" + "=" * 60)
    print("MEMORY HEADROOM & CAPACITY REPORT")
    print("=" * 60)
    if HAS_PSUTIL:
        print(f"  Total System RAM:     {info['system_ram_gb']:.2f} GB")
        print(f"  Available RAM:        {info['available_ram_gb']:.2f} GB")
        print(f"  Process Memory (RSS): {info['process_rss_mb']:.1f} MB")
    print(f"  Target Device:        {device}")
    print(f"  Model Parameter Count: {param_count:,}")

    # Estimate training memory requirement
    model_fp32_mb = param_count * 4 / (1024 ** 2)
    model_bf16_mb = param_count * 2 / (1024 ** 2)
    adam_mb = param_count * 8 / (1024 ** 2)
    total_est_mb = model_bf16_mb + adam_mb + 500  # +500MB activation buffer

    print(f"  Est. Model (bf16):    {model_bf16_mb:.1f} MB")
    print(f"  Est. Optimizer (Adam): {adam_mb:.1f} MB")
    print(f"  Est. Total Peak:      ~{total_est_mb:.1f} MB (~{total_est_mb/1024:.2f} GB)")
    print("=" * 60 + "\n")
