"""
Device detection and precision management for the SLM project.

Priority: MPS → CUDA → CPU
Detects bf16 support and selects optimal precision per §1 and §30.
"""

import sys
import torch
from typing import Optional


def get_device() -> torch.device:
    """
    Select the best available device.
    
    Priority: MPS → CUDA → CPU
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_bf16_support(device: torch.device) -> bool:
    """
    Check whether the device/PyTorch combination supports bfloat16.
    
    For MPS: attempts a small bf16 matmul to verify actual support,
    since MPS bf16 support depends on PyTorch version and macOS version.
    For CUDA: checks compute capability >= 8.0 (Ampere+).
    For CPU: bf16 is generally supported in recent PyTorch.
    """
    if device.type == 'mps':
        try:
            a = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
            b = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
            c = a @ b
            # Force sync to catch deferred errors
            c.cpu()
            return True
        except (RuntimeError, TypeError):
            return False
    elif device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(device)
            return props.major >= 8  # Ampere+
        except Exception:
            return False
    else:  # CPU
        try:
            a = torch.randn(2, 2, dtype=torch.bfloat16)
            b = torch.randn(2, 2, dtype=torch.bfloat16)
            _ = a @ b
            return True
        except (RuntimeError, TypeError):
            return False


def check_fp16_support(device: torch.device) -> bool:
    """Check whether the device supports float16."""
    try:
        a = torch.randn(2, 2, device=device, dtype=torch.float16)
        b = torch.randn(2, 2, device=device, dtype=torch.float16)
        c = a @ b
        if device.type == 'mps':
            c.cpu()  # Force sync
        return True
    except (RuntimeError, TypeError):
        return False


def select_precision(device: torch.device, requested: str = 'auto') -> torch.dtype:
    """
    Select the best available precision.
    
    Priority when 'auto': bf16 → fp16 → fp32.
    
    bf16 is preferred because it has the same exponent range as fp32 (8 bits),
    making it more stable for training than fp16 (5-bit exponent, narrow dynamic range).
    
    Args:
        device: Target device
        requested: 'auto', 'bf16', 'fp16', or 'fp32'
    
    Returns:
        The selected torch.dtype
    
    Raises:
        RuntimeError: If the requested precision is not supported
    """
    if requested == 'auto':
        if check_bf16_support(device):
            return torch.bfloat16
        elif check_fp16_support(device):
            return torch.float16
        else:
            return torch.float32
    elif requested in ('bf16', 'bfloat16'):
        if not check_bf16_support(device):
            raise RuntimeError(
                f"bfloat16 requested but not supported on {device}. "
                f"Use 'auto' to fall back to fp16/fp32."
            )
        return torch.bfloat16
    elif requested in ('fp16', 'float16'):
        if not check_fp16_support(device):
            raise RuntimeError(
                f"float16 requested but not supported on {device}. "
                f"Use 'auto' to fall back to fp32."
            )
        return torch.float16
    elif requested in ('fp32', 'float32'):
        return torch.float32
    else:
        raise ValueError(f"Unknown precision '{requested}'. Use 'auto', 'bf16', 'fp16', or 'fp32'.")


def get_autocast_dtype(precision: torch.dtype) -> Optional[torch.dtype]:
    """Get the dtype to use with torch.autocast, or None if no autocast needed."""
    if precision in (torch.bfloat16, torch.float16):
        return precision
    return None


def print_device_info(device: torch.device, precision: torch.dtype) -> None:
    """Print comprehensive device and precision information."""
    print("=" * 60)
    print("DEVICE & PRECISION INFORMATION")
    print("=" * 60)
    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  Python version:   {sys.version.split()[0]}")
    print(f"  Device:           {device}")
    
    if device.type == 'mps':
        print(f"  MPS available:    {torch.backends.mps.is_available()}")
        print(f"  MPS built:        {torch.backends.mps.is_built()}")
    elif device.type == 'cuda':
        print(f"  CUDA version:     {torch.version.cuda}")
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU:              {props.name}")
        print(f"  GPU memory:       {props.total_mem / 1024**3:.1f} GB")
    
    bf16_ok = check_bf16_support(device)
    fp16_ok = check_fp16_support(device)
    print(f"  bf16 support:     {'YES' if bf16_ok else 'NO'}")
    print(f"  fp16 support:     {'YES' if fp16_ok else 'NO'}")
    print(f"  Selected precision: {precision}")
    
    if precision == torch.bfloat16:
        print("  → bf16: same exponent range as fp32, more stable for training")
    elif precision == torch.float16:
        print("  → fp16: narrower dynamic range than bf16, may need loss scaling")
    else:
        print("  → fp32: full precision, higher memory usage")
    
    print("=" * 60)
