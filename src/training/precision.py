"""
Precision management and NaN/Inf checking per §1 & §30.
"""

import torch
import math
from typing import Tuple, Optional


class PrecisionManager:
    """Manages mixed precision autocast context and stability validation."""

    def __init__(self, device: torch.device, precision: torch.dtype):
        self.device = device
        self.precision = precision

        self.use_autocast = precision in (torch.bfloat16, torch.float16)
        self.autocast_device_type = device.type if device.type in ("cuda", "cpu") else "mps"

        # GradScaler is primarily used for CUDA fp16; for MPS bf16/fp16 or CPU, standard autocast is used
        self.scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and precision == torch.float16) else None

    def get_autocast_context(self):
        """Get context manager for PyTorch autocast."""
        if not self.use_autocast:
            return torch.no_grad() if not torch.is_grad_enabled() else _NullContext()

        if self.device.type == "mps":
            # MPS autocast support varies by PyTorch version; fallback if not present
            if hasattr(torch, "autocast"):
                return torch.autocast(device_type="mps", dtype=self.precision)
            return _NullContext()
        else:
            return torch.autocast(device_type=self.device.type, dtype=self.precision)

    def check_loss_stability(self, loss_tensor: torch.Tensor, step: int) -> None:
        """Check loss value for NaN or Inf. Fail loudly per §30."""
        loss_val = loss_tensor.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(
                f"\n[FATAL ERROR] Loss exploded to {loss_val} at training step {step}! "
                f"Precision used: {self.precision}, device: {self.device}. "
                f"Training halted per §30."
            )


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
