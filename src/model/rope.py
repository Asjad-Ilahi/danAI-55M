"""
Rotary Position Embeddings (RoPE) with Zero-Training NTK-Aware & Linear Context Scaling.

Supports:
- Standard RoPE (Su et al., 2021)
- NTK-Aware RoPE Scaling (permits 2x, 4x, 8x context extension with ZERO retraining)
- Dynamic cache expansion up to any sequence length
- MPS/CUDA/CPU fast operations
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) with NTK-Aware Scaling.
    
    RoPE encodes position by rotating pairs of dimensions in the Q/K vectors.
    NTK-Aware scaling modifies the base theta frequency so high-frequency local resolution
    is preserved while low-frequency components are smoothly distributed over long distances.
    
    Args:
        head_dim: Dimension of each attention head (e.g. 64)
        max_seq_len: Maximum sequence length to pre-compute embeddings for
        theta: Base for rotation frequencies (default: 10000.0)
        scaling_factor: Context extension multiplier (e.g. 2.0 for 2x context, 4.0 for 4x)
        scaling_type: Scaling algorithm ('ntk', 'linear', or 'none')
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "ntk",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.scaling_factor = float(scaling_factor)
        self.scaling_type = str(scaling_type).lower() if scaling_type else "none"

        # Apply NTK-Aware base theta adjustment if scaling > 1.0
        effective_theta = float(theta)
        if self.scaling_type == "ntk" and self.scaling_factor > 1.0:
            # NTK-Aware formula: theta' = theta * factor^(head_dim / (head_dim - 2))
            ntk_exponent = head_dim / max(1, head_dim - 2)
            effective_theta = theta * (self.scaling_factor ** ntk_exponent)

        self.theta = effective_theta

        # Compute inverse frequencies
        if self.scaling_type == "linear" and self.scaling_factor > 1.0:
            inv_freq = 1.0 / (
                effective_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            ) / self.scaling_factor
        else:
            inv_freq = 1.0 / (
                effective_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute cos and sin tables for positions [0, seq_len)."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor x.
        Shape of x: (batch, num_heads, seq_len, head_dim)
        """
        seq_len = x.shape[2]
        
        # Extend cache dynamically if sequence exceeds pre-computed length
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        if position_ids is not None:
            max_pos = int(position_ids.max().item()) + 1
            if max_pos > self.max_seq_len:
                self._build_cache(max_pos)
                self.max_seq_len = max_pos
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
        else:
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Rotate half: [-x2, x1]
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated_x = torch.cat([-x2, x1], dim=-1)
        
        return (x * cos) + (rotated_x * sin)
