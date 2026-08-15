"""
Rotary Position Embeddings (RoPE).

Implements RoPE with configurable theta, cached cos/sin tables,
MPS-safe operations. Applied to Q and K tensors (including reduced
KV heads for GQA).

Implemented from scratch per §16.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) from "RoFormer: Enhanced Transformer
    with Rotary Position Embedding" (Su et al., 2021).
    
    RoPE encodes position by rotating pairs of dimensions in the Q/K vectors.
    This gives the model relative position awareness through the attention dot product,
    without any absolute positional embeddings.
    
    The rotation frequencies are: theta_i = theta^(-2i/d) for dimension pair i,
    where d is the head dimension and theta is a configurable base (default 10000).
    
    Args:
        head_dim: Dimension of each attention head
        max_seq_len: Maximum sequence length to pre-compute embeddings for
        theta: Base for the rotation frequencies (default: 10000)
    """
    
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute inverse frequencies: theta^(-2i/d) for i in [0, d/2)
        # Shape: (head_dim // 2,)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Register as buffer (not a parameter, but saved in state_dict)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute and cache cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute cos and sin tables for positions [0, seq_len)."""
        # Position indices: (seq_len,)
        t = torch.arange(seq_len, dtype=torch.float32)
        
        # Outer product: (seq_len, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate for both elements of each rotated pair: (seq_len, head_dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Cache cos and sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, head_dim)
            position_ids: Optional position indices of shape (batch, seq_len)
                         If None, uses positions [0, 1, ..., seq_len-1]
        
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        if position_ids is not None:
            # Gather cos/sin for specific positions
            # cos_cached: (max_seq_len, head_dim)
            cos = self.cos_cached[position_ids]  # (batch, seq_len, head_dim)
            sin = self.sin_cached[position_ids]
            # Add head dimension: (batch, 1, seq_len, head_dim)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            # Use sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Ensure same dtype as input
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        
        return _apply_rotary(x, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the last dimension to create the rotation pair.
    
    For input [..., x1, x2, ..., xd/2, xd/2+1, ..., xd],
    returns  [..., -xd/2+1, ..., -xd, x1, ..., xd/2]
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to a tensor.
    
    Uses the formula: rotary(x) = x * cos(theta) + rotate_half(x) * sin(theta)
    
    This rotates each pair of dimensions (2i, 2i+1) by angle theta_i * position,
    which encodes relative position information in the attention dot product.
    """
    return x * cos + _rotate_half(x) * sin
