"""
SwiGLU MLP (Gated Linear Unit with SiLU activation).

Implements the SwiGLU feedforward network used in modern LLMs (LLaMA, PaLM, etc.).
SwiGLU uses three projections instead of two, with a gated activation:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

The intermediate size is ~2.67× hidden_size (8/3×) to match the compute of a
standard 4× GELU MLP, since SwiGLU has 3 matrices vs 2.

Implemented from scratch per §3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward network.
    
    Architecture:
        gate = silu(gate_proj(x))   # Gating signal with SiLU activation
        up   = up_proj(x)           # Value signal
        out  = down_proj(gate * up) # Combine and project back
    
    SiLU (Sigmoid Linear Unit) = x * sigmoid(x), also called "Swish".
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Inner dimension (typically ~2.67× hidden_size for SwiGLU)
        bias: Whether to use bias in linear layers (default: False)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Three projections: gate, up (both hidden→intermediate), down (intermediate→hidden)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
        
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # gate_proj and up_proj: (..., hidden_size) → (..., intermediate_size)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        # Element-wise gating, then project back down
        return self.down_proj(gate * up)
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
