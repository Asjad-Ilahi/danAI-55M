"""
RMSNorm (Root Mean Square Layer Normalization).

Implements RMSNorm as x / RMS(x) * scale, where RMS(x) = sqrt(mean(x^2) + eps).
More efficient than LayerNorm (no mean subtraction or learned bias), commonly used
in modern LLMs (LLaMA, Gemma, etc.).

Implemented from scratch per §16.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm normalizes the input by dividing by the root mean square,
    then scales by a learned parameter. Unlike LayerNorm, it does not
    subtract the mean or learn a bias term.
    
    Formula: output = (x / sqrt(mean(x^2) + eps)) * scale
    
    Args:
        hidden_size: Dimension of the input features
        eps: Small constant for numerical stability (default: 1e-5)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # Learned scale parameter, initialized to ones
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization without the learned scale."""
        # x shape: (..., hidden_size)
        # Compute mean of squares along the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
        
        Returns:
            Normalized tensor of same shape, scaled by learned weight
        """
        # Cast to float32 for normalization stability, then back to input dtype
        output = self._norm(x.float()).to(x.dtype)
        return output * self.weight.to(x.dtype)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
