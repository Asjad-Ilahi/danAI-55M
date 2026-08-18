"""
Transformer block (decoder layer).

Pre-normalization architecture:
    RMSNorm → GQA Attention → Residual → RMSNorm → SwiGLU MLP → Residual

Supports gradient checkpointing for memory efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Optional, Tuple

from src.model.rmsnorm import RMSNorm
from src.model.attention import GroupedQueryAttention
from src.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Single decoder transformer block with pre-normalization.
    
    Architecture:
        h = x + Attention(RMSNorm(x))
        output = h + MLP(RMSNorm(h))
    
    Args:
        hidden_size: Model hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of KV heads (GQA)
        intermediate_size: SwiGLU intermediate dimension
        max_seq_len: Maximum sequence length
        rope_theta: RoPE base frequency
        rms_norm_eps: RMSNorm epsilon
        dropout: Residual dropout (default: 0.0)
        attention_dropout: Attention dropout (default: 0.0)
        bias: Whether to use bias in projections (default: False)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "ntk",
        rms_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        # Pre-attention norm
        self.attention_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # GQA attention
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            scaling_factor=scaling_factor,
            scaling_type=scaling_type,
            attention_dropout=attention_dropout,
            bias=bias,
        )
        
        # Pre-MLP norm
        self.mlp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        # SwiGLU MLP
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
        )
        
        # Residual dropout (default 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask for document-aware packing
            position_ids: Optional position indices
            kv_cache: Optional KV cache for generation
            use_cache: Whether to return updated KV cache
            use_checkpoint: Whether to use gradient checkpointing
        
        Returns:
            Tuple of (output, kv_cache)
        """
        if use_checkpoint and self.training and kv_cache is None:
            return self._forward_with_checkpoint(x, attention_mask, position_ids)
        
        # Pre-norm → Attention → Residual
        normed = self.attention_norm(x)
        attn_output, new_kv_cache = self.attention(
            normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        x = x + self.dropout(attn_output)
        
        # Pre-norm → MLP → Residual
        normed = self.mlp_norm(x)
        mlp_output = self.mlp(normed)
        x = x + self.dropout(mlp_output)
        
        return x, new_kv_cache
    
    def _forward_with_checkpoint(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass with gradient checkpointing (trades compute for memory)."""
        
        def _attn_block(x_in, mask, pos_ids):
            normed = self.attention_norm(x_in)
            attn_out, _ = self.attention(
                normed, attention_mask=mask, position_ids=pos_ids, kv_cache=None
            )
            return x_in + self.dropout(attn_out)
        
        def _mlp_block(x_in):
            normed = self.mlp_norm(x_in)
            return x_in + self.dropout(self.mlp(normed))
        
        # Checkpoint the attention block
        x = torch_checkpoint(
            _attn_block, x, attention_mask, position_ids,
            use_reentrant=False,
        )
        
        # Checkpoint the MLP block
        x = torch_checkpoint(
            _mlp_block, x,
            use_reentrant=False,
        )
        
        return x, None
