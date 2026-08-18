"""
Grouped-Query Attention (GQA) with document-boundary masking.

Implements multi-head causal self-attention with grouped KV heads per §3 and §18.
Supports document-aware attention masking per §6 to prevent cross-document attention
in packed sequences.

Uses torch.nn.functional.scaled_dot_product_attention when available,
with a safe manual fallback.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    
    GQA uses fewer KV heads than query heads, sharing each KV head across
    multiple query heads. This reduces KV projection parameters, freeing
    budget for additional transformer layers (depth > width at small scale).
    
    At 75M parameters, GQA's primary benefit is parameter efficiency for
    depth, not inference KV-cache memory savings.
    
    Supports:
    - Grouped KV heads (num_query_heads must be divisible by num_kv_heads)
    - Causal masking (always on for decoder-only models)
    - Document-boundary masking (block-diagonal causal mask within packed sequences)
    - RoPE integration
    - scaled_dot_product_attention with manual fallback
    
    Args:
        hidden_size: Model hidden dimension
        num_query_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_query_heads)
        max_seq_len: Maximum sequence length
        rope_theta: RoPE base frequency
        attention_dropout: Dropout rate for attention weights (default: 0.0)
        bias: Whether to use bias in projections (default: False)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "ntk",
        attention_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        assert hidden_size % num_query_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_query_heads ({num_query_heads})"
        )
        assert num_query_heads % num_kv_heads == 0, (
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
        
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_query_heads
        self.num_kv_groups = num_query_heads // num_kv_heads  # Queries per KV head
        self.attention_dropout = attention_dropout
        
        # Q projection: hidden_size → hidden_size (num_query_heads * head_dim)
        self.q_proj = nn.Linear(hidden_size, num_query_heads * self.head_dim, bias=bias)
        # K projection: hidden_size → num_kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        # V projection: hidden_size → num_kv_heads * head_dim
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        # Output projection: hidden_size → hidden_size
        self.o_proj = nn.Linear(num_query_heads * self.head_dim, hidden_size, bias=bias)
        
        # RoPE with NTK-Aware context scaling
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
            scaling_factor=scaling_factor,
            scaling_type=scaling_type,
        )
        
        # Check if SDPA is available
        self._has_sdpa = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for GQA.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask of shape (batch, 1, seq_len, seq_len)
                           or (batch, num_query_heads, seq_len, seq_len).
                           True/1 = attend, False/0 = mask out.
                           If None, uses standard causal mask.
                           For document-aware packing, this should be a block-diagonal
                           causal mask from packing.py.
            position_ids: Optional position indices of shape (batch, seq_len)
            kv_cache: Optional tuple of (cached_keys, cached_values) for generation
            use_cache: Whether to return updated KV cache
        
        Returns:
            Tuple of (output, new_kv_cache):
                - output: (batch, seq_len, hidden_size)
                - new_kv_cache: Updated (keys, values) if kv_cache or use_cache was True, else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # (B, S, num_q_heads * head_dim)
        k = self.k_proj(x)  # (B, S, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, S, num_kv_heads * head_dim)
        
        # Reshape to (B, num_heads, S, head_dim)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
        if position_ids is None and past_len > 0:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Apply RoPE to Q and K
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)
        
        # Handle KV cache for generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        
        new_kv_cache = (k, v) if (kv_cache is not None or use_cache) else None
        
        # Expand KV heads to match query heads for GQA
        # (B, num_kv_heads, S, head_dim) → (B, num_query_heads, S, head_dim)
        if self.num_kv_groups > 1:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
        
        # Compute attention
        if self._has_sdpa and attention_mask is None:
            dropout_p = self.attention_dropout if self.training else 0.0
            # When using KV cache in step-by-step decoding (seq_len=1), is_causal must be False since key length > query length
            is_causal = (kv_cache is None) or (seq_len > 1)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        elif self._has_sdpa and attention_mask is not None:
            # SDPA with explicit mask (for document-boundary masking)
            dropout_p = self.attention_dropout if self.training else 0.0
            # Convert boolean mask to float mask for SDPA
            # SDPA expects: 0 = attend, -inf = mask
            if attention_mask.dtype == torch.bool:
                sdpa_mask = torch.zeros_like(attention_mask, dtype=q.dtype)
                sdpa_mask.masked_fill_(~attention_mask, float('-inf'))
            else:
                sdpa_mask = attention_mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=False,  # We provide the full mask
            )
        else:
            # Manual fallback
            attn_output = self._manual_attention(q, k, v, attention_mask)
        
        # Reshape back: (B, num_q_heads, S, head_dim) → (B, S, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, new_kv_cache
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.
        
        (B, num_kv_heads, S, head_dim) → (B, num_query_heads, S, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        # (B, num_kv_heads, 1, S, head_dim) → (B, num_kv_heads, groups, S, head_dim)
        x = x.unsqueeze(2).expand(
            batch_size, num_kv_heads, self.num_kv_groups, seq_len, head_dim
        )
        # (B, num_query_heads, S, head_dim)
        return x.reshape(batch_size, self.num_query_heads, seq_len, head_dim)
    
    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Manual scaled dot-product attention (fallback when SDPA is unavailable).
        
        Implements: softmax(QK^T / sqrt(d) + mask) V
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # (B, H, S, D) @ (B, H, D, S) → (B, H, S, S)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        kv_seq_len = k.shape[2]
        q_seq_len = q.shape[2]
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(q_seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - q_seq_len + 1,
        )
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional attention mask (e.g., document boundary mask)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights.masked_fill_(~attention_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        
        # (B, H, S, S) @ (B, H, S, D) → (B, H, S, D)
        return torch.matmul(attn_weights, v)
    
    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_query_heads={self.num_query_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"num_kv_groups={self.num_kv_groups}"
        )
