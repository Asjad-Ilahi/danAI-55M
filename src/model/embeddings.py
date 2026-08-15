"""
Token embeddings for the SLM.

Simple embedding layer without positional embeddings (RoPE handles position).
Supports weight tying with the LM head.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.
    
    Converts token IDs to dense vectors. No positional encoding is applied here —
    position information is injected via RoPE in the attention layers.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Dimension of the embedding vectors
    """
    
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up token embeddings.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
        
        Returns:
            Embeddings of shape (batch, seq_len, hidden_size)
        """
        return self.embedding(input_ids)
    
    @property
    def weight(self) -> torch.Tensor:
        """Expose embedding weight for weight tying with LM head."""
        return self.embedding.weight
