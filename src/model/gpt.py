"""
Full decoder-only causal language model (GPT-style).

Architecture: Token Embedding → N × TransformerBlock → Final RMSNorm → LM Head (tied)

Weight initialization per §20:
- Embeddings: normal, std = 1/sqrt(hidden_size)
- Linear layers: normal, std = 0.02
- Residual output projections (attn o_proj, MLP down_proj): additional
  1/sqrt(2*num_layers) scaling to prevent residual stream variance growth with depth.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from src.model.embeddings import TokenEmbedding
from src.model.transformer_block import TransformerBlock
from src.model.rmsnorm import RMSNorm
from src.utils.config import Config, resolve_intermediate_size


class CausalLM(nn.Module):
    """
    Decoder-only causal language model.
    
    Combines token embeddings, a stack of transformer blocks with GQA,
    final layer norm, and a tied LM head for next-token prediction.
    
    Args:
        config: Model configuration (Config object or dict-like with model.* attributes)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        mc = config.model if hasattr(config, 'model') else config
        
        self.vocab_size = mc.vocab_size
        self.hidden_size = mc.hidden_size
        self.num_layers = mc.num_layers
        self.max_seq_len = mc.max_seq_len
        self.tie_embeddings = mc.get('tie_embeddings', True)
        self.use_checkpoint = False  # Set externally by trainer
        
        intermediate_size = resolve_intermediate_size(
            mc.hidden_size, mc.get('intermediate_size', 'auto')
        )
        
        # Token embedding (no positional — RoPE in attention)
        self.token_embedding = TokenEmbedding(mc.vocab_size, mc.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=mc.hidden_size,
                num_query_heads=mc.num_query_heads,
                num_kv_heads=mc.num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=mc.max_seq_len,
                rope_theta=mc.get('rope_theta', 10000.0),
                rms_norm_eps=mc.get('rms_norm_eps', 1e-5),
                dropout=mc.get('dropout', 0.0),
                attention_dropout=mc.get('attention_dropout', 0.0),
                bias=mc.get('use_bias', False),
            )
            for _ in range(mc.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(mc.hidden_size, eps=mc.get('rms_norm_eps', 1e-5))
        
        # LM head (output projection to vocab)
        self.lm_head = nn.Linear(mc.hidden_size, mc.vocab_size, bias=False)
        
        # Weight tying: share embedding weights with LM head
        if self.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights(mc)
    
    def _init_weights(self, mc) -> None:
        """
        Initialize all weights with explicit, documented initialization.
        
        Strategy (GPT-2/NanoGPT style):
        - Embeddings: normal(0, 1/sqrt(hidden_size))
        - Linear layers: normal(0, 0.02)
        - Residual output projections: additional 1/sqrt(2*num_layers) scaling
        - RMSNorm weights: initialized to 1 (done in RMSNorm __init__)
        - Biases (if any): initialized to 0
        
        The residual scaling prevents the variance of the residual stream from
        growing proportionally with depth. This is especially important when
        maximizing depth (20 layers) as we do here.
        """
        use_residual_scale = mc.get('residual_scale_init', True)
        residual_scale = 1.0 / math.sqrt(2 * self.num_layers) if use_residual_scale else 1.0
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Apply residual scaling to output projections
                if any(proj_name in name for proj_name in ['o_proj', 'down_proj']):
                    std *= residual_scale
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden_size))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask for document-aware packing
            position_ids: Optional position indices
            targets: Optional target token IDs for loss computation (batch, seq_len)
            kv_caches: Optional list of KV caches per layer (for generation)
            use_cache: If True, return new_kv_caches even if input kv_caches was None
        
        Returns:
            Tuple of (logits, loss, new_kv_caches):
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar if targets provided, else None
                - new_kv_caches: list of (K, V) per layer if kv_caches or use_cache was True
        """
        # Token embeddings
        h = self.token_embedding(input_ids)  # (B, S, H)
        
        # Pass through transformer blocks
        new_kv_caches = [] if (kv_caches is not None or use_cache) else None
        
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            h, new_cache = layer(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=layer_kv_cache,
                use_cache=use_cache,
                use_checkpoint=self.use_checkpoint,
            )
            if new_kv_caches is not None:
                new_kv_caches.append(new_cache)
        
        # Final norm
        h = self.final_norm(h)
        
        # LM head (logits)
        logits = self.lm_head(h)  # (B, S, V)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Check if targets are already aligned with logits (e.g. from ShardDataset where x=chunk[:-1], y=chunk[1:])
            if targets.shape[1] == logits.shape[1]:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )
            else:
                # Shift: logits[:, :-1] predicts targets[:, 1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )
        
        return logits, loss, new_kv_caches
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Count total or non-embedding parameters.
        
        Args:
            non_embedding: If True, exclude token embedding parameters
        
        Returns:
            Parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    def estimate_memory_mb(self, dtype: torch.dtype = torch.float32) -> dict:
        """Estimate memory usage for model parameters and optimizer."""
        n_params = self.get_num_params()
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(dtype, 4)
        
        model_mb = n_params * bytes_per_param / (1024 ** 2)
        # AdamW: 2 moment buffers in fp32 (always)
        optimizer_mb = n_params * 4 * 2 / (1024 ** 2)
        # Gradients in same dtype as model
        gradient_mb = n_params * bytes_per_param / (1024 ** 2)
        
        return {
            'model_mb': model_mb,
            'optimizer_mb': optimizer_mb,
            'gradient_mb': gradient_mb,
            'total_mb': model_mb + optimizer_mb + gradient_mb,
            'dtype': str(dtype),
            'num_params': n_params,
        }
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all transformer blocks."""
        self.use_checkpoint = True
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.use_checkpoint = False
