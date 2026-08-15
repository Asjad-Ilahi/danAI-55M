"""
AdamW optimizer setup with parameter group splitting per §27.

Disables weight decay for 1D parameters (RMSNorm scales, biases) and token embeddings,
while applying weight decay (default 0.1) to 2D weight matrices (Linear projections).
"""

import torch
import torch.nn as nn
from typing import Tuple


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1.0e-8,
) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with decay and no-decay parameter groups.
    
    No weight decay is applied to:
    - 1D parameters (RMSNorm scales, biases)
    - Token embedding weights
    """
    decay_params = []
    no_decay_params = []
    seen_param_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)

        # 1D parameters (norms, biases) and embeddings get no weight decay
        if param.ndim < 2 or "token_embedding" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )

    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer setup: {len(decay_params)} decay tensors ({num_decay:,} params), "
          f"{len(no_decay_params)} no-decay tensors ({num_no_decay:,} params)")

    return optimizer
