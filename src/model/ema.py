"""
Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model weights that is updated after each optimizer step:
    ema_weight = decay * ema_weight + (1 - decay) * model_weight

EMA reduces noise in small-batch/small-model training and generally produces
better eval numbers and cleaner generations than raw trained weights.

Per §32: EMA weights used for validation, best checkpoint, and generation.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional


class EMAModel:
    """
    Exponential Moving Average of model weights.
    
    Usage:
        ema = EMAModel(model, decay=0.999)
        # In training loop, after optimizer.step():
        ema.update(model)
        # For validation:
        ema.apply(model)  # Copy EMA weights to model
        ... validate ...
        ema.restore(model)  # Restore original weights
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.999). Higher = smoother, slower tracking.
               0 disables EMA.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.enabled = decay > 0
        
        if self.enabled:
            # Shadow copy of model parameters
            self.shadow = {}
            self.backup = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        else:
            self.shadow = {}
            self.backup = {}
    
    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA weights after an optimizer step.
        
        ema_weight = decay * ema_weight + (1 - decay) * model_weight
        """
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )
    
    def apply(self, model: nn.Module) -> None:
        """
        Apply EMA weights to the model (for validation/generation).
        
        Backs up current model weights so they can be restored later.
        """
        if not self.enabled:
            return
        
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module) -> None:
        """
        Restore original model weights after EMA was applied.
        """
        if not self.enabled:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            'decay': self.decay,
            'enabled': self.enabled,
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
        }
    
    def load_state_dict(self, state: dict, device: torch.device) -> None:
        """Load EMA state from a checkpoint."""
        self.decay = state['decay']
        self.enabled = state['enabled']
        self.shadow = {k: v.to(device) for k, v in state['shadow'].items()}
