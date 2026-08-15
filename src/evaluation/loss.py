"""
Validation loss evaluation helper per §35.

Computes loss on validation dataset, using EMA weights if available.
"""

import math
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Any


def evaluate_validation_loss(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    precision_mgr,
    batch_size: int = 1,
    max_eval_batches: int = 100,
    ema_model: Optional[Any] = None,
) -> Tuple[float, float]:
    """
    Compute average cross-entropy loss and perplexity on validation set.
    Samples evenly across all domains in the validation set.
    
    Returns: (val_loss, perplexity)
    """
    # Apply EMA weights if present
    if ema_model is not None:
        ema_model.apply(model)

    model.eval()
    num_samples = len(val_dataset)
    num_eval = min(max_eval_batches, num_samples)
    
    # Evenly spaced indices to cover all domains (Math, Science, Wiki, Code, QA, Stories)
    step_stride = max(1, num_samples // num_eval)
    eval_indices = [i * step_stride for i in range(num_eval)]

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for sample_idx in eval_indices:
            batch = val_dataset[sample_idx]

            # Convert single sample to batch dimension
            x = batch["x"].unsqueeze(0).to(device)
            y = batch["y"].unsqueeze(0).to(device)
            attn_mask = batch.get("attn_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0).to(device)

            with precision_mgr.get_autocast_context():
                _, loss, _ = model(x, attention_mask=attn_mask, targets=y)

            total_loss += loss.item()
            total_batches += 1

    model.train()

    # Restore original weights if EMA was applied
    if ema_model is not None:
        ema_model.restore(model)

    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(min(20.0, avg_loss))  # Cap exp for stability

    return avg_loss, perplexity

