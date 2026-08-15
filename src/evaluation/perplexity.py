"""
Perplexity calculation functions per §35 & §43.
"""

import math
import torch


def compute_perplexity(loss: float) -> float:
    """Compute perplexity = exp(loss)."""
    try:
        return math.exp(min(20.0, loss))
    except OverflowError:
        return float("inf")
