"""
Reproducibility utilities for the SLM project.

Seeds Python, NumPy, and PyTorch for reproducible training.
Documents MPS non-determinism caveat.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    CAVEAT: MPS operations may still be non-deterministic even with seeds set.
    PyTorch's MPS backend does not guarantee deterministic behavior for all
    operations. This is a known limitation. For fully deterministic results,
    use CPU.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA-specific determinism (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These may slow down training but improve reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed for reproducible hashing
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_rng_state() -> dict:
    """
    Capture current RNG state for checkpointing.
    
    Returns:
        Dictionary containing RNG states for Python, NumPy, and PyTorch.
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    # Note: MPS does not expose RNG state for checkpointing as of PyTorch 2.x
    return state


def set_rng_state(state: dict) -> None:
    """
    Restore RNG state from a checkpoint.
    
    Args:
        state: Dictionary of RNG states as returned by get_rng_state()
    """
    if 'python' in state and state['python'] is not None:
        try:
            random.setstate(state['python'])
        except Exception:
            pass
    if 'numpy' in state and state['numpy'] is not None:
        try:
            np.random.set_state(state['numpy'])
        except Exception:
            pass
    if 'torch' in state and state['torch'] is not None:
        try:
            torch_state = state['torch']
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.cpu().to(torch.uint8)
            torch.random.set_rng_state(torch_state)
        except Exception:
            pass
    if 'cuda' in state and state['cuda'] is not None and torch.cuda.is_available():
        try:
            cuda_states = state['cuda']
            if isinstance(cuda_states, (list, tuple)):
                cuda_states = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in cuda_states]
            elif isinstance(cuda_states, torch.Tensor):
                cuda_states = [cuda_states.cpu().to(torch.uint8)]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass


