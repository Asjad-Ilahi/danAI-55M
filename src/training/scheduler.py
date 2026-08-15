"""
Learning rate scheduler per §13 & §28.

Supports both WSD (Warmup-Stable-Decay) and Cosine decay with linear warmup.
Step-based scheduling.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WSDOrCosineScheduler(_LRScheduler):
    """
    Step-based Learning Rate Scheduler.
    
    Modes:
    1. WSD (Warmup-Stable-Decay):
       - Linear Warmup (step < warmup_steps): 0 -> peak_lr
       - Stable Phase (warmup_steps <= step < max_steps - annealing_steps): peak_lr
       - Annealing Phase (step >= max_steps - annealing_steps): peak_lr -> min_lr
       
    2. Cosine:
       - Linear Warmup (step < warmup_steps): 0 -> peak_lr
       - Cosine Decay (step >= warmup_steps): peak_lr -> min_lr
    """

    def __init__(
        self,
        optimizer,
        max_steps: int,
        peak_lr: float = 3.0e-4,
        min_lr: float = 3.0e-5,
        warmup_steps: int = 100,
        annealing_steps: int = 500,
        schedule: str = "wsd",
        start_step: int = 0,
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        self.schedule = schedule.lower()
        self.start_step = start_step

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group.get("lr", peak_lr))

        super().__init__(optimizer, last_epoch)

    def get_lr_for_step(self, step: int) -> float:
        """Calculate learning rate for a specific step, relative to start_step."""
        # Calculate relative step from continuation start
        rel_step = max(0, step - self.start_step)
        rel_max_steps = max(1, self.max_steps - self.start_step)

        if rel_step < self.warmup_steps:
            # Linear warmup starting from min_lr up to peak_lr
            progress = float(rel_step + 1) / float(max(1, self.warmup_steps))
            return self.min_lr + (self.peak_lr - self.min_lr) * progress

        if self.schedule == "wsd":
            decay_start_step = rel_max_steps - self.annealing_steps
            if rel_step < decay_start_step:
                # Constant stable phase
                return self.peak_lr
            else:
                # Decay phase (linear decay to min_lr)
                decay_progress = float(rel_step - decay_start_step) / float(max(1, self.annealing_steps))
                decay_progress = min(1.0, max(0.0, decay_progress))
                return self.min_lr + (self.peak_lr - self.min_lr) * (1.0 - decay_progress)

        else:  # Cosine schedule
            if rel_step >= rel_max_steps:
                return self.min_lr
            progress = float(rel_step - self.warmup_steps) / float(max(1, rel_max_steps - self.warmup_steps))
            cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_coeff

    def get_lr(self):
        step = self.last_epoch
        current_lr = self.get_lr_for_step(step)
        return [current_lr for _ in self.optimizer.param_groups]
