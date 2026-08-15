"""
Checkpoint saving and resuming module per §33 & §34.

Saves full training state (model, EMA, optimizer, scheduler, step, tokens_seen, config, RNG state).
Supports weights-only export mode for inference deployment.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from src.utils.seed import get_rng_state, set_rng_state


class CheckpointManager:
    """Manages saving and resuming training checkpoints."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        filename: str,
        model: torch.nn.Module,
        ema_model: Optional[Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        tokens_seen: int,
        best_val_loss: float,
        config: Dict[str, Any],
    ) -> Path:
        """Save full training state to file."""
        checkpoint_path = self.checkpoint_dir / filename

        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "config": config,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_state": get_rng_state(),
        }

        # Save atomically via unique temporary file
        import time
        tmp_path = self.checkpoint_dir / f".tmp_{filename}_{os.getpid()}_{time.time_ns()}"
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except Exception as err:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise err

        print(f"Checkpoint saved: {checkpoint_path} (step {step:,}, tokens {tokens_seen:,})")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        ema_model: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
        reset_optimizer: bool = False,
        reset_scheduler: bool = False,
    ) -> Dict[str, Any]:
        """Restore training state from checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load EMA state if present
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"], device=device)

        # Load optimizer unless reset_optimizer is True
        if not reset_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  Loaded optimizer state dict from checkpoint.")
        elif reset_optimizer:
            print("  Resetting optimizer state for continuation training.")

        # Load scheduler unless reset_scheduler is True
        if not reset_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("  Loaded scheduler state dict from checkpoint.")
        elif reset_scheduler:
            print("  Resetting scheduler for continuation training.")

        # Restore RNG state
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            set_rng_state(checkpoint["rng_state"])

        return {
            "step": checkpoint.get("step", 0),
            "tokens_seen": checkpoint.get("tokens_seen", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "config": checkpoint.get("config", {}),
        }

    def export_weights_only(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        use_ema: bool = True,
    ) -> Path:
        """Export lightweight weights-only checkpoint for inference/deployment."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if use_ema and ckpt.get("ema_state_dict") is not None and "shadow" in ckpt["ema_state_dict"]:
            weights = ckpt["ema_state_dict"]["shadow"]
        else:
            weights = ckpt["model_state_dict"]

        torch.save({"model_state_dict": weights, "config": ckpt.get("config")}, out_p)
        print(f"Exported weights-only model to {out_p}")
        return out_p
