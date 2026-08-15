"""
Structured logging for the SLM project.

Logs to both `logs/metrics.jsonl` (machine-readable) and `logs/training.log` (human-readable).
Optional TensorBoard support.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


class MetricsLogger:
    """
    Structured metrics logger that writes to JSONL and optionally TensorBoard.
    
    Each log entry is a JSON object on a single line in metrics.jsonl,
    containing step, tokens_seen, and any metrics passed.
    """
    
    def __init__(
        self,
        log_dir: str | Path = 'logs',
        experiment_dir: Optional[str | Path] = None,
        use_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # JSONL metrics file
        self.metrics_path = self.log_dir / 'metrics.jsonl'
        self.metrics_file = open(self.metrics_path, 'a')
        
        # Experiment-specific copy if provided
        self.exp_metrics_file = None
        if experiment_dir is not None:
            exp_dir = Path(experiment_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            self.exp_metrics_file = open(exp_dir / 'metrics.jsonl', 'a')
        
        # Human-readable log
        self.logger = self._setup_logger()
        
        # Optional TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / 'tensorboard'
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                self.logger.warning(
                    "TensorBoard requested but not installed. "
                    "Install with: pip install tensorboard"
                )
        
        self._start_time = time.time()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up Python logger for human-readable output."""
        logger = logging.getLogger('slm')
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_metrics(self, step: int, tokens_seen: int, **metrics: Any) -> None:
        """
        Log a set of metrics for a given step.
        
        Args:
            step: Global training step
            tokens_seen: Total tokens processed so far
            **metrics: Arbitrary key-value metric pairs
        """
        entry = {
            'step': step,
            'tokens_seen': tokens_seen,
            'elapsed_seconds': time.time() - self._start_time,
            **metrics,
        }
        
        # Write to JSONL
        line = json.dumps(entry)
        self.metrics_file.write(line + '\n')
        self.metrics_file.flush()
        
        if self.exp_metrics_file is not None:
            self.exp_metrics_file.write(line + '\n')
            self.exp_metrics_file.flush()
        
        # Write to TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, global_step=step)
    
    def log_training_step(
        self,
        step: int,
        tokens_seen: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_sec: float,
        memory_mb: Optional[float] = None,
    ) -> None:
        """Log a standard training step with formatted console output."""
        self.log_metrics(
            step=step,
            tokens_seen=tokens_seen,
            train_loss=loss,
            learning_rate=lr,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            memory_mb=memory_mb,
        )
        
        elapsed = time.time() - self._start_time
        elapsed_str = _format_time(elapsed)
        
        mem_str = f" | mem: {memory_mb:.0f}MB" if memory_mb is not None else ""
        self.logger.info(
            f"step {step:>7d} | tokens: {tokens_seen:>12,} | "
            f"loss: {loss:.4f} | lr: {lr:.2e} | "
            f"grad: {grad_norm:.4f} | tok/s: {tokens_per_sec:.0f}{mem_str} | "
            f"elapsed: {elapsed_str}"
        )
    
    def log_validation(
        self,
        step: int,
        tokens_seen: int,
        val_loss: float,
        perplexity: float,
        is_best: bool = False,
    ) -> None:
        """Log validation results."""
        self.log_metrics(
            step=step,
            tokens_seen=tokens_seen,
            val_loss=val_loss,
            val_perplexity=perplexity,
            is_best=is_best,
        )
        
        best_str = " ★ NEW BEST" if is_best else ""
        self.logger.info(
            f"{'=' * 60}\n"
            f"  VALIDATION @ step {step:,} | tokens: {tokens_seen:,}\n"
            f"  val_loss: {val_loss:.4f} | perplexity: {perplexity:.2f}{best_str}\n"
            f"{'=' * 60}"
        )
    
    def info(self, msg: str) -> None:
        """Log an informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg)
    
    def close(self) -> None:
        """Close all file handles."""
        self.metrics_file.close()
        if self.exp_metrics_file is not None:
            self.exp_metrics_file.close()
        if self.tb_writer is not None:
            self.tb_writer.close()


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    elif seconds < 86400:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m"
    else:
        d, remainder = divmod(seconds, 86400)
        h, _ = divmod(remainder, 3600)
        return f"{int(d)}d {int(h)}h"
