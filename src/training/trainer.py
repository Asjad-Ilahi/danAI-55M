"""
Main training loop orchestrator per §13, §23-34.

Handles:
- Gradient accumulation & clipping
- Mixed precision autocast & loss checking
- Step-based learning rate schedule (WSD or Cosine)
- EMA weight updates
- Validation and checkpointing
- Tokens/sec throughput measurement
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.model.gpt import CausalLM
from src.model.ema import EMAModel
from src.training.optimizer import create_optimizer
from src.training.scheduler import WSDOrCosineScheduler
from src.training.precision import PrecisionManager
from src.training.checkpoint import CheckpointManager
from src.training.memory import print_memory_report, get_memory_info
from src.utils.config import Config, get_effective_batch_size, validate_config
from src.utils.logging import MetricsLogger
from src.utils.seed import set_seed
from src.evaluation.loss import evaluate_validation_loss


class Trainer:
    """Orchestrates model pretraining."""

    def __init__(
        self,
        config: Config,
        train_dataset,
        val_dataset=None,
        experiment_dir: Optional[str | Path] = None,
        resume_checkpoint: Optional[str | Path] = None,
    ):
        validate_config(config)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.experiment_dir = Path(experiment_dir) if experiment_dir else Path("experiments/exp_000")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        set_seed(config.training.seed)

        # Device & Precision setup
        from src.utils.device import get_device, select_precision, print_device_info
        self.device = get_device()
        self.precision = select_precision(self.device, config.training.precision)
        print_device_info(self.device, self.precision)

        # Model instantiation
        self.model = CausalLM(config).to(self.device)
        total_params = self.model.get_num_params()
        print_memory_report(self.device, total_params)

        if config.training.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            print("Gradient checkpointing: ENABLED")

        # EMA Setup (§32)
        ema_decay = config.training.get("ema_decay", 0.999)
        self.ema_model = EMAModel(self.model, decay=ema_decay) if ema_decay > 0 else None

        # Effective batch calculation (§25)
        batch_info = get_effective_batch_size(config)
        self.micro_batch_size = batch_info["micro_batch_size"]
        self.grad_accum_steps = batch_info["gradient_accumulation_steps"]
        self.seq_len = batch_info["max_seq_len"]
        self.effective_batch_tokens = batch_info["effective_batch_tokens"]

        print(f"Batch config: micro={self.micro_batch_size}, accum={self.grad_accum_steps}, "
              f"seq_len={self.seq_len} → Effective Batch = {self.effective_batch_tokens:,} tokens/step")

        # Data Loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=config.training.get("num_workers", 0),
            pin_memory=False,
        )

        # Total steps calculation based on max_tokens budget
        max_tokens = config.training.max_tokens
        self.max_steps = max_tokens // self.effective_batch_tokens
        warmup_steps = int(self.max_steps * config.training.warmup_ratio)
        annealing_steps = int(self.max_steps * config.training.annealing_fraction)

        print(f"Token budget: {max_tokens:,} tokens → {self.max_steps:,} total steps "
              f"(warmup: {warmup_steps}, annealing: {annealing_steps})")

        # Optimizer & Scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta1=config.training.beta1,
            beta2=config.training.beta2,
            eps=config.training.eps,
        )

        self.scheduler = WSDOrCosineScheduler(
            self.optimizer,
            max_steps=self.max_steps,
            peak_lr=config.training.learning_rate,
            min_lr=config.training.min_learning_rate,
            warmup_steps=warmup_steps,
            annealing_steps=annealing_steps,
            schedule=config.training.schedule,
        )

        # Precision Manager & Checkpoint Manager
        self.precision_mgr = PrecisionManager(self.device, self.precision)
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir=self.experiment_dir / "checkpoints")
        self.logger = MetricsLogger(log_dir=self.experiment_dir / "logs", experiment_dir=self.experiment_dir)

        # Training state
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        # Resume if specified
        if resume_checkpoint:
            is_continuation = config.training.get("continuation", False)
            reset_optimizer = config.training.get("reset_optimizer", False)
            reset_scheduler = is_continuation or config.training.get("reset_scheduler", False)

            saved_state = self.checkpoint_mgr.load_checkpoint(
                resume_checkpoint,
                self.model,
                ema_model=self.ema_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                reset_optimizer=reset_optimizer,
                reset_scheduler=reset_scheduler,
            )
            self.step = saved_state["step"]
            self.tokens_seen = saved_state["tokens_seen"]
            self.best_val_loss = saved_state["best_val_loss"]

            if is_continuation:
                cont_warmup = config.training.get("continuation_warmup_steps", 50)
                self.scheduler.start_step = self.step
                self.scheduler.warmup_steps = cont_warmup
                self.scheduler.peak_lr = config.training.learning_rate
                self.scheduler.min_lr = config.training.min_learning_rate
                self.scheduler.max_steps = self.max_steps
                self.scheduler.last_epoch = self.step - 1
                print(f"Continuation mode active: start_step={self.step}, peak_lr={self.scheduler.peak_lr}, warmup_steps={cont_warmup}")
            else:
                # Ensure scheduler max_steps and annealing_steps reflect updated config max_tokens
                self.scheduler.max_steps = self.max_steps
                self.scheduler.annealing_steps = int(self.max_steps * config.training.annealing_fraction)
                self.scheduler.last_epoch = self.step - 1

            print(f"Resumed from step {self.step:,} (tokens_seen: {self.tokens_seen:,}) | Scheduler updated to max_steps={self.max_steps:,}")

    def train(self) -> None:
        """Execute pretraining loop."""
        self.logger.info("Starting LM Pretraining...")
        self.model.train()

        start_time = time.time()
        step_tokens_accum = 0
        accum_loss = 0.0

        loader_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_start_time = time.time()
            accum_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                attn_mask = batch.get("attn_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)

                # Forward pass under autocast
                with self.precision_mgr.get_autocast_context():
                    logits, loss, _ = self.model(x, attention_mask=attn_mask, targets=y)
                    scaled_loss = loss / self.grad_accum_steps

                # Backward pass
                scaled_loss.backward()
                accum_loss += loss.item() / self.grad_accum_steps

            # Check loss for NaN/Inf
            loss_tensor = torch.tensor(accum_loss)
            self.precision_mgr.check_loss_stability(loss_tensor, self.step)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.training.grad_clip
            )
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()

            # Optimizer step & LR step
            self.optimizer.step()
            self.scheduler.step()

            # EMA update
            if self.ema_model:
                self.ema_model.update(self.model)

            self.step += 1
            self.tokens_seen += self.effective_batch_tokens

            step_duration = time.time() - step_start_time
            tokens_per_sec = self.effective_batch_tokens / max(1e-5, step_duration)
            mem_info = get_memory_info(self.device)

            # Log step metrics
            if self.step % self.config.training.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_training_step(
                    step=self.step,
                    tokens_seen=self.tokens_seen,
                    loss=accum_loss,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    tokens_per_sec=tokens_per_sec,
                    memory_mb=mem_info["process_rss_mb"],
                )

            # Validation
            if self.val_dataset is not None and (self.step % self.config.training.validation_interval == 0 or self.step == self.max_steps):
                val_loss, perplexity = evaluate_validation_loss(
                    self.model,
                    self.val_dataset,
                    self.device,
                    self.precision_mgr,
                    max_eval_batches=getattr(self.config.training, "max_eval_batches", 100),
                    ema_model=self.ema_model,
                    step=self.step,
                )
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                self.logger.log_validation(self.step, self.tokens_seen, val_loss, perplexity, is_best=is_best)

                if is_best:
                    self.checkpoint_mgr.save_checkpoint(
                        "best.pt",
                        self.model,
                        self.ema_model,
                        self.optimizer,
                        self.scheduler,
                        self.step,
                        self.tokens_seen,
                        self.best_val_loss,
                        self.config.to_dict(),
                    )

            # Checkpoint
            if self.step % self.config.training.checkpoint_interval == 0 or self.step == self.max_steps:
                self.checkpoint_mgr.save_checkpoint(
                    f"checkpoint_step_{self.step:07d}.pt",
                    self.model,
                    self.ema_model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    self.tokens_seen,
                    self.best_val_loss,
                    self.config.to_dict(),
                )
                self.checkpoint_mgr.save_checkpoint(
                    "latest.pt",
                    self.model,
                    self.ema_model,
                    self.optimizer,
                    self.scheduler,
                    self.step,
                    self.tokens_seen,
                    self.best_val_loss,
                    self.config.to_dict(),
                )

        self.logger.info("Training complete!")
