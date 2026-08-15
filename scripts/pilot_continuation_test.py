"""
Pilot Benchmark Script for SLM Continuation Training (§44).

Evaluates baseline checkpoint experiments/exp_005/checkpoints/best.pt on validation sets,
and runs a 40-step continuation test across candidate continuation learning rates
(3.0e-5, 5.0e-5, 8.0e-5, 1.0e-4) to empirically verify non-regressive fine-tuning.
"""

import copy
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.utils.seed import set_seed
from src.model.gpt import CausalLM
from src.model.ema import EMAModel
from src.data.shard_dataset import ShardDataset
from src.training.optimizer import create_optimizer
from src.training.scheduler import WSDOrCosineScheduler
from src.training.precision import PrecisionManager
from src.training.checkpoint import CheckpointManager
from src.evaluation.loss import evaluate_validation_loss


def run_pilot_experiment(candidate_lr: float, num_steps: int = 40) -> dict:
    """Run a pilot continuation experiment for a given candidate LR."""
    print(f"\n" + "=" * 70)
    print(f"  STARTING PILOT RUN: Candidate Continuation LR = {candidate_lr:.2e}")
    print("=" * 70)

    set_seed(42)
    device = get_device()
    precision = select_precision(device, "auto")
    precision_mgr = PrecisionManager(device, precision)

    ckpt_path = Path("experiments/exp_005/checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at {ckpt_path}")

    # Load baseline state
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = Config(checkpoint.get("config", {}))

    # Override training params for pilot
    config.training.learning_rate = candidate_lr
    config.training.min_learning_rate = 1.0e-5
    config.training.continuation = True
    config.training.continuation_warmup_steps = 20

    # Load model
    model = CausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load EMA model if present
    ema_decay = config.training.get("ema_decay", 0.999)
    ema_model = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None
    if ema_model and checkpoint.get("ema_state_dict"):
        ema_model.load_state_dict(checkpoint["ema_state_dict"], device=device)

    start_step = checkpoint.get("step", 2349)
    tokens_seen = checkpoint.get("tokens_seen", 76972032)
    target_max_steps = start_step + num_steps

    # Combined training dataset (70% new domain + 30% baseline replay)
    train_dirs = [Path("data/shards/train"), Path("data/archive_30m/shards/train")]
    train_dataset = ShardDataset(train_dirs, seq_len=config.model.max_seq_len, pack_with_document_mask=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Validation datasets
    val_baseline_ds = ShardDataset(Path("data/archive_30m/shards/val"), seq_len=config.model.max_seq_len, pack_with_document_mask=True)
    val_new_ds = ShardDataset(Path("data/shards/val"), seq_len=config.model.max_seq_len, pack_with_document_mask=True)

    # Setup optimizer and scheduler
    optimizer = create_optimizer(
        model,
        learning_rate=candidate_lr,
        weight_decay=config.training.get("weight_decay", 0.1),
    )

    scheduler = WSDOrCosineScheduler(
        optimizer,
        max_steps=target_max_steps + 100,
        peak_lr=candidate_lr,
        min_lr=1.0e-5,
        warmup_steps=20,
        schedule="wsd",
        start_step=start_step,
        last_epoch=start_step - 1,
    )

    # Pre-eval baseline before step 1
    model.eval()
    init_val_baseline, init_ppl_baseline = evaluate_validation_loss(model, val_baseline_ds, device, precision_mgr, ema_model=ema_model)
    init_val_new, init_ppl_new = evaluate_validation_loss(model, val_new_ds, device, precision_mgr, ema_model=ema_model)
    print(f"  [Step {start_step}] Pre-training Baseline Val Loss -> Original (30M): {init_val_baseline:.4f} | New (60M): {init_val_new:.4f}")

    # Training loop
    model.train()
    loader_iter = iter(train_loader)
    step = start_step
    grad_accum_steps = 8
    
    first_train_loss = None
    last_train_loss = None

    step_history = []

    while step < target_max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            x = batch["x"].to(device)
            y = batch["y"].to(device)
            attn_mask = batch.get("attn_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            with precision_mgr.get_autocast_context():
                logits, loss, _ = model(x, attention_mask=attn_mask, targets=y)
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            accum_loss += loss.item() / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ema_model:
            ema_model.update(model)

        step += 1
        current_lr = optimizer.param_groups[0]["lr"]

        if first_train_loss is None:
            first_train_loss = accum_loss
        last_train_loss = accum_loss

        print(f"  Step {step}/{target_max_steps} | Train Loss: {accum_loss:.4f} | LR: {current_lr:.2e}")

        # Mid & end evaluations
        if (step - start_step) % 20 == 0 or step == target_max_steps:
            model.eval()
            val_base_loss, val_base_ppl = evaluate_validation_loss(model, val_baseline_ds, device, precision_mgr, ema_model=ema_model)
            val_new_loss, val_new_ppl = evaluate_validation_loss(model, val_new_ds, device, precision_mgr, ema_model=ema_model)
            print(f"  >>> Checkpoint Step {step} Eval -> Baseline Val Loss: {val_base_loss:.4f} | New Val Loss: {val_new_loss:.4f}")
            step_history.append({
                "step": step,
                "train_loss": accum_loss,
                "val_baseline_loss": val_base_loss,
                "val_new_loss": val_new_loss,
            })
            model.train()

    final_eval = step_history[-1]
    regressed = final_eval["val_baseline_loss"] > (init_val_baseline + 0.15)

    return {
        "candidate_lr": candidate_lr,
        "start_train_loss": first_train_loss,
        "final_train_loss": last_train_loss,
        "init_val_baseline": init_val_baseline,
        "final_val_baseline": final_eval["val_baseline_loss"],
        "init_val_new": init_val_new,
        "final_val_new": final_eval["val_new_loss"],
        "regressed": regressed,
    }


def main():
    print("=" * 80)
    print("  SLM CONTINUATION TRAINING PILOT BENCHMARK")
    print("=" * 80)

    candidate_lrs = [3.0e-5, 5.0e-5, 8.0e-5, 1.0e-4]
    results = []

    for lr in candidate_lrs:
        res = run_pilot_experiment(candidate_lr=lr, num_steps=40)
        results.append(res)

    print("\n" + "=" * 85)
    print("  PILOT BENCHMARK SUMMARY REPORT")
    print("=" * 85)
    header = f"{'LR':<10} | {'Start Train':<11} | {'Final Train':<11} | {'Base Val (3.77)':<15} | {'New Val (5.13)':<14} | {'Status':<10}"
    print(header)
    print("-" * 85)

    for r in results:
        status = "REGRESSED" if r["regressed"] else "SAFE / OK"
        line = (f"{r['candidate_lr']:<10.2e} | "
                f"{r['start_train_loss']:<11.4f} | "
                f"{r['final_train_loss']:<11.4f} | "
                f"{r['init_val_baseline']:.4f} -> {r['final_val_baseline']:<6.4f} | "
                f"{r['init_val_new']:.4f} -> {r['final_val_new']:<5.4f} | "
                f"{status:<10}")
        print(line)

    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
