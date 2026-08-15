"""
Mandatory tiny overfitting test script per §40.

The model MUST overfit a tiny repeated corpus to near-zero loss before any real
training is attempted. If it cannot overfit, training pipeline logic or attention/RoPE is broken.
"""

import sys
import torch
import torch.nn as nn

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionManager


def run_overfit_test():
    print("=" * 60)
    print("MANDATORY OVERFITTING TEST GATE (§40)")
    print("=" * 60)

    device = get_device()
    precision = select_precision(device, "auto")
    precision_mgr = PrecisionManager(device, precision)

    print(f"Device: {device}, Precision: {precision}")

    # Use debug / tiny model for fast overfit gate
    config = Config.from_yaml("configs/debug.yaml")
    model = CausalLM(config).to(device)
    model.train()

    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)

    # Create fixed repeated token sequence (batch=1, seq_len=128)
    seq_len = config.model.max_seq_len
    vocab_size = config.model.vocab_size

    torch.manual_seed(42)
    # Repeated 16-token pattern over seq_len
    pattern = torch.randint(10, vocab_size - 10, (16,), device=device)
    full_seq = pattern.repeat(seq_len // 16 + 1)[: seq_len + 1]

    x = full_seq[:-1].unsqueeze(0).to(device)
    y = full_seq[1:].unsqueeze(0).to(device)

    print(f"Training model ({model.get_num_params():,} params) to overfit 128-token sequence...")

    max_steps = 150
    target_loss = 0.05
    achieved = False

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        with precision_mgr.get_autocast_context():
            logits, loss, _ = model(x, targets=y)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if step % 20 == 0 or step == 1 or loss_val < target_loss:
            print(f"  Step {step:>3d}/{max_steps} | Loss: {loss_val:.6f}")

        if loss_val < target_loss:
            achieved = True
            print(f"\n✓ SUCCESS: Overfit gate passed at step {step}! Loss = {loss_val:.6f} (< {target_loss})")
            break

    print("=" * 60 + "\n")

    if not achieved:
        print(f"\n❌ FATAL: Overfit test FAILED! Model failed to achieve loss < {target_loss} in {max_steps} steps.")
        print("Do NOT proceed to full training until this issue is resolved.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_overfit_test()
