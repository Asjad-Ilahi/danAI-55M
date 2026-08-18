"""
Supervised Fine-Tuning (SFT) Training Script for 54.5M SLM.

Features:
- Multi-domain validation evaluating ALL 1,570 validation samples (no first-N truncation).
- Comprehensive per-domain loss & perplexity reporting:
  * Conversation (SmolTalk)
  * General Q&A (Everyday + OpenOrca)
  * Mathematics (Verified Arithmetic + NuminaMath)
  * Coding (CodeFeedback)
  * Reasoning (Tulu 3 English)
- Strict prompt loss masking: System/User tokens have target ID = -100.
- Intelligent multi-turn sequence truncation preserving the latest complete assistant response.
- Comprehensive progress tracking: epoch, tokens processed, examples processed, optimizer steps.
- Live qualitative test generation during validation checks.
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import yaml
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.data.sft_dataset import SFTDataset, sft_collate_fn

from src.evaluation.generation import TextGenerator
from src.model.gpt import CausalLM
from src.model.ema import EMAModel
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionManager
from src.utils.config import Config
from src.utils.device import get_device


# Standard qualitative benchmark prompts
EVAL_PROMPTS = [
    "User: Hello! How are you?\n\nAssistant:",
    "User: What is the capital of Pakistan?\n\nAssistant:",
    "User: What is photosynthesis?\n\nAssistant:",
    "User: I have 4 mangoes and I give away 2. How many mangoes do I have left?\n\nAssistant:",
    "System: You are an intelligent AI assistant. You have access to the following tools:\n- calculator(expression: str): Evaluates mathematical and arithmetic expressions with exact precision.\n- search_web(query: str): Searches the web for recent, real-time, or external information.\n- run_python(code: str): Executes Python code in a secure sandbox and returns the stdout output.\nAlways think carefully step-by-step inside a <think> block first. When a tool is needed, respond with a <tool_call> block containing a JSON object with 'name' and 'arguments'.\n\nUser: What is 48291 * 7182?\n\nAssistant:",
    "System: You are an intelligent AI assistant. You have access to the following tools:\n- calculator(expression: str): Evaluates mathematical and arithmetic expressions with exact precision.\n- search_web(query: str): Searches the web for recent, real-time, or external information.\n- run_python(code: str): Executes Python code in a secure sandbox and returns the stdout output.\nAlways think carefully step-by-step inside a <think> block first. When a tool is needed, respond with a <tool_call> block containing a JSON object with 'name' and 'arguments'.\n\nUser: Search the web for NASA's Artemis II mission.\n\nAssistant:",
    "User: Write a Python function `reverse_string(s)`.\n\nAssistant:",
]


def map_domain_name(raw_domain: str) -> str:
    """Normalize raw dataset domain keys to human-readable categories."""
    d = raw_domain.lower()
    if "calculator" in d:
        return "Agentic: Calculator"
    elif "web search" in d or "search" in d:
        return "Agentic: Web Search"
    elif "python" in d or "sandbox" in d:
        return "Agentic: Python Sandbox"
    elif "error" in d or "recovery" in d:
        return "Agentic: Error Recovery"
    elif "prompt engineering" in d or "system" in d:
        return "Prompt Engineering (System Prompts)"
    elif "rag" in d or "context" in d:
        return "RAG (Context-Grounded QA)"
    elif "logic" in d or "reason" in d:
        return "Logic & Reasoning"
    elif "math" in d or "arithmetic" in d:
        return "Mathematics"
    elif "coding" in d or "code" in d:
        return "Coding"
    elif "conversation" in d or "smol" in d:
        return "Conversation"
    else:
        return "General Knowledge & Instructions"



def evaluate_full_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    precision_mgr: PrecisionManager,
    device: torch.device,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    Evaluate the ENTIRE validation dataset (all samples) and compute
    both overall and per-domain loss and perplexity.
    """
    model.eval()
    domain_losses = defaultdict(float)
    domain_token_counts = defaultdict(int)
    domain_sample_counts = defaultdict(int)

    total_loss_sum = 0.0
    total_tokens = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            domains = batch["domains"]

            # Per-sample evaluation for accurate domain breakdown
            for i in range(x.size(0)):
                xi = x[i : i + 1]
                yi = y[i : i + 1]
                dom_key = map_domain_name(domains[i])

                with precision_mgr.get_autocast_context():
                    _, loss, _ = model(xi, targets=yi)

                asst_tokens = (yi != -100).sum().item()
                if asst_tokens > 0 and loss is not None:
                    loss_val = loss.item()
                    domain_losses[dom_key] += loss_val * asst_tokens
                    domain_token_counts[dom_key] += asst_tokens
                    domain_sample_counts[dom_key] += 1

                    total_loss_sum += loss_val * asst_tokens
                    total_tokens += asst_tokens
                    total_samples += 1

    overall_avg_loss = total_loss_sum / max(1, total_tokens)
    overall_ppl = math.exp(min(20.0, overall_avg_loss))

    domain_metrics = {}
    for dom, token_loss in domain_losses.items():
        toks = domain_token_counts[dom]
        samps = domain_sample_counts[dom]
        avg_d_loss = token_loss / max(1, toks)
        d_ppl = math.exp(min(20.0, avg_d_loss))
        domain_metrics[dom] = {
            "loss": avg_d_loss,
            "perplexity": d_ppl,
            "samples": samps,
            "tokens": toks,
        }

    return overall_avg_loss, overall_ppl, domain_metrics


def train_sft(
    model_config_path: str = "configs/model.yaml",
    sft_config_path: str = "configs/train_sft.yaml",
    train_data_path: str = "data_sft/sft_train.jsonl",
    val_data_path: str = "data_sft/sft_val.jsonl",
    tokenizer_path: str = "tokenizer/tokenizer.json",
    base_checkpoint_path: str = "experiments/exp_008/checkpoints/best.pt",
    output_dir: str = "experiments/exp_009_sft",
):
    print("=" * 80)
    print("  SUPERVISED FINE-TUNING (SFT) — 54.5M PARAMETER SLM")
    print("=" * 80)

    # 1. Load configs
    config = Config.from_yaml(model_config_path)
    with open(sft_config_path, "r") as f:
        sft_cfg = yaml.safe_load(f)["training"]

    from src.utils.device import get_device, select_precision
    device = get_device()
    precision = select_precision(device, sft_cfg.get("precision", "auto"))
    precision_mgr = PrecisionManager(device=device, precision=precision)
    print(f"Device: {device} | Precision: {precision}")

    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>") or 0

    # 3. Build datasets and loaders
    print(f"\nLoading SFT datasets...")
    train_dataset = SFTDataset(train_data_path, tokenizer, max_seq_len=sft_cfg["max_seq_len"])
    val_dataset = SFTDataset(val_data_path, tokenizer, max_seq_len=sft_cfg["max_seq_len"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=sft_cfg["micro_batch_size"],
        shuffle=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
        num_workers=sft_cfg.get("num_workers", 0),
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=sft_cfg["micro_batch_size"],
        shuffle=False,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
        num_workers=sft_cfg.get("num_workers", 0),
    )

    grad_accum_steps = sft_cfg["gradient_accumulation_steps"]
    effective_batch_examples = sft_cfg["micro_batch_size"] * grad_accum_steps
    steps_per_epoch = len(train_dataset) // effective_batch_examples
    total_steps = sft_cfg.get("max_steps") or (steps_per_epoch * sft_cfg.get("epochs", 2))

    # STARTUP REPORTING (Requirement #6)
    print("\n" + "-" * 80)
    print("  SFT DATASET & TRAINING SETUP REPORT")
    print("-" * 80)
    print(f"  • Total SFT Training Tokens   : {train_dataset.total_tokens:,} tokens ({len(train_dataset):,} examples)")
    print(f"  • Total SFT Validation Tokens : {val_dataset.total_tokens:,} tokens ({len(val_dataset):,} examples)")
    print(f"  • Effective Batch Size        : {effective_batch_examples} examples/step ({sft_cfg['micro_batch_size']} micro x {grad_accum_steps} accum)")
    print(f"  • Planned SFT Epochs          : {sft_cfg.get('epochs', 2)} epochs ({steps_per_epoch:,} steps/epoch)")
    print(f"  • Total Optimizer Steps       : {total_steps:,} steps")
    print(f"  • Peak Learning Rate          : {sft_cfg['learning_rate']:.2e} (Cosine decay to {sft_cfg['min_learning_rate']:.2e})")
    print("-" * 80 + "\n")

    # 4. Instantiate Model & Load Base Checkpoint
    model = CausalLM(config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    if Path(base_checkpoint_path).exists():
        print(f"Loading pretrained base weights from {base_checkpoint_path}...")
        ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
        if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None and "shadow" in ckpt["ema_state_dict"]:
            model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
            print("✓ Loaded EMA weights from base checkpoint!")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("✓ Loaded raw weights from base checkpoint!")
        else:
            model.load_state_dict(ckpt, strict=False)
        if model.tie_embeddings:
            model.lm_head.weight = model.token_embedding.weight
    else:
        print(f"Warning: Base checkpoint {base_checkpoint_path} not found. Training from scratch.")

    # 5. Optimizer & Cosine Scheduler
    optimizer = create_optimizer(
        model=model,
        learning_rate=sft_cfg["learning_rate"],
        weight_decay=sft_cfg["weight_decay"],
        beta1=sft_cfg.get("beta1", 0.9),
        beta2=sft_cfg.get("beta2", 0.95),
        eps=sft_cfg.get("eps", 1.0e-8),
    )

    warmup_steps = sft_cfg.get("continuation_warmup_steps", 30)

    def get_lr(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return sft_cfg["learning_rate"] * (step_idx + 1) / max(1, warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = sft_cfg["min_learning_rate"]
        return min_lr + (sft_cfg["learning_rate"] - min_lr) * cosine_decay

    ema = EMAModel(model, decay=sft_cfg.get("ema_decay", 0.999))

    # 6. SFT Training Loop
    print("\n" + "=" * 80)
    print(f"  STARTING SFT TRAINING LOOP")
    print("=" * 80)

    step = 0
    processed_examples = 0
    processed_tokens = 0
    best_val_loss = float("inf")
    model.train()
    optimizer.zero_grad()

    start_time = time.time()
    train_iter = iter(train_loader)

    while step < total_steps:
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        loss_val_sum = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch["x"].to(device)
            y = batch["y"].to(device)

            with precision_mgr.get_autocast_context():
                _, loss, _ = model(x, targets=y)
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            loss_val_sum += loss.item() / grad_accum_steps

            asst_tokens = (y != -100).sum().item()
            processed_tokens += asst_tokens
            processed_examples += x.size(0)

        # Gradient clipping & step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), sft_cfg["grad_clip"])
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        optimizer.step()
        ema.update(model)
        step += 1

        if device.type == "mps" and step % 250 == 0:
            torch.mps.empty_cache()

        current_epoch = round(processed_examples / len(train_dataset), 2)

        # Step Progress Logging (Requirement #6)
        if step % sft_cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            print(
                f"step {step:5d}/{total_steps} | "
                f"epoch: {current_epoch:4.2f}/{sft_cfg.get('epochs', 2)} | "
                f"loss: {loss_val_sum:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"grad: {grad_norm:.4f} | "
                f"tokens: {processed_tokens:,} | "
                f"examples: {processed_examples:,} | "
                f"elapsed: {int(elapsed//60)}m {int(elapsed%60)}s"
            )

        # Full Validation & Qualitative Multi-Domain Checks (Requirements #1, #4, #5, #6)
        if step % sft_cfg["validation_interval"] == 0 or step == total_steps:
            ema.apply(model)

            print("\n" + "=" * 70)
            print(f"  FULL VALIDATION @ Step {step:,} / {total_steps:,} (Epoch {current_epoch:4.2f})")
            print("=" * 70)

            overall_val_loss, overall_ppl, domain_metrics = evaluate_full_validation(
                model=model,
                val_loader=val_loader,
                precision_mgr=precision_mgr,
                device=device,
            )

            is_best = overall_val_loss < best_val_loss
            if is_best:
                best_val_loss = overall_val_loss

            print(f"  ▶ OVERALL SFT VAL LOSS : {overall_val_loss:.4f} | PPL: {overall_ppl:.2f} {'★ NEW BEST' if is_best else ''}")
            print("  " + "-" * 66)
            print(f"  {'Domain':<26} {'Val Loss':<10} {'Perplexity':<12} {'Samples':<8} {'Tokens'}")
            print("  " + "-" * 66)
            for dom_name, m in sorted(domain_metrics.items()):
                print(f"  {dom_name:<26} {m['loss']:<10.4f} {m['perplexity']:<12.2f} {m['samples']:<8,} {m['tokens']:,}")
            print("=" * 70)

            # Live Qualitative Generation Samples
            print("\n--- QUALITATIVE SFT GENERATION SAMPLES ---")
            generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
            for p in EVAL_PROMPTS:
                out_txt = generator.generate(
                    prompt=p,
                    max_new_tokens=120,
                    temperature=0.1,
                    top_k=20,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    use_kv_cache=True,
                )
                print(f"  [PROMPT]: {p}")
                print(f"  [ANSWER]:\n{out_txt}\n")
            print("-------------------------------------------\n")

            # Save checkpoints
            ckpt_state = {
                "step": step,
                "epoch": current_epoch,
                "processed_tokens": processed_tokens,
                "processed_examples": processed_examples,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": overall_val_loss,
                "domain_metrics": domain_metrics,
                "config": {
                    "model": config.model.to_dict() if hasattr(config, "model") else config.to_dict(),
                    "training": sft_cfg,
                },
            }

            if is_best:
                torch.save(ckpt_state, ckpt_dir / "best.pt")
                print(f"✓ Saved best model to {ckpt_dir / 'best.pt'}")

            torch.save(ckpt_state, ckpt_dir / "latest.pt")

            ema.restore(model)
            model.train()

        if step % sft_cfg["checkpoint_interval"] == 0:
            torch.save({
                "step": step,
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_dir / f"checkpoint_step_{step:06d}.pt")

    print("\n" + "=" * 80)
    print(f"  SFT TRAINING COMPLETE! Checkpoints saved in {ckpt_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for 54.5M SLM")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--sft-config", type=str, default="configs/train_sft.yaml", help="Path to SFT train config")
    parser.add_argument("--train-data", type=str, default="data_sft/sft_train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val-data", type=str, default="data_sft/sft_val.jsonl", help="Path to val JSONL")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--base-checkpoint", type=str, default="experiments/exp_008/checkpoints/best.pt", help="Path to base pretrained checkpoint")
    parser.add_argument("--output-dir", type=str, default="experiments/exp_009_sft", help="Output experiment directory")
    args = parser.parse_args()

    train_sft(
        model_config_path=args.model_config,
        sft_config_path=args.sft_config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        tokenizer_path=args.tokenizer,
        base_checkpoint_path=args.base_checkpoint,
        output_dir=args.output_dir,
    )
