"""
Evaluation harness entrypoint script per §43.

Evaluates validation loss/perplexity and lightweight benchmark tasks on a trained checkpoint.
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.model.gpt import CausalLM
from src.data.shard_dataset import ShardDataset
from src.evaluation.loss import evaluate_validation_loss
from src.evaluation.benchmark_tasks import BenchmarkSuite
from src.training.precision import PrecisionManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate 75M SLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data-dir", type=str, default="data/processed/validation", help="Path to val shards")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Path to tokenizer directory")
    args = parser.parse_args()

    device = get_device()
    precision = select_precision(device, "auto")
    precision_mgr = PrecisionManager(device, precision)

    print(f"Device: {device}, Precision: {precision}")

    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = Config(checkpoint.get("config", {})) if checkpoint.get("config") else Config.from_yaml("configs/model.yaml")

    model = CausalLM(config).to(device)

    # Use EMA weights if available
    if checkpoint.get("ema_state_dict") is not None and "shadow" in checkpoint["ema_state_dict"]:
        print("Evaluating with EMA weights...")
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"], strict=False)
    else:
        print("Evaluating with raw model weights...")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Val dataset loss evaluation (supports comma-separated directories)
    data_dirs = [d.strip() for d in args.data_dir.split(",") if d.strip()]
    for data_dir_str in data_dirs:
        val_path = Path(data_dir_str)
        if val_path.exists() and list(val_path.glob("shard_*.bin")):
            val_dataset = ShardDataset(val_path, seq_len=config.model.max_seq_len, pack_with_document_mask=True)
            val_loss, perplexity = evaluate_validation_loss(model, val_dataset, device, precision_mgr)

            print("\n" + "=" * 60)
            print(f"EVALUATION RESULTS REPORT [{val_path}]")
            print("=" * 60)
            print(f"  Validation Loss:       {val_loss:.4f}")
            print(f"  Validation Perplexity: {perplexity:.2f}")
            print("=" * 60 + "\n")
        else:
            print(f"Validation dataset path {val_path} not found; skipping validation loss calculation.")

    # Benchmark tasks
    tokenizer_file = Path(args.tokenizer_dir) / "tokenizer.json"
    if tokenizer_file.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        suite = BenchmarkSuite(model, tokenizer, device)

        # Sample LAMBADA-style evaluation
        lambada_samples = [
            {"context": "The chef prepared a delicious slice of pizza and put it on a", "target_word": "plate"},
            {"context": "She turned off the lights and went to sleep in her", "target_word": "bed"},
        ]
        lambada_res = suite.evaluate_lambada_sample(lambada_samples)
        print(f"  LAMBADA Sample Accuracy:   {lambada_res['lambada_accuracy']*100:.1f}%")
        print(f"  LAMBADA Sample Perplexity: {lambada_res['lambada_perplexity']:.2f}")

        # Sample ARC-Easy style 4-way multiple choice
        mc_questions = [
            {
                "prompt": "Which object absorbs light best?",
                "choices": ["White paper", "Black cloth", "Clear glass", "Mirror"],
                "gold": 1,
            }
        ]
        mc_res = suite.evaluate_multiple_choice(mc_questions, task_name="arc_easy")
        print(f"  ARC-Easy Sample Accuracy:  {mc_res['arc_easy_accuracy']*100:.1f}%")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
