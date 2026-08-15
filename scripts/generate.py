"""
CLI text generation script per §42.

Loads trained model checkpoint (preferring EMA weights) and runs interactive or single-prompt generation.
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.utils.config import Config
from src.utils.device import get_device
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained 75M SLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.0, help="Repetition penalty (1.0 = none, >1.0 penalizes repeats)")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache")
    parser.add_argument("--disable-ema", "--no-ema", action="store_true", help="Disable EMA weights and use raw model weights for generation")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load config
    cfg_dict = checkpoint.get("config", {})
    if isinstance(cfg_dict, dict) and ("model" in cfg_dict or "vocab_size" in cfg_dict):
        config = Config(cfg_dict)
    else:
        config = Config.from_yaml(args.model_config)

    # Build model
    model = CausalLM(config).to(device)

    # Check if EMA weights should be used
    use_ema = not args.disable_ema and checkpoint.get("ema_state_dict") is not None and "shadow" in checkpoint["ema_state_dict"]
    if use_ema:
        print("Using EMA weights for generation...")
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"], strict=False)
        if model.tie_embeddings:
            model.lm_head.weight = model.token_embedding.weight
    else:
        print("Using raw model weights for generation...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()

    # Load tokenizer
    tokenizer_file = Path(args.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # Generator
    generator = TextGenerator(model, tokenizer, device)

    print(f"\n--- PROMPT ---\n{args.prompt}\n")
    print("--- GENERATING ---")

    output = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_kv_cache=not args.no_cache,
    )

    print(f"\n--- GENERATED TEXT ---\n{output}\n")


if __name__ == "__main__":
    main()
