"""
CLI text generation script per §42.

Loads trained model checkpoint (preferring EMA weights) and runs interactive or single-prompt generation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from tokenizers import Tokenizer

from src.utils.config import Config
from src.utils.device import get_device
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained 54.5M SLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--prompt", type=str, default="What is 7 + 5?", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (lower = more deterministic/factual)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", "--rep-penalty", type=float, default=1.15, help="Repetition penalty (>1.0 prevents repetition loops)")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache")
    parser.add_argument("--disable-ema", "--no-ema", action="store_true", help="Disable EMA weights and use raw model weights for generation")
    parser.add_argument("--chat", action="store_true", help="Force ChatML formatting (User: ... / Assistant:)")
    parser.add_argument("--raw", action="store_true", help="Force raw document completion without chat template")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive prompt loop")
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
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)

    model.eval()

    # Load tokenizer
    tokenizer_file = Path(args.tokenizer_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    # Generator
    generator = TextGenerator(model, tokenizer, device)

    # Determine chat mode: auto-detect if SFT checkpoint
    is_chat = not args.raw and (args.chat or "sft" in str(ckpt_path).lower())

    if args.interactive:
        print("\n" + "=" * 70)
        print("  INTERACTIVE MODEL TESTING MODE")
        print(f"  Mode: {'Chat Assistant (User: ... / Assistant:)' if is_chat else 'Raw Text Completion'}")
        print("  - Type any question or coding task and press Enter")
        print("  - Supports literal '\\n' for newlines")
        print("  - Type 'exit', 'quit', or 'q' to stop")
        print("=" * 70)

        while True:
            try:
                raw_input = input("\nEnter Prompt > ").strip()
                if not raw_input:
                    continue
                if raw_input.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive mode.")
                    break

                user_text = raw_input.replace("\\n", "\n")

                if is_chat:
                    if not user_text.startswith("User:") and "Assistant:" not in user_text:
                        full_prompt = f"User: {user_text}\n\nAssistant:"
                    else:
                        full_prompt = user_text
                else:
                    full_prompt = user_text

                print("\n--- GENERATING ---")
                output = generator.generate(
                    prompt=full_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    use_kv_cache=not args.no_cache,
                )

                if is_chat and full_prompt in output:
                    reply = output[len(full_prompt):].strip()
                    # Truncate if model hallucinates another User turn
                    if "\nUser:" in reply:
                        reply = reply.split("\nUser:")[0].strip()
                else:
                    reply = output

                print(f"\n--- ASSISTANT REPLY ---\n{reply}\n")
                print("-" * 70)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
    else:
        user_text = args.prompt.replace("\\n", "\n")
        if is_chat and not user_text.startswith("User:") and "Assistant:" not in user_text:
            full_prompt = f"User: {user_text}\n\nAssistant:"
        else:
            full_prompt = user_text

        print(f"\n--- PROMPT ---\n{full_prompt}\n")
        print("--- GENERATING ---")

        output = generator.generate(
            prompt=full_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_kv_cache=not args.no_cache,
        )

        if is_chat and full_prompt in output:
            reply = output[len(full_prompt):].strip()
            if "\nUser:" in reply:
                reply = reply.split("\nUser:")[0].strip()
        else:
            reply = output

        print(f"\n--- ASSISTANT REPLY ---\n{reply}\n")


if __name__ == "__main__":
    main()


