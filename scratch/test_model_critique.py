"""
Comprehensive Evaluation and Model Performance Critique Script.
Compares EMA ON vs EMA DISABLED on SFT and Pretrained checkpoints.
"""

import math
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.training.precision import PrecisionManager
from src.model.gpt import CausalLM
from src.data.sft_dataset import SFTDataset, sft_collate_fn
from src.data.shard_dataset import ShardDataset
from src.evaluation.loss import evaluate_validation_loss
from src.evaluation.generation import TextGenerator


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[CausalLM, Config, Tokenizer, Dict[str, Any]]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg_dict = checkpoint.get("config", {})
    if isinstance(cfg_dict, dict) and ("model" in cfg_dict or "vocab_size" in cfg_dict):
        config = Config(cfg_dict)
    else:
        config = Config.from_yaml("configs/model.yaml")

    model = CausalLM(config).to(device)

    if use_ema and checkpoint.get("ema_state_dict") is not None and "shadow" in checkpoint["ema_state_dict"]:
        print(f"[{ckpt_path.name}] Loading EMA shadow weights...")
        model.load_state_dict(checkpoint["ema_state_dict"]["shadow"], strict=False)
    elif "model_state_dict" in checkpoint:
        print(f"[{ckpt_path.name}] Loading raw model weights (EMA DISABLED)...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        print(f"[{ckpt_path.name}] Loading raw weights directly...")
        model.load_state_dict(checkpoint, strict=False)

    if model.tie_embeddings:
        model.lm_head.weight = model.token_embedding.weight

    model.eval()

    tokenizer_path = "tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    return model, config, tokenizer, checkpoint


def map_domain_name(raw_domain: str) -> str:
    d = raw_domain.lower()
    if "general_instruction" in d or "smol" in d:
        return "Conversation (SmolTalk)"
    elif "instruction_following" in d or "tulu" in d:
        return "Reasoning (Tulu 3)"
    elif "math" in d:
        return "Mathematics"
    elif "code" in d or "coding" in d:
        return "Coding"
    elif "qa" in d or "everyday" in d or "orca" in d:
        return "General Q&A"
    return "General"


def evaluate_sft_val_dataset(
    model: CausalLM,
    val_loader: DataLoader,
    precision_mgr: PrecisionManager,
    device: torch.device,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
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


def evaluate_lambada(model: CausalLM, tokenizer: Tokenizer, device: torch.device) -> Dict[str, float]:
    samples = [
        {"context": "The chef prepared a delicious slice of pizza and put it on a", "target_word": "plate"},
        {"context": "She turned off the lights and went to sleep in her", "target_word": "bed"},
        {"context": "He took out his wallet and paid with a ten dollar", "target_word": "bill"},
        {"context": "The teacher wrote the math equation on the black", "target_word": "board"},
        {"context": "The dog was thirsty so it drank water from the", "target_word": "bowl"},
        {"context": "He opened the door using a metal", "target_word": "key"},
        {"context": "She was feeling cold so she put on a warm winter", "target_word": "coat"},
        {"context": "The bird spread its wings and flew up into the blue", "target_word": "sky"},
        {"context": "He checked his wrist to see the time on his", "target_word": "watch"},
        {"context": "The children built a large sandcastle on the sandy ocean", "target_word": "beach"},
        {"context": "She poured hot coffee into her favorite ceramic", "target_word": "mug"},
        {"context": "He sat down at his computer desk and typed on the", "target_word": "keyboard"},
    ]

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for sample in samples:
            context = sample["context"]
            target = sample["target_word"]
            full_text = context + " " + target

            ctx_ids = tokenizer.encode(context).ids
            full_ids = tokenizer.encode(full_text).ids

            if len(full_ids) <= len(ctx_ids):
                continue

            input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            logits, _, _ = model(input_tensor)

            target_start_pos = len(ctx_ids) - 1
            word_logits = logits[0, target_start_pos:]
            word_targets = target_tensor[0, target_start_pos:]

            loss = F.cross_entropy(word_logits, word_targets, reduction="mean")
            total_loss += loss.item()

            preds = torch.argmax(word_logits, dim=-1)
            if torch.equal(preds, word_targets):
                correct += 1
            total += 1

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    ppl = math.exp(min(20.0, avg_loss))
    return {"accuracy": acc, "loss": avg_loss, "perplexity": ppl, "total": total, "correct": correct}


def evaluate_multiple_choice(model: CausalLM, tokenizer: Tokenizer, device: torch.device) -> Dict[str, float]:
    questions = [
        {
            "prompt": "Which object absorbs light best?",
            "choices": ["White paper", "Black cloth", "Clear glass", "Mirror"],
            "gold": 1,
        },
        {
            "prompt": "What is the primary organ used for pumping blood in the human body?",
            "choices": ["Lungs", "Brain", "Heart", "Stomach"],
            "gold": 2,
        },
        {
            "prompt": "Which planet is closest to the Sun?",
            "choices": ["Venus", "Earth", "Mars", "Mercury"],
            "gold": 3,
        },
        {
            "prompt": "Water freezes at what temperature in Celsius?",
            "choices": ["0 degrees", "32 degrees", "100 degrees", "-10 degrees"],
            "gold": 0,
        },
        {
            "prompt": "What gas do plants absorb during photosynthesis?",
            "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Helium"],
            "gold": 1,
        },
        {
            "prompt": "Which animal is a mammal?",
            "choices": ["Frog", "Dolphin", "Eagle", "Snake"],
            "gold": 1,
        },
        {
            "prompt": "What force pulls objects toward the center of the Earth?",
            "choices": ["Magnetism", "Friction", "Gravity", "Electricity"],
            "gold": 2,
        },
        {
            "prompt": "Which tool is used to measure temperature?",
            "choices": ["Barometer", "Thermometer", "Speedometer", "Compass"],
            "gold": 1,
        },
    ]

    correct = 0
    total = 0

    with torch.no_grad():
        for q in questions:
            prompt = q["prompt"]
            choices = q["choices"]
            gold = q["gold"]

            choice_losses = []
            for choice in choices:
                text = prompt + " " + choice
                prompt_ids = tokenizer.encode(prompt).ids
                full_ids = tokenizer.encode(text).ids

                if len(full_ids) <= len(prompt_ids):
                    choice_losses.append(float("inf"))
                    continue

                input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
                target_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

                logits, _, _ = model(input_tensor)

                start_pos = len(prompt_ids) - 1
                choice_logits = logits[0, start_pos:]
                choice_targets = target_tensor[0, start_pos:]

                loss = F.cross_entropy(choice_logits, choice_targets, reduction="mean")
                choice_losses.append(loss.item())

            pred_choice = int(torch.argmin(torch.tensor(choice_losses)).item())
            if pred_choice == gold:
                correct += 1
            total += 1

    acc = correct / max(1, total)
    return {"accuracy": acc, "total": total, "correct": correct}


def run_qualitative_suite(generator: TextGenerator) -> List[Dict[str, str]]:
    prompts = [
        # Math & Arithmetic
        {"cat": "Arithmetic (Basic)", "p": "User: What is 1 + 1?\n\nAssistant:"},
        {"cat": "Arithmetic (Addition)", "p": "User: What is 23 + 45?\n\nAssistant:"},
        {"cat": "Arithmetic (Hundreds)", "p": "User: What is 100 + 500?\n\nAssistant:"},
        {"cat": "Arithmetic (Multiplication)", "p": "User: What is 7 * 8?\n\nAssistant:"},
        {"cat": "Arithmetic (Division)", "p": "User: Calculate 125 / 5.\n\nAssistant:"},
        {"cat": "Arithmetic (Edge)", "p": "User: What is 999 + 1?\n\nAssistant:"},
        
        # Word Problems
        {"cat": "Word Problem (Apples)", "p": "User: Sarah has 15 apples. She gives 4 to Bob and buys 7 more. How many apples does she have now?\n\nAssistant:"},
        {"cat": "Word Problem (Speed)", "p": "User: A car travels at 60 miles per hour for 3 hours. How far does it travel in miles?\n\nAssistant:"},
        {"cat": "Word Problem (Discount)", "p": "User: A shirt costs $40. It is on sale for 20% off. What is the discount amount and what is the final price?\n\nAssistant:"},

        # Python Coding
        {"cat": "Code (Even Check)", "p": "User: Write a Python function to check if a number is even.\n\nAssistant:"},
        {"cat": "Code (Reverse String)", "p": "User: Write a Python function to reverse a string.\n\nAssistant:"},
        {"cat": "Code (Fibonacci)", "p": "User: Write a Python function to return the nth Fibonacci number.\n\nAssistant:"},
        {"cat": "Code (Binary Search)", "p": "User: Write a Python function to perform binary search on a sorted list.\n\nAssistant:"},

        # Factual Science & Knowledge
        {"cat": "Science (Photosynthesis)", "p": "User: What is photosynthesis?\n\nAssistant:"},
        {"cat": "Science (Blue Sky)", "p": "User: Why is the sky blue?\n\nAssistant:"},
        {"cat": "Science (Newton)", "p": "User: State Newton's first law of motion.\n\nAssistant:"},
        {"cat": "Knowledge (Capital)", "p": "User: What is the capital of France?\n\nAssistant:"},

        # Reasoning & Instruction Following
        {"cat": "Instruction (List 3)", "p": "User: List exactly 3 health benefits of drinking water. Format as a numbered list 1, 2, 3.\n\nAssistant:"},
        {"cat": "Logic (Syllogism)", "p": "User: All roses are flowers. All flowers are plants. Are all roses plants? Answer with yes or no and a short explanation.\n\nAssistant:"},
        {"cat": "Logic (Weight)", "p": "User: Which is heavier: 1 pound of iron or 1 pound of feathers?\n\nAssistant:"},

        # Conversation & Multi-turn
        {"cat": "Chat (Greeting)", "p": "User: Hello! Who are you and what can you do?\n\nAssistant:"},
        {"cat": "Chat (Advice)", "p": "User: I have an exam tomorrow and I'm stressed. Give me two quick tips to relax.\n\nAssistant:"},

        # Hallucination / Edge Cases
        {"cat": "Trap (Historical Fact)", "p": "User: In what year did George Washington fly to the moon?\n\nAssistant:"},
        {"cat": "Trap (3-legged dog)", "p": "User: How many legs does a three-legged dog have?\n\nAssistant:"},
    ]

    results = []
    for item in prompts:
        out = generator.generate(
            prompt=item["p"],
            max_new_tokens=80,
            temperature=0.1,
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        results.append({
            "category": item["cat"],
            "prompt": item["p"],
            "response": out,
        })
    return results


def main():
    device = get_device()
    precision = select_precision(device, "auto")
    precision_mgr = PrecisionManager(device=device, precision=precision)

    print("=" * 80)
    print("  COMPREHENSIVE MODEL EVALUATION & CRITIQUE HARNESS")
    print(f"  Device: {device} | Precision: {precision}")
    print("=" * 80)

    # 1. Setup Val DataLoader
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    pad_id = tokenizer.token_to_id("<pad>") or 0
    val_dataset = SFTDataset("data_sft/sft_val.jsonl", tokenizer, max_seq_len=1024)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
    )

    configs_to_test = [
        {
            "name": "SFT Checkpoint Step 600 (EMA ON)",
            "path": "experiments/exp_009_sft/checkpoints/checkpoint_step_000600.pt",
            "use_ema": True,
        },
        {
            "name": "SFT Checkpoint Step 600 (EMA DISABLED)",
            "path": "experiments/exp_009_sft/checkpoints/checkpoint_step_000600.pt",
            "use_ema": False,
        },
        {
            "name": "SFT Best Checkpoint Step 610 (EMA ON)",
            "path": "experiments/exp_009_sft/checkpoints/best.pt",
            "use_ema": True,
        },
        {
            "name": "Base Pretrained Model (Exp 008 Best - EMA ON)",
            "path": "experiments/exp_008/checkpoints/best.pt",
            "use_ema": True,
        },
    ]

    all_results = {}

    for cfg in configs_to_test:
        print(f"\n>>> Running Evaluation for: {cfg['name']} <<<")
        model, mconfig, tok, ckpt = load_model_from_checkpoint(cfg["path"], device, use_ema=cfg["use_ema"])
        
        # 1. SFT Validation Loss & Perplexity
        print("  Evaluating Full SFT Validation Set (1,570 samples)...")
        overall_loss, overall_ppl, domain_metrics = evaluate_sft_val_dataset(
            model, val_loader, precision_mgr, device
        )

        # 2. Pretraining Validation Shards
        pretrain_val_path = Path("data_100m/shards/val")
        pretrain_loss, pretrain_ppl = None, None
        if pretrain_val_path.exists() and list(pretrain_val_path.glob("shard_*.bin")):
            print("  Evaluating Pretraining Validation Shards...")
            pt_val_dataset = ShardDataset(pretrain_val_path, seq_len=mconfig.model.max_seq_len, pack_with_document_mask=True)
            pretrain_loss, pretrain_ppl = evaluate_validation_loss(model, pt_val_dataset, device, precision_mgr)

        # 3. LAMBADA & Multiple Choice
        print("  Evaluating LAMBADA & Multiple Choice Benchmarks...")
        lambada_res = evaluate_lambada(model, tok, device)
        mc_res = evaluate_multiple_choice(model, tok, device)

        # 4. Qualitative Test Suite
        print("  Generating Qualitative Test Suite Samples...")
        generator = TextGenerator(model=model, tokenizer=tok, device=device)
        qual_results = run_qualitative_suite(generator)

        all_results[cfg["name"]] = {
            "overall_loss": overall_loss,
            "overall_ppl": overall_ppl,
            "domain_metrics": domain_metrics,
            "pretrain_loss": pretrain_loss,
            "pretrain_ppl": pretrain_ppl,
            "lambada": lambada_res,
            "multiple_choice": mc_res,
            "qualitative": qual_results,
        }

    # Save all raw results to scratch
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)
    out_file = scratch_dir / "evaluation_results_full.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved complete evaluation results to {out_file}")


if __name__ == "__main__":
    main()
