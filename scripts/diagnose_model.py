"""
Comprehensive Capability & Failure Mode Diagnostic Suite for 54.5M SLM (exp_008).
Probes the model across Coding, Mathematics, World Knowledge, Multi-Step Reasoning,
Instruction Adherence, and Syntax/Grammar to identify exact architectural and data gaps.
"""

import sys
import json
import torch
from pathlib import Path
from tokenizers import Tokenizer

from src.utils.config import Config
from src.utils.device import get_device
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator


def load_model_and_tokenizer(checkpoint_path="experiments/exp_008/checkpoints/best.pt"):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config.from_yaml("configs/model.yaml")
    model = CausalLM(config).to(device)
    
    if "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
        model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
        if model.tie_embeddings:
            model.lm_head.weight = model.token_embedding.weight
        print("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Loaded raw weights.")
        
    model.eval()
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    generator = TextGenerator(model, tokenizer, device)
    return generator, device


TEST_PROBES = [
    # 1. Code Generation
    {
        "category": "Coding",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
        "temp": 0.2,
        "rep_pen": 1.15,
        "max_tokens": 80,
    },
    {
        "category": "Coding",
        "prompt": "def is_prime(num):\n    \"\"\"Return True if num is a prime number, else False.\"\"\"\n",
        "temp": 0.2,
        "rep_pen": 1.15,
        "max_tokens": 80,
    },
    {
        "category": "Coding",
        "prompt": "# Python function to reverse a string\ndef reverse_string(s):\n",
        "temp": 0.2,
        "rep_pen": 1.15,
        "max_tokens": 60,
    },
    
    # 2. Arithmetic & Mathematics
    {
        "category": "Mathematics",
        "prompt": "Question: What is 47 + 58?\nAnswer:",
        "temp": 0.1,
        "rep_pen": 1.1,
        "max_tokens": 50,
    },
    {
        "category": "Mathematics",
        "prompt": "Question: What is 12 * 15?\nAnswer:",
        "temp": 0.1,
        "rep_pen": 1.1,
        "max_tokens": 50,
    },
    {
        "category": "Mathematics",
        "prompt": "Solve for x: 3*x + 9 = 24.\nStep 1:",
        "temp": 0.2,
        "rep_pen": 1.15,
        "max_tokens": 80,
    },
    {
        "category": "Mathematics",
        "prompt": "Sarah has 5 apples. She buys 3 more boxes of apples, with 6 apples in each box. How many apples does Sarah have in total?\nLet's calculate step by step:\n",
        "temp": 0.2,
        "rep_pen": 1.15,
        "max_tokens": 100,
    },
    
    # 3. Factual & Scientific Knowledge
    {
        "category": "Scientific Knowledge",
        "prompt": "Photosynthesis is the biological process by which plants",
        "temp": 0.6,
        "rep_pen": 1.15,
        "max_tokens": 80,
    },
    {
        "category": "Scientific Knowledge",
        "prompt": "The three states of matter are solid, liquid, and gas. A solid has",
        "temp": 0.6,
        "rep_pen": 1.15,
        "max_tokens": 80,
    },
    {
        "category": "Scientific Knowledge",
        "prompt": "The capital of France is",
        "temp": 0.2,
        "rep_pen": 1.1,
        "max_tokens": 30,
    },
    
    # 4. Multi-Step Reasoning & Logic
    {
        "category": "Logic & Reasoning",
        "prompt": "All humans are mortal. Socrates is a human. Therefore,",
        "temp": 0.2,
        "rep_pen": 1.1,
        "max_tokens": 40,
    },
    {
        "category": "Logic & Reasoning",
        "prompt": "If today is Tuesday, what day will it be in 4 days?\nAnswer:",
        "temp": 0.2,
        "rep_pen": 1.1,
        "max_tokens": 40,
    },
    
    # 5. Instruction & Conversational Turn-Taking (User/Assistant)
    {
        "category": "Instruction Following",
        "prompt": "User: What are 3 healthy habits for daily life?\nAssistant:",
        "temp": 0.6,
        "rep_pen": 1.15,
        "max_tokens": 100,
    },
    {
        "category": "Instruction Following",
        "prompt": "User: Write a 2-line poem about the moon.\nAssistant:",
        "temp": 0.7,
        "rep_pen": 1.15,
        "max_tokens": 60,
    }
]


def run_diagnostics():
    print("=" * 80)
    print("  RUNNING DIAGNOSTIC PROBES ON 54.5M SLM (exp_008)")
    print("=" * 80)
    
    generator, device = load_model_and_tokenizer()
    results = []
    
    for i, test in enumerate(TEST_PROBES, 1):
        print(f"\n[{i}/{len(TEST_PROBES)}] Testing {test['category']}...")
        prompt = test["prompt"]
        output = generator.generate(
            prompt=prompt,
            max_new_tokens=test["max_tokens"],
            temperature=test["temp"],
            top_k=40,
            top_p=0.9,
            repetition_penalty=test["rep_pen"],
            use_kv_cache=True,
        )
        
        # Extract generated continuation only
        continuation = output[len(prompt):]
        
        print(f"PROMPT:\n{prompt}")
        print(f"GENERATION:\n{continuation.strip()}")
        print("-" * 60)
        
        results.append({
            "category": test["category"],
            "prompt": prompt,
            "continuation": continuation.strip(),
            "full_output": output,
            "params": test
        })
        
    out_file = Path("experiments/diagnostic_probe_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved diagnostic results to {out_file}")


if __name__ == "__main__":
    run_diagnostics()
