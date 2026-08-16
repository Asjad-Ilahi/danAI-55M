import sys, json, torch, re, argparse, ast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tokenizers import Tokenizer
from src.utils.config import Config
from src.utils.device import get_device
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator


def load_model(checkpoint_path: str = "experiments/exp_010_sft/checkpoints/best.pt", disable_ema: bool = False):
    device = get_device()
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        if Path("experiments/exp_010_sft/checkpoints/best.pt").exists():
            ckpt_file = Path("experiments/exp_010_sft/checkpoints/best.pt")
        elif Path("experiments/exp_010_sft/checkpoints/latest.pt").exists():
            ckpt_file = Path("experiments/exp_010_sft/checkpoints/latest.pt")
        elif Path("experiments/exp_009/checkpoints/latest.pt").exists():
            ckpt_file = Path("experiments/exp_009/checkpoints/latest.pt")

    print(f"Loading checkpoint: {ckpt_file} onto {device}...")
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    config = Config.from_yaml("configs/model.yaml")
    model = CausalLM(config).to(device)
    if not disable_ema and "ema_state_dict" in ckpt and "shadow" in ckpt["ema_state_dict"]:
        model.load_state_dict(ckpt["ema_state_dict"]["shadow"], strict=False)
        if model.tie_embeddings:
            model.lm_head.weight = model.token_embedding.weight
        print("  [OK] Loaded EMA weights for evaluation.")
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("  [OK] Loaded raw model weights for evaluation.")
    model.eval()
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    return TextGenerator(model, tokenizer, device), str(ckpt_file)


PROBES = [
    # === MATH: Basic Arithmetic ===
    {"cat": "Math-Addition",      "prompt": "What is 7 + 5?", "gold": ["12"], "temp": 0.1, "max": 40},
    {"cat": "Math-Addition",      "prompt": "What is 23 + 45?", "gold": ["68"], "temp": 0.1, "max": 40},
    {"cat": "Math-Addition",      "prompt": "What is 156 + 289?", "gold": ["445"], "temp": 0.1, "max": 40},
    {"cat": "Math-Multiplication","prompt": "What is 6 * 7?", "gold": ["42"], "temp": 0.1, "max": 40},
    {"cat": "Math-Multiplication","prompt": "What is 12 * 15?", "gold": ["180"], "temp": 0.1, "max": 40},
    {"cat": "Math-Division",      "prompt": "What is 56 / 8?", "gold": ["7"], "temp": 0.1, "max": 40},
    {"cat": "Math-Subtraction",   "prompt": "What is 100 - 37?", "gold": ["63"], "temp": 0.1, "max": 40},
    {"cat": "Math-WordProblem",   "prompt": "Sarah has 15 apples. She gives 4 to Bob and buys 7 more. How many apples does she have now?", "gold": ["18"], "temp": 0.1, "max": 80},

    # === SCIENCE ===
    {"cat": "Science-Biology",    "prompt": "What is photosynthesis?", "gold": ["sunlight", "light", "glucose", "food", "energy from sun"], "temp": 0.1, "max": 80},
    {"cat": "Science-Chemistry",  "prompt": "What is the chemical formula for water?", "gold": ["H2O", "h2o"], "temp": 0.1, "max": 40},
    {"cat": "Science-Physics",    "prompt": "Why does an apple fall from a tree?", "gold": ["gravity", "gravitational"], "temp": 0.1, "max": 80},
    {"cat": "Science-Biology",    "prompt": "How many chambers does the human heart have?", "gold": ["4", "four"], "temp": 0.1, "max": 40},
    {"cat": "Science-Physics",    "prompt": "What is the speed of light in a vacuum?", "gold": ["300", "299", "300,000", "186,000", "3 x 10", "3*10"], "temp": 0.1, "max": 50},
    {"cat": "Science-Chemistry",  "prompt": "What gas do plants absorb from the atmosphere during photosynthesis?", "gold": ["carbon dioxide", "co2"], "temp": 0.1, "max": 50},

    # === GENERAL KNOWLEDGE ===
    {"cat": "GK-Geography",       "prompt": "What is the capital of France?", "gold": ["Paris", "paris"], "temp": 0.1, "max": 30},
    {"cat": "GK-Geography",       "prompt": "What is the largest ocean on Earth?", "gold": ["Pacific", "pacific"], "temp": 0.1, "max": 40},
    {"cat": "GK-Geography",       "prompt": "What is the longest river in the world?", "gold": ["Nile", "nile", "Amazon"], "temp": 0.1, "max": 40},
    {"cat": "GK-History",         "prompt": "In what year did World War II end?", "gold": ["1945"], "temp": 0.1, "max": 40},
    {"cat": "GK-History",         "prompt": "Who was the first President of the United States?", "gold": ["Washington", "washington"], "temp": 0.1, "max": 40},
    {"cat": "GK-Science",         "prompt": "How many planets are in our solar system?", "gold": ["8", "eight"], "temp": 0.1, "max": 40},

    # === REASONING ===
    {"cat": "Logic-Syllogism",    "prompt": "All dogs are mammals. All mammals are animals. Therefore, all dogs are:", "gold": ["animals", "mammals and animals"], "temp": 0.1, "max": 40},
    {"cat": "Logic-Negation",     "prompt": "Is 1 + 1 equal to 3?", "gold": ["No", "no", "not equal", "false", "2"], "temp": 0.1, "max": 40},
    {"cat": "Logic-Negation",     "prompt": "Can fish fly in the sky like birds?", "gold": ["No", "no", "cannot fly", "can't fly"], "temp": 0.1, "max": 40},
    {"cat": "Logic-Temporal",     "prompt": "If today is Monday, what day will it be in 3 days?", "gold": ["Thursday", "thursday"], "temp": 0.1, "max": 40},
    {"cat": "Logic-Adversarial",  "prompt": "Did Abraham Lincoln use a smartphone?", "gold": ["No", "no", "not invented", "didn't", "did not"], "temp": 0.1, "max": 50},

    # === CODE ===
    {"cat": "Code-Function",      "prompt": "Write a Python function to check if a number is even:\n```python\ndef is_even(n):\n", "gold": ["n % 2 == 0", "return n % 2 == 0", "not n % 2", "n % 2"], "temp": 0.1, "max": 60},
    {"cat": "Code-Function",      "prompt": "Write a Python function to find the maximum value in a list:\n```python\ndef find_max(lst):\n", "gold": ["max", "max_val", "item >"], "temp": 0.1, "max": 80},
]


def grade(output: str, gold_list: list, cat: str = "") -> bool:
    """Check if any gold answer variant appears accurately in the output."""
    out_lower = output.lower()
    
    # Specific code validation
    if "Code" in cat and "```python" in output:
        blocks = re.findall(r"```python\s*(.*?)\s*```", output, re.DOTALL)
        if blocks:
            try:
                ast.parse(blocks[0])
                if "is_even" in cat.lower() or "%" in blocks[0]:
                    return True
                if "max" in blocks[0]:
                    return True
            except Exception:
                pass

    for g in gold_list:
        if g.lower() in out_lower:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Weakness Diagnostic Benchmark")
    parser.add_argument("--checkpoint", type=str, default="experiments/exp_010_sft/checkpoints/best.pt",
                        help="Path to model checkpoint to evaluate")
    parser.add_argument("--disable-ema", action="store_true", help="Use raw online weights instead of EMA shadow weights")
    parser.add_argument("--format", type=str, choices=["base", "chat", "auto"], default="auto",
                        help="Prompt format: 'chat' (User: / Assistant:), 'base' (raw prompt), or 'auto'")
    args = parser.parse_args()

    gen, ckpt_path = load_model(args.checkpoint, args.disable_ema)
    is_chat = args.format == "chat" or (args.format == "auto" and "sft" in ckpt_path.lower())

    results = []
    categories = {}

    print("=" * 90)
    print(f"  COMPREHENSIVE MODEL WEAKNESS DIAGNOSTIC ({ckpt_path})")
    print(f"  Format: {'ChatML (User: ... / Assistant:)' if is_chat else 'Base Completion'}")
    print("=" * 90)

    for i, p in enumerate(PROBES, 1):
        if is_chat:
            if "```python" in p["prompt"]:
                # Code prompt with starter block
                parts = p["prompt"].split("\n```python\n")
                prompt_text = f"User: {parts[0]}\n\nAssistant: ```python\n{parts[1]}"
            else:
                prompt_text = f"User: {p['prompt']}\n\nAssistant:"
        else:
            prompt_text = f"Question: {p['prompt']}\nAnswer:" if not p['prompt'].startswith("Write a") else p['prompt']

        out = gen.generate(
            prompt=prompt_text,
            max_new_tokens=p["max"],
            temperature=p["temp"],
            top_k=20, top_p=0.9,
            repetition_penalty=1.15,
            use_kv_cache=True,
        )
        continuation = out[len(prompt_text):].strip()
        passed = grade(continuation, p["gold"], p["cat"])
        status = "PASS" if passed else "FAIL"

        cat = p["cat"].split("-")[0]
        if cat not in categories:
            categories[cat] = {"pass": 0, "fail": 0, "total": 0}
        categories[cat]["total"] += 1
        categories[cat]["pass" if passed else "fail"] += 1

        print(f"\n[{i:02d}/{len(PROBES)}] {p['cat']} [{status}]")
        print(f"  Prompt: {p['prompt'][:80]}...")
        clean_out = continuation.replace("\n", " ")[:120]
        print(f"  Output: {clean_out}")
        print(f"  Gold:   {p['gold'][0]}")

        results.append({
            "category": p["cat"],
            "prompt": p["prompt"],
            "gold": p["gold"][0],
            "output": continuation,
            "passed": passed,
        })

    print("\n" + "=" * 90)
    print("  CATEGORY SCORECARD")
    print("=" * 90)
    for cat, scores in sorted(categories.items()):
        pct = (scores["pass"] / scores["total"]) * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"  {cat:12s} [{bar}] {scores['pass']:2d}/{scores['total']:2d} ({pct:5.1f}%)")

    total_pass = sum(s["pass"] for s in categories.values())
    total = sum(s["total"] for s in categories.values())
    print(f"\n  OVERALL:     {total_pass}/{total} ({total_pass/total*100:.1f}%)")

    out_file = Path("experiments/weakness_diagnostic_results_sft.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "categories": {k: v for k, v in categories.items()}}, f, indent=2)
    print(f"\n[OK] Saved results to {out_file}")


if __name__ == "__main__":
    main()
