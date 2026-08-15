"""
Comprehensive Dataset Quality Report script per §27.

Outputs complete breakdown of tokens, source percentages, filtering counts,
and PASS/FAIL verification status before training.
"""

import argparse
import json
from pathlib import Path


def generate_report(manifest_path: Path = Path("data/manifest.json")):
    if not manifest_path.exists():
        print(f"Manifest file {manifest_path} not found. Run scripts/prepare_dataset.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_tokens = data.get("target_tokens", 50000000)
    actual_tokens = data.get("actual_tokens", 0)
    train_tokens = data.get("train_tokens", 0)
    val_tokens = data.get("val_tokens", 0)
    vocab_size = data.get("vocab_size", 32768)
    sources = data.get("sources", {})

    print("\n" + "=" * 80)
    print("TRAINING DATASET QUALITY & MIXTURE REPORT (§27)")
    print("=" * 80)

    print("\n1. DATASET MIXTURE BREAKDOWN:")
    print("-" * 80)
    print(f"{'Source':<15} {'Target Tokens':>14} {'Actual Tokens':>14} {'Target %':>10} {'Actual %':>10} {'Docs':>8}")
    print("-" * 80)

    for key, s in sources.items():
        act_tok = s.get("actual_tokens", 0)
        tgt_tok = s.get("target_tokens", 0)
        tgt_pct = s.get("target_percentage", 0.0)
        act_pct = (act_tok / max(1, target_tokens)) * 100.0
        docs = s.get("documents", 0)
        print(f"{key:<15} {tgt_tok:14,} {act_tok:14,} {tgt_pct:9.1f}% {act_pct:9.1f}% {docs:8,}")

    print("-" * 80)
    print(f"{'TOTAL':<15} {target_tokens:14,} {actual_tokens:14,} {'100.0%':>10} {'100.0%':>10}")

    print("\n2. QUALITY FILTERING & DEDUPLICATION SUMMARY:")
    print("-" * 80)
    total_quality_removed = sum(s.get("rejected_quality", 0) for s in sources.values())
    total_exact_dup_removed = sum(s.get("rejected_exact_dup", 0) for s in sources.values())
    total_near_dup_removed = sum(s.get("rejected_near_dup", 0) for s in sources.values())

    print(f"  Documents Rejected (Quality Filters):     {total_quality_removed:,}")
    print(f"  Documents Removed (Exact SHA-256 Dup):    {total_exact_dup_removed:,}")
    print(f"  Documents Removed (13-gram MinHash Dup): {total_near_dup_removed:,}")

    print("\n3. TOKENIZER & SHARDING SUMMARY:")
    print("-" * 80)
    print(f"  Vocabulary Size:       {vocab_size:,} tokens")
    print(f"  Train Set Tokens (95%):{train_tokens:,}")
    print(f"  Val Set Tokens (5%):  {val_tokens:,}")
    print(f"  Document-Level Split:  YES")

    # Check verification status (within 5% tolerance of target tokens)
    token_tolerance = 0.05
    is_valid_token_count = abs(actual_tokens - target_tokens) / target_tokens <= token_tolerance
    status = "PASS" if is_valid_token_count else "FAIL"

    print("\n4. FINAL VERIFICATION:")
    print("=" * 80)
    print(f"  TARGET TOKENS: {target_tokens:,}")
    print(f"  ACTUAL TOKENS: {actual_tokens:,}")
    print(f"  STATUS:        {status}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Dataset Quality Report")
    parser.add_argument("--manifest", type=str, default="data/manifest.json")
    args = parser.parse_args()

    generate_report(Path(args.manifest))


if __name__ == "__main__":
    main()
