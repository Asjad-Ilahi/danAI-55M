"""
Model inspection script per §4 & §50.

Prints device, vocab size, context length, layers, hidden size, query/KV heads,
total and trainable parameters, memory estimations, AND the full architecture
search report table (§4) showing candidate architectures considered and why
this configuration was selected.
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.model.gpt import CausalLM
from src.model.parameter_count import (
    compute_parameter_count,
    run_architecture_search,
    print_search_report,
    print_parameter_breakdown,
    verify_against_model,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect model architecture and print search report")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = get_device()
    precision = select_precision(device, "auto")

    # Instantiated model
    model = CausalLM(config).to(device)

    total_params = model.get_num_params()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. Run architecture search across space and print report (§4)
    search_results = run_architecture_search(target_min=65_000_000, target_max=85_000_000)
    print_search_report(search_results)

    # 2. Analytical parameter breakdown for current config
    mc = config.model
    analytical_counts = compute_parameter_count(
        vocab_size=mc.vocab_size,
        hidden_size=mc.hidden_size,
        num_layers=mc.num_layers,
        num_query_heads=mc.num_query_heads,
        num_kv_heads=mc.num_kv_heads,
        intermediate_size=mc.get("intermediate_size", "auto"),
        tie_embeddings=mc.get("tie_embeddings", True),
        use_bias=mc.get("use_bias", False),
    )
    print_parameter_breakdown(analytical_counts)

    # 3. Verify analytical match (§4)
    verify_against_model(model, analytical_counts["total"])
    print("✓ Analytical parameter count matches instantiated model EXACTLY!")

    # 4. Memory estimation (§26)
    mem_est = model.estimate_memory_mb(dtype=precision)

    print("\n" + "=" * 60)
    print("MODEL INSPECTION & MEMORY SUMMARY")
    print("=" * 60)
    print(f"  Device:                 {device}")
    print(f"  Precision Dtype:        {precision}")
    print(f"  Total Parameters:       {total_params:,}")
    print(f"  Trainable Parameters:   {trainable_params:,}")
    print(f"  Model Memory ({precision}): {mem_est['model_mb']:.1f} MB")
    print(f"  Optimizer Memory (Adam):{mem_est['optimizer_mb']:.1f} MB")
    print(f"  Gradients Memory:       {mem_est['gradient_mb']:.1f} MB")
    print(f"  Static Training Memory: ~{mem_est['total_mb']:.1f} MB (~{mem_est['total_mb']/1024:.2f} GB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
