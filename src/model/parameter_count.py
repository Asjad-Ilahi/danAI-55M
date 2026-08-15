"""
Parameter count analysis and architecture search.

Analytically computes parameter counts per component, runs a search across
the configuration space, and cross-validates against an instantiated model.

Per §4: search is biased toward depth (more layers) within the 65-85M budget.
"""

import math
from typing import Optional
from src.utils.config import resolve_intermediate_size


def compute_parameter_count(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    intermediate_size: int | str = 'auto',
    tie_embeddings: bool = True,
    use_bias: bool = False,
) -> dict:
    """
    Analytically compute parameter counts for each component.
    
    Components:
    - Token embedding: vocab_size × hidden_size
    - Per-layer attention (GQA):
        Q: hidden_size × (num_query_heads × head_dim)
        K: hidden_size × (num_kv_heads × head_dim)
        V: hidden_size × (num_kv_heads × head_dim)
        O: (num_query_heads × head_dim) × hidden_size
    - Per-layer SwiGLU MLP:
        gate: hidden_size × intermediate_size
        up:   hidden_size × intermediate_size
        down: intermediate_size × hidden_size
    - Per-layer norms: 2 × hidden_size (two RMSNorm weight vectors)
    - Final norm: hidden_size
    - LM head: 0 if tied, else vocab_size × hidden_size
    
    Returns:
        Dictionary with detailed parameter breakdown.
    """
    inter = resolve_intermediate_size(hidden_size, intermediate_size)
    head_dim = hidden_size // num_query_heads
    kv_dim = num_kv_heads * head_dim
    
    # Bias parameters per linear layer (if bias=True)
    bias_mult = 1 if use_bias else 0
    
    # Embedding
    embedding_params = vocab_size * hidden_size
    
    # Per-layer attention
    q_params = hidden_size * hidden_size + bias_mult * hidden_size  # Q proj
    k_params = hidden_size * kv_dim + bias_mult * kv_dim            # K proj
    v_params = hidden_size * kv_dim + bias_mult * kv_dim            # V proj
    o_params = hidden_size * hidden_size + bias_mult * hidden_size  # O proj
    attn_params_per_layer = q_params + k_params + v_params + o_params
    
    # Per-layer MLP (SwiGLU: 3 projections)
    gate_params = hidden_size * inter + bias_mult * inter
    up_params = hidden_size * inter + bias_mult * inter
    down_params = inter * hidden_size + bias_mult * hidden_size
    mlp_params_per_layer = gate_params + up_params + down_params
    
    # Per-layer norms (2 × RMSNorm weight vectors)
    norm_params_per_layer = 2 * hidden_size
    
    # Total per layer
    layer_params = attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer
    
    # All layers
    all_layers_params = num_layers * layer_params
    
    # Final norm
    final_norm_params = hidden_size
    
    # LM head
    lm_head_params = 0 if tie_embeddings else (vocab_size * hidden_size + bias_mult * vocab_size)
    
    # Total
    total = embedding_params + all_layers_params + final_norm_params + lm_head_params
    
    return {
        'embedding': embedding_params,
        'embedding_pct': embedding_params / total * 100,
        'attn_per_layer': attn_params_per_layer,
        'mlp_per_layer': mlp_params_per_layer,
        'norm_per_layer': norm_params_per_layer,
        'layer_total': layer_params,
        'all_layers': all_layers_params,
        'all_layers_pct': all_layers_params / total * 100,
        'final_norm': final_norm_params,
        'lm_head': lm_head_params,
        'total': total,
        # Config echo
        'config': {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_query_heads': num_query_heads,
            'num_kv_heads': num_kv_heads,
            'intermediate_size': inter,
            'head_dim': head_dim,
            'tie_embeddings': tie_embeddings,
            'use_bias': use_bias,
        },
    }


def verify_against_model(model, expected_total: int) -> bool:
    """
    Cross-validate analytical count against an instantiated model.
    
    Args:
        model: An instantiated PyTorch model
        expected_total: The analytically computed parameter count
    
    Returns:
        True if counts match exactly
    
    Raises:
        ValueError if counts don't match
    """
    actual_total = sum(p.numel() for p in model.parameters())
    
    if actual_total != expected_total:
        raise ValueError(
            f"Parameter count mismatch!\n"
            f"  Analytical: {expected_total:,}\n"
            f"  Actual:     {actual_total:,}\n"
            f"  Difference: {abs(actual_total - expected_total):,}\n"
            f"Check the analytical formula or model implementation."
        )
    
    return True


def run_architecture_search(
    target_min: int = 65_000_000,
    target_max: int = 85_000_000,
    target_center: int = 75_000_000,
) -> list[dict]:
    """
    Search the configuration space for architectures in the target parameter range.
    
    Search space per §4:
    - layers: 12, 14, 16, 18, 20
    - hidden_size: 384, 448, 512
    - query_heads: 8, 12
    - kv_heads: 2, 4
    - vocab_size: 16384, 24576, 32768
    
    Selection priority:
    1. Land in target range
    2. Maximize depth
    3. Memory/attention efficiency
    4. Sensible head_dim (~48-64)
    
    Returns:
        List of valid configurations sorted by (-depth, closeness_to_center)
    """
    results = []
    
    for num_layers in [12, 14, 16, 18, 20]:
        for hidden_size in [384, 448, 512]:
            for num_query_heads in [8, 12]:
                for num_kv_heads in [2, 4]:
                    for vocab_size in [16384, 24576, 32768]:
                        # Skip invalid combinations
                        if num_query_heads % num_kv_heads != 0:
                            continue
                        if hidden_size % num_query_heads != 0:
                            continue
                        
                        head_dim = hidden_size // num_query_heads
                        
                        # Check head_dim is sensible (~48-64)
                        if head_dim < 32 or head_dim > 128:
                            continue
                        
                        counts = compute_parameter_count(
                            vocab_size=vocab_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            num_query_heads=num_query_heads,
                            num_kv_heads=num_kv_heads,
                            intermediate_size='auto',
                            tie_embeddings=True,
                            use_bias=False,
                        )
                        
                        total = counts['total']
                        if target_min <= total <= target_max:
                            results.append({
                                **counts['config'],
                                'total_params': total,
                                'embedding_pct': counts['embedding_pct'],
                                'transformer_pct': counts['all_layers_pct'],
                                'depth_score': num_layers,
                                'distance_from_center': abs(total - target_center),
                            })
    
    # Sort: maximize depth, then minimize distance from center
    results.sort(key=lambda x: (-x['depth_score'], x['distance_from_center']))
    
    return results


def print_search_report(results: list[dict]) -> None:
    """Print a formatted table of architecture search results."""
    print(f"\n{'=' * 100}")
    print(f"ARCHITECTURE SEARCH RESULTS — {len(results)} valid configurations in 65-85M range")
    print(f"{'=' * 100}")
    print(
        f"{'L':>3} {'H':>4} {'QH':>3} {'KVH':>3} {'HD':>3} "
        f"{'Inter':>5} {'Vocab':>6} {'Total':>12} "
        f"{'Emb%':>5} {'Trans%':>6} {'Depth':>5}"
    )
    print('-' * 100)
    
    for r in results:
        marker = ' ←' if r == results[0] else ''
        print(
            f"{r['num_layers']:3d} {r['hidden_size']:4d} "
            f"{r['num_query_heads']:3d} {r['num_kv_heads']:3d} "
            f"{r['head_dim']:3d} {r['intermediate_size']:5d} "
            f"{r['vocab_size']:6d} {r['total_params']:12,} "
            f"{r['embedding_pct']:5.1f} {r['transformer_pct']:6.1f} "
            f"{r['depth_score']:5d}{marker}"
        )
    
    if results:
        best = results[0]
        print(f"\n{'=' * 100}")
        print(f"RECOMMENDED: L={best['num_layers']}, H={best['hidden_size']}, "
              f"QH={best['num_query_heads']}, KVH={best['num_kv_heads']}, "
              f"V={best['vocab_size']} → {best['total_params']:,} params")
        print(f"  Depth score: {best['depth_score']} (maximum)")
        print(f"  Embedding: {best['embedding_pct']:.1f}% of budget")
        print(f"  Transformer: {best['transformer_pct']:.1f}% of budget")
        print(f"{'=' * 100}\n")


def print_parameter_breakdown(counts: dict) -> None:
    """Print a detailed parameter breakdown."""
    c = counts['config']
    print(f"\n{'=' * 60}")
    print(f"PARAMETER BREAKDOWN")
    print(f"{'=' * 60}")
    print(f"  Architecture:")
    print(f"    Layers:          {c['num_layers']}")
    print(f"    Hidden size:     {c['hidden_size']}")
    print(f"    Query heads:     {c['num_query_heads']}")
    print(f"    KV heads:        {c['num_kv_heads']} (GQA ratio: {c['num_query_heads']//c['num_kv_heads']}:1)")
    print(f"    Head dim:        {c['head_dim']}")
    print(f"    Intermediate:    {c['intermediate_size']}")
    print(f"    Vocab size:      {c['vocab_size']}")
    print(f"    Tied embeddings: {c['tie_embeddings']}")
    print()
    print(f"  Parameters:")
    print(f"    Embedding:       {counts['embedding']:>12,} ({counts['embedding_pct']:.1f}%)")
    print(f"    Per layer:")
    print(f"      Attention:     {counts['attn_per_layer']:>12,}")
    print(f"      MLP:           {counts['mlp_per_layer']:>12,}")
    print(f"      Norms:         {counts['norm_per_layer']:>12,}")
    print(f"      Total:         {counts['layer_total']:>12,}")
    print(f"    All layers (×{c['num_layers']}): {counts['all_layers']:>12,} ({counts['all_layers_pct']:.1f}%)")
    print(f"    Final norm:      {counts['final_norm']:>12,}")
    print(f"    LM head:         {counts['lm_head']:>12,} ({'tied' if c['tie_embeddings'] else 'separate'})")
    print(f"    {'─' * 40}")
    print(f"    TOTAL:           {counts['total']:>12,}")
    print(f"{'=' * 60}\n")
