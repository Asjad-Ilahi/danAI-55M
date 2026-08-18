"""
Model Fusion & Weight Merging Engine for Small Language Models (SLM).

Techniques:
- Linear Task Arithmetic / Model Soup (w_fused = (1 - a)*w1 + a*w2)
- Spherical Linear Interpolation (SLERP) for orthogonal subspace preservation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def slerp(p0: torch.Tensor, p1: torch.Tensor, val: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two parameter tensors."""
    p0_flat = p0.view(-1).float()
    p1_flat = p1.view(-1).float()

    dot = torch.sum(p0_flat * p1_flat) / (torch.norm(p0_flat) * torch.norm(p1_flat) + eps)
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    if torch.abs(sin_theta) < eps:
        return (1.0 - val) * p0 + val * p1

    s0 = torch.sin((1.0 - val) * theta) / sin_theta
    s1 = torch.sin(val * theta) / sin_theta

    return (s0 * p0 + s1 * p1).type_as(p0)


def merge_checkpoints(
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    alpha: float = 0.5,
    method: str = "linear",
):
    print("=" * 80)
    print("       🧬 SLM WEIGHT FUSION & MODEL MERGER")
    print("=" * 80)
    print(f"  • Model A (Base/Everyday):   {model_a_path}")
    print(f"  • Model B (Tools/Agentic):   {model_b_path}")
    print(f"  • Alpha (Blend Weight):      {alpha} ({method.upper()})")
    print(f"  • Output Target:             {output_path}")
    print("-" * 80)

    ckpt_a = torch.load(model_a_path, map_location="cpu", weights_only=True)
    ckpt_b = torch.load(model_b_path, map_location="cpu", weights_only=True)

    w_a = ckpt_a.get("ema_weights") or ckpt_a.get("model_state_dict") or ckpt_a
    w_b = ckpt_b.get("ema_weights") or ckpt_b.get("model_state_dict") or ckpt_b

    merged_weights: Dict[str, torch.Tensor] = {}

    for k in w_a.keys():
        if k in w_b:
            tensor_a = w_a[k]
            tensor_b = w_b[k]

            if method == "slerp" and tensor_a.dim() >= 2:
                merged_weights[k] = slerp(tensor_a, tensor_b, alpha)
            else:
                # Linear Task Arithmetic
                merged_weights[k] = (1.0 - alpha) * tensor_a + alpha * tensor_b
        else:
            merged_weights[k] = w_a[k]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "ema_weights": merged_weights,
        "model_state_dict": merged_weights,
        "fusion_metadata": {
            "model_a": model_a_path,
            "model_b": model_b_path,
            "alpha": alpha,
            "method": method,
        }
    }

    torch.save(save_dict, out_file)
    print(f"\n[OK] Successfully merged {len(merged_weights)} weight tensors to {out_file}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse two SLM checkpoints into a single unified model")
    parser.add_argument("--model-a", type=str, default="experiments/exp_015_clean_reasoning/checkpoints/best.pt", help="Model A path (Everyday/Reasoning)")
    parser.add_argument("--model-b", type=str, default="experiments/exp_016_agentic_slm/checkpoints/best.pt", help="Model B path (Agentic Tools)")
    parser.add_argument("--output", type=str, default="experiments/exp_018_fused_slm/checkpoints/best.pt", help="Output checkpoint path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Interpolation weight for Model B (0.0 to 1.0)")
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "slerp"], help="Fusion method (linear or slerp)")
    args = parser.parse_args()

    merge_checkpoints(args.model_a, args.model_b, args.output, alpha=args.alpha, method=args.method)
