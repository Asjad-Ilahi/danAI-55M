"""
Comprehensive GPU Training Benchmark for 54.5M SLM on NVIDIA CUDA.
Measures throughput (tokens/sec), step latency (ms), VRAM allocation & peak (MB/GB),
and projects training durations for continuation runs and new token targets.
"""

import gc
import os
import sys
import time
import json
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

from src.utils.config import Config
from src.utils.device import get_device, select_precision, check_bf16_support
from src.model.gpt import CausalLM
from src.model.ema import EMAModel
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionManager


def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_vram_gb(device: torch.device):
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        return allocated, reserved, peak_allocated
    return 0.0, 0.0, 0.0


def benchmark_step_round(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    precision_mgr: PrecisionManager,
    device: torch.device,
    micro_batch: int,
    seq_len: int,
    grad_accum_steps: int,
    vocab_size: int,
    ema_model: Optional[EMAModel] = None,
    warmup_steps: int = 3,
    active_steps: int = 10,
) -> Dict[str, Any]:
    """
    Benchmarks a full training round including micro-batches, gradient accumulation,
    gradient clipping, optimizer step, and EMA update.
    """
    clean_gpu_memory()
    model.train()
    
    # Pre-generate synthetic batches on GPU
    batches = []
    for _ in range(grad_accum_steps):
        x = torch.randint(0, vocab_size, (micro_batch, seq_len), dtype=torch.long, device=device)
        y = torch.randint(0, vocab_size, (micro_batch, seq_len), dtype=torch.long, device=device)
        batches.append((x, y))

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        for x, y in batches:
            with precision_mgr.get_autocast_context():
                _, loss, _ = model(x, targets=y)
                scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize()

    clean_gpu_memory()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Timed benchmark
    start_time = time.perf_counter()
    for _ in range(active_steps):
        optimizer.zero_grad(set_to_none=True)
        for x, y in batches:
            with precision_mgr.get_autocast_context():
                _, loss, _ = model(x, targets=y)
                scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    
    allocated_gb, reserved_gb, peak_allocated_gb = get_gpu_vram_gb(device)
    
    tokens_per_optimizer_step = micro_batch * seq_len * grad_accum_steps
    total_tokens_processed = tokens_per_optimizer_step * active_steps
    tokens_per_sec = total_tokens_processed / elapsed
    ms_per_optimizer_step = (elapsed / active_steps) * 1000.0
    ms_per_micro_batch = ms_per_optimizer_step / grad_accum_steps

    return {
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "grad_accum_steps": grad_accum_steps,
        "tokens_per_step": tokens_per_optimizer_step,
        "tokens_per_sec": tokens_per_sec,
        "step_time_ms": ms_per_optimizer_step,
        "micro_batch_time_ms": ms_per_micro_batch,
        "vram_allocated_gb": allocated_gb,
        "vram_reserved_gb": reserved_gb,
        "vram_peak_gb": peak_allocated_gb,
        "status": "OK",
    }


def main():
    print("=" * 80)
    print("  54.5M SLM GPU TRAINING BENCHMARK & HARDWARE PROFILER")
    print("=" * 80)

    device = get_device()
    precision = select_precision(device, "auto")
    precision_mgr = PrecisionManager(device, precision)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A"
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3) if device.type == "cuda" else 0.0

    print(f"  Device:           {device} ({gpu_name})")
    print(f"  Total VRAM:       {total_vram:.2f} GB")
    print(f"  PyTorch:          {torch.__version__}")
    print(f"  CUDA Version:     {torch.version.cuda}")
    print(f"  Precision:        {precision}")
    print("=" * 80)

    config = Config.from_yaml("configs/model.yaml")
    vocab_size = config.model.vocab_size

    # Section 1: Checkpointing ON vs OFF comparison
    print("\n[1/4] Benchmarking Gradient Checkpointing Impact (Micro-batch=4, Seq=1024)...")
    ckpt_results = []
    for gc_flag in [False, True]:
        cfg_dict = config.to_dict()
        cfg_dict["training"] = {"gradient_checkpointing": gc_flag}
        test_cfg = Config(cfg_dict)
        model = CausalLM(test_cfg).to(device)
        if gc_flag:
            model.enable_gradient_checkpointing()
        optimizer = create_optimizer(model, learning_rate=3e-4)
        
        res = benchmark_step_round(
            model=model,
            optimizer=optimizer,
            precision_mgr=precision_mgr,
            device=device,
            micro_batch=4,
            seq_len=1024,
            grad_accum_steps=8,
            vocab_size=vocab_size,
        )
        res["checkpointing"] = gc_flag
        ckpt_results.append(res)
        del model, optimizer
        clean_gpu_memory()

    # Section 2: Micro-Batch Scaling Profile (Seq=1024, no checkpointing)
    print("\n[2/4] Benchmarking Micro-Batch Scaling (Micro-batches: 1, 2, 4, 8, 16, 24, 32)...")
    batch_results = []
    for mb in [1, 2, 4, 8, 16, 24, 32]:
        try:
            cfg_dict = config.to_dict()
            cfg_dict["training"] = {"gradient_checkpointing": False}
            test_cfg = Config(cfg_dict)
            model = CausalLM(test_cfg).to(device)
            optimizer = create_optimizer(model, learning_rate=3e-4)
            
            res = benchmark_step_round(
                model=model,
                optimizer=optimizer,
                precision_mgr=precision_mgr,
                device=device,
                micro_batch=mb,
                seq_len=1024,
                grad_accum_steps=1,
                vocab_size=vocab_size,
            )
            batch_results.append(res)
            print(f"  ✓ Micro-batch {mb:2d}: {res['tokens_per_sec']:8,.0f} tok/s | Peak VRAM: {res['vram_peak_gb']:.2f} GB | Step: {res['step_time_ms']:.1f} ms")
        except RuntimeError as e:
            print(f"  ✗ Micro-batch {mb:2d} failed: {e}")
            break
        finally:
            del model, optimizer
            clean_gpu_memory()

    # Section 3: Context Length Scaling Profile (Seq: 512, 1024, 2048)
    print("\n[3/4] Benchmarking Context Length Scaling...")
    context_results = []
    for seq_len in [512, 1024, 2048]:
        for mb in [4, 8, 16]:
            try:
                cfg_dict = config.to_dict()
                cfg_dict["model"]["max_seq_len"] = seq_len
                cfg_dict["training"] = {"gradient_checkpointing": False}
                test_cfg = Config(cfg_dict)
                model = CausalLM(test_cfg).to(device)
                optimizer = create_optimizer(model, learning_rate=3e-4)
                
                res = benchmark_step_round(
                    model=model,
                    optimizer=optimizer,
                    precision_mgr=precision_mgr,
                    device=device,
                    micro_batch=mb,
                    seq_len=seq_len,
                    grad_accum_steps=1,
                    vocab_size=vocab_size,
                )
                context_results.append(res)
                print(f"  ✓ Context {seq_len:4d} | Micro-batch {mb:2d}: {res['tokens_per_sec']:8,.0f} tok/s | Peak VRAM: {res['vram_peak_gb']:.2f} GB")
            except RuntimeError as e:
                print(f"  ✗ Context {seq_len:4d} | Micro-batch {mb:2d} failed: {e}")
                break
            finally:
                del model, optimizer
                clean_gpu_memory()

    # Section 4: Full Training Round Configurations (Simulating 32,768 & 65,536 tokens/step effective batches)
    print("\n[4/4] Benchmarking Full Training Rounds with EMA & Gradient Accumulation...")
    effective_batch_configs = [
        # (micro_batch, grad_accum, seq_len, desc)
        (4, 8, 1024, "Standard Baseline (exp_008 default: 4×8×1024 = 32,768 tokens)"),
        (8, 4, 1024, "High Throughput 1 (8×4×1024 = 32,768 tokens)"),
        (16, 2, 1024, "High Throughput 2 (16×2×1024 = 32,768 tokens)"),
        (32, 1, 1024, "Maximum GPU Saturation (32×1×1024 = 32,768 tokens)"),
        (16, 4, 1024, "Large Effective Batch (16×4×1024 = 65,536 tokens)"),
        (32, 2, 1024, "Large Effective Batch (32×2×1024 = 65,536 tokens)"),
    ]
    
    full_round_results = []
    for mb, accum, sl, desc in effective_batch_configs:
        try:
            cfg_dict = config.to_dict()
            cfg_dict["training"] = {"gradient_checkpointing": False, "ema_decay": 0.999}
            test_cfg = Config(cfg_dict)
            model = CausalLM(test_cfg).to(device)
            ema = EMAModel(model, decay=0.999)
            optimizer = create_optimizer(model, learning_rate=3e-4)
            
            res = benchmark_step_round(
                model=model,
                optimizer=optimizer,
                precision_mgr=precision_mgr,
                device=device,
                micro_batch=mb,
                seq_len=sl,
                grad_accum_steps=accum,
                vocab_size=vocab_size,
                ema_model=ema,
                warmup_steps=3,
                active_steps=10,
            )
            res["description"] = desc
            full_round_results.append(res)
            print(f"  ✓ {desc}")
            print(f"    → Throughput: {res['tokens_per_sec']:8,.0f} tok/s | Step Latency: {res['step_time_ms']:6.1f} ms | Peak VRAM: {res['vram_peak_gb']:.2f} GB ({res['vram_peak_gb']/total_vram*100:.1f}%)")
        except RuntimeError as e:
            print(f"  ✗ {desc} failed: {e}")
        finally:
            del model, ema, optimizer
            clean_gpu_memory()

    # Save comprehensive results to JSON
    output_data = {
        "device": str(device),
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": str(precision),
        "checkpointing_comparison": ckpt_results,
        "micro_batch_scaling": batch_results,
        "context_scaling": context_results,
        "full_training_rounds": full_round_results,
    }
    
    output_path = Path("experiments/gpu_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved raw benchmark data to {output_path}")


if __name__ == "__main__":
    main()
