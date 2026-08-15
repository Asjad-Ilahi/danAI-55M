import gc
import os
import time
import platform
import subprocess
from dataclasses import dataclass

import numpy as np
import torch

from src.utils.config import Config
from src.utils.device import get_device, select_precision
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer
from src.training.precision import PrecisionManager


# ============================================================
# Configuration
# ============================================================

MODEL_CONFIG = "configs/model.yaml"
TRAIN_CONFIG = "configs/train.yaml"

SEQ_LENGTHS = [512, 1024, 2048]
MICRO_BATCHES = [1, 2, 4, 8]

WARMUP_STEPS = 3
BENCHMARK_STEPS = 10

LEARNING_RATE = 3e-4

# Do not let the benchmark consume the entire machine.
# Leave approximately this much memory for macOS/background apps.
SAFETY_MEMORY_GB = 4.0

# If available memory falls below this amount, mark configuration unsafe.
MIN_AVAILABLE_MEMORY_GB = 2.0


# ============================================================
# Data structures
# ============================================================

@dataclass
class BenchmarkResult:
    checkpointing: bool
    micro_batch: int
    seq_length: int

    tokens_per_second: float
    step_time_ms: float

    system_memory_before_gb: float
    system_memory_after_gb: float

    peak_system_memory_gb: float | None
    mps_peak_allocated_gb: float | None
    mps_peak_reserved_gb: float | None

    status: str
    error: str | None = None


# ============================================================
# Memory helpers
# ============================================================

def get_system_memory():
    """
    Returns:
        total_gb
        available_gb
        used_gb
    """

    try:
        import psutil

        vm = psutil.virtual_memory()

        return (
            vm.total / (1024 ** 3),
            vm.available / (1024 ** 3),
            vm.used / (1024 ** 3),
        )

    except ImportError:
        return None, None, None


def get_process_memory_gb():
    """
    Python process RSS.

    This is NOT the complete MPS memory usage.
    It is included only as an additional diagnostic.
    """

    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    except ImportError:
        return None


def get_mps_memory():
    """
    Returns MPS allocator statistics where available.

    These numbers should be treated as allocator statistics,
    not total macOS unified-memory consumption.
    """

    if not hasattr(torch, "mps"):
        return None, None

    try:
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
    except Exception:
        allocated = None

    try:
        driver_allocated = torch.mps.driver_allocated_memory() / (1024 ** 3)
    except Exception:
        driver_allocated = None

    return allocated, driver_allocated


def reset_mps_memory():

    if hasattr(torch, "mps"):

        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        try:
            torch.mps.synchronize()
        except Exception:
            pass


def cleanup():

    gc.collect()

    reset_mps_memory()

    gc.collect()


# ============================================================
# Device synchronization
# ============================================================

def synchronize(device):

    if device.type == "mps":

        try:
            torch.mps.synchronize()
        except Exception:
            pass

    elif device.type == "cuda":

        torch.cuda.synchronize()


# ============================================================
# Precision
# ============================================================

def create_precision_manager(device):

    precision = select_precision(device, "auto")

    manager = PrecisionManager(device, precision)

    return precision, manager


# ============================================================
# Model creation
# ============================================================

def create_model(config, device, gradient_checkpointing):

    model_config = config.to_dict()

    model_config["training"]["gradient_checkpointing"] = (
        gradient_checkpointing
    )

    model = CausalLM(Config(model_config)).to(device)

    if gradient_checkpointing:

        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

    else:

        if hasattr(model, "disable_gradient_checkpointing"):
            model.disable_gradient_checkpointing()

    return model


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Safe random batch
# ============================================================

def create_batch(
    vocab_size,
    batch_size,
    seq_length,
    device,
):

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        dtype=torch.long,
        device=device,
    )

    y = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        dtype=torch.long,
        device=device,
    )

    return x, y


# ============================================================
# Single benchmark
# ============================================================

def benchmark_configuration(
    config,
    device,
    precision_manager,
    checkpointing,
    micro_batch,
    seq_length,
):

    cleanup()

    total_before, available_before, used_before = get_system_memory()

    try:

        # ----------------------------------------------------
        # Check available system memory before starting
        # ----------------------------------------------------

        if available_before is not None:

            if available_before < MIN_AVAILABLE_MEMORY_GB:

                return BenchmarkResult(
                    checkpointing,
                    micro_batch,
                    seq_length,
                    0,
                    0,
                    available_before,
                    available_before,
                    None,
                    None,
                    None,
                    "SKIPPED",
                    "Insufficient system memory before test",
                )

        # ----------------------------------------------------
        # Configure sequence length and batch size
        # ----------------------------------------------------

        cfg = config.to_dict()

        cfg["model"]["max_seq_len"] = seq_length
        cfg["training"]["max_seq_len"] = seq_length

        cfg["training"]["micro_batch_size"] = micro_batch

        cfg["training"]["gradient_checkpointing"] = checkpointing

        test_config = Config(cfg)

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = CausalLM(test_config).to(device)

        if checkpointing:

            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()

        else:

            if hasattr(model, "disable_gradient_checkpointing"):
                model.disable_gradient_checkpointing()

        model.train()

        optimizer = create_optimizer(
            model,
            learning_rate=LEARNING_RATE,
        )

        # ----------------------------------------------------
        # Create random training data
        # ----------------------------------------------------

        vocab_size = test_config.model.vocab_size

        x, y = create_batch(
            vocab_size=vocab_size,
            batch_size=micro_batch,
            seq_length=seq_length,
            device=device,
        )

        # ----------------------------------------------------
        # Clear MPS cache before warmup
        # ----------------------------------------------------

        synchronize(device)

        cleanup()

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        for _ in range(WARMUP_STEPS):

            optimizer.zero_grad(set_to_none=True)

            with precision_manager.get_autocast_context():

                _, loss, _ = model(
                    x,
                    targets=y,
                )

            loss.backward()

            optimizer.step()

            synchronize(device)

        # ----------------------------------------------------
        # Reset peak MPS statistics
        # ----------------------------------------------------

        if device.type == "mps":

            try:
                torch.mps.reset_peak_memory_stats()
            except Exception:
                pass

        # ----------------------------------------------------
        # Benchmark
        # ----------------------------------------------------

        synchronize(device)

        start_time = time.perf_counter()

        available_samples = []

        for step in range(BENCHMARK_STEPS):

            optimizer.zero_grad(set_to_none=True)

            with precision_manager.get_autocast_context():

                _, loss, _ = model(
                    x,
                    targets=y,
                )

            loss.backward()

            optimizer.step()

            synchronize(device)

            # Record system memory after each step
            _, available_now, _ = get_system_memory()

            if available_now is not None:
                available_samples.append(available_now)

        synchronize(device)

        elapsed = time.perf_counter() - start_time

        # ----------------------------------------------------
        # Calculate throughput
        # ----------------------------------------------------

        tokens_per_step = (
            micro_batch *
            seq_length
        )

        tokens_per_second = (
            tokens_per_step *
            BENCHMARK_STEPS /
            elapsed
        )

        step_time_ms = (
            elapsed /
            BENCHMARK_STEPS *
            1000
        )

        # ----------------------------------------------------
        # Memory after benchmark
        # ----------------------------------------------------

        total_after, available_after, used_after = (
            get_system_memory()
        )

        process_memory = get_process_memory_gb()

        mps_allocated, mps_reserved = get_mps_memory()

        if available_samples:

            peak_system_memory = min(
                available_samples
            )

        else:

            peak_system_memory = None

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        status = "OK"

        if available_after is not None:

            if available_after < MIN_AVAILABLE_MEMORY_GB:

                status = "LOW_MEMORY"

        # ----------------------------------------------------
        # Print detailed result
        # ----------------------------------------------------

        if available_before is not None and available_after is not None:
            print(
                f"\n"
                f"Checkpointing : {checkpointing}\n"
                f"Micro-batch   : {micro_batch}\n"
                f"Sequence      : {seq_length}\n"
                f"Tokens/step   : {tokens_per_step:,}\n"
                f"Tokens/sec    : {tokens_per_second:,.0f}\n"
                f"Step time     : {step_time_ms:,.1f} ms\n"
                f"Available RAM : {available_before:.2f} GB -> {available_after:.2f} GB"
            )
        else:
            print(
                f"\n"
                f"Checkpointing : {checkpointing}\n"
                f"Micro-batch   : {micro_batch}\n"
                f"Sequence      : {seq_length}\n"
                f"Tokens/step   : {tokens_per_step:,}\n"
                f"Tokens/sec    : {tokens_per_second:,.0f}\n"
                f"Step time     : {step_time_ms:,.1f} ms"
            )

        if process_memory is not None:

            print(
                f"Process RSS   : {process_memory:.2f} GB"
            )

        if mps_allocated is not None:

            print(
                f"MPS allocated : {mps_allocated:.2f} GB"
            )

        if mps_reserved is not None:

            print(
                f"MPS driver    : {mps_reserved:.2f} GB"
            )

        print(
            f"Status        : {status}"
        )

        return BenchmarkResult(
            checkpointing=checkpointing,
            micro_batch=micro_batch,
            seq_length=seq_length,
            tokens_per_second=tokens_per_second,
            step_time_ms=step_time_ms,
            system_memory_before_gb=available_before,
            system_memory_after_gb=available_after,
            peak_system_memory_gb=peak_system_memory,
            mps_peak_allocated_gb=mps_allocated,
            mps_peak_reserved_gb=mps_reserved,
            status=status,
        )

    except RuntimeError as e:

        message = str(e)

        print(
            f"\n"
            f"Checkpointing : {checkpointing}\n"
            f"Micro-batch   : {micro_batch}\n"
            f"Sequence      : {seq_length}\n"
            f"STATUS        : OOM / ERROR\n"
            f"Error         : {message[:500]}"
        )

        return BenchmarkResult(
            checkpointing=checkpointing,
            micro_batch=micro_batch,
            seq_length=seq_length,
            tokens_per_second=0,
            step_time_ms=0,
            system_memory_before_gb=available_before,
            system_memory_after_gb=get_system_memory()[1],
            peak_system_memory_gb=None,
            mps_peak_allocated_gb=None,
            mps_peak_reserved_gb=None,
            status="OOM_OR_ERROR",
            error=message,
        )

    finally:

        try:
            del x
            del y
        except Exception:
            pass

        try:
            del optimizer
            del model
        except Exception:
            pass

        cleanup()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("75M SLM TRAINING BENCHMARK")
    print("=" * 75)

    print(
        f"Python      : {platform.python_version()}"
    )

    print(
        f"PyTorch     : {torch.__version__}"
    )

    print(
        f"Platform    : {platform.platform()}"
    )

    print(
        f"Machine     : {platform.machine()}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    precision, precision_manager = (
        create_precision_manager(device)
    )

    print(
        f"Device      : {device}"
    )

    print(
        f"Precision   : {precision}"
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    total, available, used = get_system_memory()

    if total is not None:

        print(
            f"Total RAM   : {total:.2f} GB"
        )

        print(
            f"Available   : {available:.2f} GB"
        )

        print(
            f"Used        : {used:.2f} GB"
        )

    print()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    model_config = Config.from_yaml(
        MODEL_CONFIG
    )

    train_config = Config.from_yaml(
        TRAIN_CONFIG
    )

    combined = model_config.to_dict()

    combined.update(
        train_config.to_dict()
    )

    config = Config(combined)

    # --------------------------------------------------------
    # Create one model just to inspect parameter count
    # --------------------------------------------------------

    print("Inspecting model...")

    model = CausalLM(config)

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"Total parameters     : {total_params:,}"
    )

    print(
        f"Trainable parameters : {trainable_params:,}"
    )

    print(
        f"Approx FP32 weights  : "
        f"{total_params * 4 / (1024 ** 2):.1f} MB"
    )

    print(
        f"Approx FP16/BF16     : "
        f"{total_params * 2 / (1024 ** 2):.1f} MB"
    )

    del model

    cleanup()

    # --------------------------------------------------------
    # Benchmark matrix
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 75)
    print("STARTING BENCHMARK")
    print("=" * 75)

    # --------------------------------------------------------
    # First compare checkpointing ON/OFF at safe config
    # --------------------------------------------------------

    print()
    print("### CHECKPOINTING COMPARISON ###")

    for checkpointing in [True, False]:

        result = benchmark_configuration(
            config=config,
            device=device,
            precision_manager=precision_manager,
            checkpointing=checkpointing,
            micro_batch=1,
            seq_length=1024,
        )

        results.append(result)

    # --------------------------------------------------------
    # Batch scaling without checkpointing
    # --------------------------------------------------------

    print()
    print("### MICRO-BATCH SCALING ###")

    for batch in MICRO_BATCHES:

        result = benchmark_configuration(
            config=config,
            device=device,
            precision_manager=precision_manager,
            checkpointing=False,
            micro_batch=batch,
            seq_length=1024,
        )

        results.append(result)

        # Stop if we hit OOM.
        if result.status == "OOM_OR_ERROR":

            print(
                f"\nStopping batch scaling after "
                f"micro-batch={batch}."
            )

            break

    # --------------------------------------------------------
    # Context length scaling
    # --------------------------------------------------------

    print()
    print("### CONTEXT LENGTH SCALING ###")

    for seq_length in SEQ_LENGTHS:

        result = benchmark_configuration(
            config=config,
            device=device,
            precision_manager=precision_manager,
            checkpointing=False,
            micro_batch=1,
            seq_length=seq_length,
        )

        results.append(result)

        if result.status == "OOM_OR_ERROR":

            print(
                f"\nStopping context scaling after "
                f"seq_length={seq_length}."
            )

            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"{'CKPT':<7}"
        f"{'BATCH':<8}"
        f"{'SEQ':<8}"
        f"{'TOK/S':<12}"
        f"{'STEP(ms)':<12}"
        f"{'RAM BEFORE':<14}"
        f"{'RAM AFTER':<13}"
        f"{'STATUS':<15}"
    )

    print("-" * 100)

    for r in results:

        before = (
            f"{r.system_memory_before_gb:.2f} GB"
            if r.system_memory_before_gb is not None
            else "N/A"
        )

        after = (
            f"{r.system_memory_after_gb:.2f} GB"
            if r.system_memory_after_gb is not None
            else "N/A"
        )

        print(
            f"{str(r.checkpointing):<7}"
            f"{r.micro_batch:<8}"
            f"{r.seq_length:<8}"
            f"{r.tokens_per_second:<12,.0f}"
            f"{r.step_time_ms:<12,.1f}"
            f"{before:<14}"
            f"{after:<13}"
            f"{r.status:<15}"
        )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    successful = [
        r
        for r in results
        if r.status == "OK"
    ]

    if successful:

        best = max(
            successful,
            key=lambda r: r.tokens_per_second
        )

        print()
        print("=" * 75)
        print("FASTEST SAFE CONFIGURATION")
        print("=" * 75)

        print(
            f"Gradient checkpointing : "
            f"{best.checkpointing}"
        )

        print(
            f"Micro-batch size       : "
            f"{best.micro_batch}"
        )

        print(
            f"Sequence length        : "
            f"{best.seq_length}"
        )

        print(
            f"Tokens/sec             : "
            f"{best.tokens_per_second:,.0f}"
        )

        print(
            f"Step time              : "
            f"{best.step_time_ms:,.1f} ms"
        )

        print()
        print(
            "IMPORTANT: choose a configuration with "
            "comfortable memory headroom, not merely "
            "the highest tokens/sec."
        )

    print()
    print("=" * 75)
    print("BENCHMARK COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
