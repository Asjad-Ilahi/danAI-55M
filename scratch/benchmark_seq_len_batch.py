import time
import torch
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer

def benchmark_sequence_and_batch_settings():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("==========================================================================")
    print("  SEQUENCE LENGTH & BATCH SIZE PERFORMANCE MATRIX BENCHMARK (54.5M Model)")
    print("==========================================================================")
    print(f"Device: {device} | PyTorch: {torch.__version__}\n")
    
    cfg = Config.from_yaml("configs/model.yaml")
    
    # Test combinations: (seq_len, micro_batch_size, grad_accum_steps)
    # Keeping target effective step size fixed around ~32,768 tokens where possible
    test_configs = [
        (512, 2, 32),    # Effective batch: 2 * 32 * 512 = 32,768 tokens
        (512, 4, 16),    # Effective batch: 4 * 16 * 512 = 32,768 tokens
        (512, 8, 8),     # Effective batch: 8 * 8 * 512  = 32,768 tokens
        (1024, 1, 32),   # Effective batch: 1 * 32 * 1024 = 32,768 tokens
        (1024, 2, 16),   # Effective batch: 2 * 16 * 1024 = 32,768 tokens (Default)
        (1024, 4, 8),    # Effective batch: 4 * 8 * 1024  = 32,768 tokens
        (2048, 1, 16),   # Effective batch: 1 * 16 * 2048 = 32,768 tokens (2048 Context)
        (2048, 2, 8),    # Effective batch: 2 * 8 * 2048  = 32,768 tokens (2048 Context)
    ]
    
    print(f"{'Seq Len':^10} | {'Micro BSize':^12} | {'Accum Steps':^12} | {'Step Time':^12} | {'Throughput':^16} | {'Status'}")
    print("-" * 85)
    
    for seq_len, micro_bsize, accum_steps in test_configs:
        cfg._data['model']['max_seq_len'] = seq_len
        model = CausalLM(cfg.model).to(device)
        model.train()
        optimizer = create_optimizer(model, learning_rate=3e-4)
        
        try:
            x = torch.randint(0, 32768, (micro_bsize, seq_len), device=device)
            y = torch.randint(0, 32768, (micro_bsize, seq_len), device=device)
            
            # Warmup pass
            for _ in range(2):
                optimizer.zero_grad()
                logits, loss, _ = model(x, targets=y)
                (loss / accum_steps).backward()
            
            # Benchmark 1 full optimizer step
            start = time.time()
            optimizer.zero_grad()
            for _ in range(accum_steps):
                logits, loss, _ = model(x, targets=y)
                (loss / accum_steps).backward()
            optimizer.step()
            
            elapsed = time.time() - start
            total_tokens = micro_bsize * accum_steps * seq_len
            tok_sec = total_tokens / elapsed
            
            print(f"{seq_len:^10} | {micro_bsize:^12} | {accum_steps:^12} | {elapsed:^11.2f}s | {tok_sec:^14.0f} tok/s | OK")
        except Exception as e:
            print(f"{seq_len:^10} | {micro_bsize:^12} | {accum_steps:^12} | {'N/A':^12} | {'N/A':^16} | FAILED: {str(e)[:25]}")
            
    print("=" * 85)

if __name__ == "__main__":
    benchmark_sequence_and_batch_settings()
