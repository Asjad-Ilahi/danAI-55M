import time
import torch
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer

def benchmark_configs():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 20 layers
    cfg_20 = Config.from_yaml("configs/model.yaml")
    
    # 12 layers
    cfg_12 = Config.from_yaml("configs/model.yaml")
    cfg_12._data['model']['num_layers'] = 12
    
    # 10 layers
    cfg_10 = Config.from_yaml("configs/model.yaml")
    cfg_10._data['model']['num_layers'] = 10

    for layers, cfg in [(20, cfg_20), (12, cfg_12), (10, cfg_10)]:
        model = CausalLM(cfg.model).to(device)
        model.train()
        optimizer = create_optimizer(model, learning_rate=3e-4)
        
        x = torch.randint(0, 32768, (2, 1024), device=device)
        y = torch.randint(0, 32768, (2, 1024), device=device)
        
        # Warmup
        for _ in range(2):
            optimizer.zero_grad()
            logits, loss, _ = model(x, targets=y)
            (loss / 16).backward()
        
        # Benchmark 1 step (16 micro steps = 32,768 tokens)
        start = time.time()
        optimizer.zero_grad()
        for _ in range(16):
            logits, loss, _ = model(x, targets=y)
            (loss / 16).backward()
        optimizer.step()
        elapsed = time.time() - start
        tok_sec = 32768 / elapsed
        
        print(f"Layers: {layers:2d} | Params: {model.get_num_params():,} | Step Time: {elapsed:.2f}s | Throughput: {tok_sec:.0f} tok/s")

if __name__ == "__main__":
    benchmark_configs()
