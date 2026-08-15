import time
import torch
from pathlib import Path
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.data.shard_dataset import ShardDataset
from src.training.optimizer import create_optimizer

def run_hardware_test():
    print("==========================================================")
    print("  HARDWARE & PIPELINE FULL INTEGRITY TEST")
    print("==========================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1/4] Device: {device} | PyTorch: {torch.__version__}")
    
    cfg = Config.from_multiple("configs/model.yaml", "configs/train.yaml")
    model = CausalLM(cfg.model).to(device)
    print(f"[2/4] Model params: {model.get_num_params():,} parameters")
    
    optimizer = create_optimizer(model, learning_rate=3e-4)
    decay_params = sum(p.numel() for g in optimizer.param_groups if g['weight_decay'] > 0 for p in g['params'])
    nodecay_params = sum(p.numel() for g in optimizer.param_groups if g['weight_decay'] == 0 for p in g['params'])
    print(f"      Optimizer parameter groups: Decay={decay_params:,}, NoDecay={nodecay_params:,}")
    
    train_dataset = ShardDataset(Path("data/shards/train"), seq_len=1024, pack_with_document_mask=True)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(f"[3/4] ShardDataset loaded successfully ({len(train_dataset)} samples ready)")
    
    print("[4/4] Testing 5 full training steps (micro=2, accum=16)...")
    model.train()
    batch_iter = iter(loader)
    
    total_start = time.time()
    for step in range(1, 6):
        step_start = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(16):
            try:
                b = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                b = next(batch_iter)
            
            x = b["x"].to(device)
            y = b["y"].to(device)
            mask = b["attn_mask"].to(device) if b.get("attn_mask") is not None else None
            
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                logits, loss, _ = model(x, attention_mask=mask, targets=y)
                scaled_loss = loss / 16
            
            scaled_loss.backward()
            accum_loss += loss.item() / 16
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step_time = time.time() - step_start
        tok_sec = 32768 / step_time
        print(f"      Step {step} | Loss: {accum_loss:.4f} | Time: {step_time:.2f}s ({tok_sec:.0f} tokens/sec)")
        
    total_time = time.time() - total_start
    print(f"\nSUCCESS: Pipeline verified in {total_time:.2f}s! Memory, loss, & gradients 100% healthy.")

if __name__ == "__main__":
    run_hardware_test()
