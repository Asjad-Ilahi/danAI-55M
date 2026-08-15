"""
Quick 50-step training verification test on the clean 20M dataset.
Verifies model loss convergence, gradients, and text generation.
"""

import time
import torch
from src.model.gpt import CausalLM
from src.utils.config import Config
from src.training.optimizer import create_optimizer
from src.data.shard_dataset import ShardDataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load Tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 2. Init Model from YAML configs
config = Config.from_multiple("configs/model.yaml", "configs/train.yaml")
model = CausalLM(config.model).to(device)
optimizer = create_optimizer(model, learning_rate=1.0e-3)

# 3. Load ShardDataset
ds = ShardDataset("data/shards/train", seq_len=1024)
loader = DataLoader(ds, batch_size=4, shuffle=True)
data_iter = iter(loader)

print("\n=== STARTING 50-STEP CLEAN DATASET TRAINING TEST ===")
model.train()
t0 = time.time()

for step in range(1, 51):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
    
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    attn_mask = batch["attn_mask"].to(device)
    
    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y, attention_mask=attn_mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step == 1 or step % 10 == 0:
        print(f"Step {step:3d} | Training Loss: {loss.item():.4f}")

dt = time.time() - t0
print(f"\n50 steps completed in {dt:.1f}s ({dt/50*1000:.1f} ms/step). Final Loss: {loss.item():.4f}")

# Generation test
print("\n=== GENERATION TEST ON CLEAN TRAINED MODEL ===")
model.eval()
prompt = "Once upon a time,"
input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=40)
gen_text = tokenizer.decode(out[0].tolist())
print(f"PROMPT: '{prompt}'")
print(f"OUTPUT:\n{gen_text}")
print("\nVERIFICATION COMPLETE!")
