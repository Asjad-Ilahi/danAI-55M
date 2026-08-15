import os
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from src.utils.config import Config
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer
from src.evaluation.generation import TextGenerator

# 1. Clean synthetic text corpus (~5k tokens)
clean_texts = [
    "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy. Sammy loved collecting shiny acorns and sharing them with his best friend, Barnaby the owl. Every evening, Sammy and Barnaby sat under the big oak tree and watched the golden sunset. Barnaby would tell stories about the stars, and Sammy would listen with a smile.",
    "Artificial intelligence is a branch of computer science dedicated to building systems capable of performing tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. Modern language models use deep neural networks with self-attention mechanisms to process sequential text.",
    "The solar system consists of the Sun and the celestial objects bound to it by gravity. The eight planets in order from the Sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Earth is the only known planet in the universe that supports biological life, thanks to its liquid water oceans and oxygen-rich atmosphere.",
    "Photosynthesis is the biochemical process by which green plants, algae, and some bacteria convert light energy into chemical energy. Using sunlight, plants absorb carbon dioxide from the air and water from the soil to produce glucose and release oxygen into the atmosphere.",
    "To solve the Fibonacci sequence in Python, one can write a simple recursive or iterative function. The Fibonacci numbers form a sequence where each number is the sum of the two preceding ones, starting from 0 and 1. The sequence begins: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, and so on."
]

# Repeat text to reach ~5,000 tokens
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

full_text = "\n\n".join(clean_texts * 20)
encoded = tokenizer.encode(full_text)
token_ids = encoded.ids
print(f"Tiny Overfit Dataset: {len(token_ids):,} tokens")

# 2. Dataset wrapper
class MemoryDataset(Dataset):
    def __init__(self, tokens, seq_len=256):
        self.seq_len = seq_len
        self.samples = []
        sample_len = seq_len + 1
        for i in range(0, len(tokens) - sample_len, seq_len):
            chunk = tokens[i : i + sample_len]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.samples.append({"x": x, "y": y})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

dataset = MemoryDataset(token_ids, seq_len=256)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Model setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device for overfit test: {device}")

cfg = Config.from_yaml("configs/model.yaml")
model = CausalLM(cfg).to(device)

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

print("\n=== STARTING TINY OVERFIT TEST (300 STEPS) ===")
model.train()
start_time = time.time()

for step in range(1, 301):
    epoch_loss = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(loader)
    if step % 20 == 0 or step == 1:
        print(f"Step {step:3d} | Training Loss: {avg_loss:.4f}")

print(f"\nCompleted in {time.time() - start_time:.1f}s. Final Loss: {avg_loss:.4f}")

# 5. Test generation from overfitted model
model.eval()
generator = TextGenerator(model, tokenizer, device)

prompts = [
    "Once upon a time, in a bright green forest,",
    "The solar system consists of",
    "Photosynthesis is the biochemical process"
]

print("\n=== OVERFIT MODEL GENERATION TEST ===")
for p in prompts:
    generated = generator.generate(p, max_new_tokens=60, temperature=0.2, top_k=10, top_p=0.9, use_kv_cache=True)
    print(f"\nPROMPT: '{p}'")
    print(f"OUTPUT:\n{generated}")
