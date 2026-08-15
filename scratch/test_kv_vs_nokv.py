import torch
from tokenizers import Tokenizer
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator

# Load tokenizer and model
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
cfg = Config.from_yaml("configs/model.yaml")

# Create a small model and train on 1 sentence for 50 steps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CausalLM(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

sentence = "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy."
tokens = tokenizer.encode(sentence).ids
x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

print("Training on 1 sentence for 100 steps...")
model.train()
for step in range(1, 101):
    optimizer.zero_grad()
    _, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

model.eval()
generator = TextGenerator(model, tokenizer, device)

prompt = "Once upon a time, in a bright green"

print("\n--- GENERATION WITH KV CACHE (use_kv_cache=True) ---")
out_kv = generator.generate(prompt, max_new_tokens=30, temperature=0.0, use_kv_cache=True)
print(out_kv)

print("\n--- GENERATION WITHOUT KV CACHE (use_kv_cache=False) ---")
out_nokv = generator.generate(prompt, max_new_tokens=30, temperature=0.0, use_kv_cache=False)
print(out_nokv)
