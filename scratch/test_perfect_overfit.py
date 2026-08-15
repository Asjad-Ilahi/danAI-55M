import torch
from tokenizers import Tokenizer
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
cfg = Config.from_yaml("configs/model.yaml")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CausalLM(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

sentence = "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy."
tokens = tokenizer.encode(sentence).ids
print(f"Target sentence tokens ({len(tokens)}): {tokens}")
print(f"Target tokens text: {[tokenizer.id_to_token(i) for i in tokens]}")

x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

print("\nTraining on 1 sentence for 500 steps...")
model.train()
for step in range(1, 501):
    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or step == 1:
        print(f"Step {step:3d} | Loss: {loss.item():.8f}")

model.eval()
generator = TextGenerator(model, tokenizer, device)

prompt = "Once upon a time, in a bright green"

print("\n--- GENERATION WITH KV CACHE (use_kv_cache=True) ---")
out_kv = generator.generate(prompt, max_new_tokens=30, temperature=0.0, use_kv_cache=True)
print(f"KV CACHE OUTPUT: '{out_kv}'")

print("\n--- GENERATION WITHOUT KV CACHE (use_kv_cache=False) ---")
out_nokv = generator.generate(prompt, max_new_tokens=30, temperature=0.0, use_kv_cache=False)
print(f"NO-KV CACHE OUTPUT: '{out_nokv}'")
