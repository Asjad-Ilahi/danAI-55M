import torch
from tokenizers import Tokenizer
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.evaluation.generation import TextGenerator

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
cfg = Config.from_yaml("configs/model.yaml")
cfg.model.num_layers = 4

device = torch.device("cpu")
model = CausalLM(cfg).to(device)

# Monkey-patch forward in model to fix double shift!
def fixed_loss_forward(self, input_ids, attention_mask=None, position_ids=None, targets=None, kv_caches=None, use_cache=False):
    h = self.token_embedding(input_ids)
    new_kv_caches = [] if (kv_caches is not None or use_cache) else None
    
    for i, layer in enumerate(self.layers):
        layer_kv_cache = kv_caches[i] if kv_caches is not None else None
        h, new_cache = layer(h, attention_mask=attention_mask, position_ids=position_ids, kv_cache=layer_kv_cache, use_cache=use_cache, use_checkpoint=self.use_checkpoint)
        if new_kv_caches is not None:
            new_kv_caches.append(new_cache)
            
    h = self.final_norm(h)
    logits = self.lm_head(h)
    
    loss = None
    if targets is not None:
        # x and y (targets) are ALREADY aligned: targets[:, i] is the next token for input_ids[:, i]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=-100
        )
    return logits, loss, new_kv_caches

# Bind fixed forward
model.forward = fixed_loss_forward.__get__(model, CausalLM)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

sentence = "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy."
tokens = tokenizer.encode(sentence).ids

x = torch.tensor([tokens[:-1]], dtype=torch.long)
y = torch.tensor([tokens[1:]], dtype=torch.long)

print("Training with FIXED loss alignment for 100 steps...")
model.train()
for step in range(1, 101):
    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

model.eval()

print("\n--- POSITION-BY-POSITION PREDICTIONS ON FULL X ---")
with torch.no_grad():
    logits, _, _ = model(x)
    for pos in range(len(tokens) - 1):
        target_tok = tokens[pos + 1]
        top1_tok = logits[0, pos].argmax().item()
        target_str = tokenizer.id_to_token(target_tok)
        top1_str = tokenizer.id_to_token(top1_tok)
        match = (target_tok == top1_tok)
        print(f"Pos {pos:2d} ({repr(tokenizer.id_to_token(tokens[pos])):<12}): target={target_tok} ({repr(target_str):<12}), pred={top1_tok} ({repr(top1_str):<12}), match={match}")

generator = TextGenerator(model, tokenizer, device)
prompt = "Once upon a time, in a bright green"

print("\n--- GENERATION WITH FIXED LOSS (KV Cache) ---")
print(generator.generate(prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=True))

print("\n--- GENERATION WITH FIXED LOSS (No KV Cache) ---")
print(generator.generate(prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=False))
