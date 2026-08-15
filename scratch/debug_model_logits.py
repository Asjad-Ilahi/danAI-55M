import torch
from tokenizers import Tokenizer
from src.utils.config import Config
from src.model.gpt import CausalLM

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
cfg = Config.from_yaml("configs/model.yaml")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CausalLM(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

sentence = "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy."
tokens = tokenizer.encode(sentence).ids

x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

print("Training for 500 steps...")
model.train()
for step in range(1, 501):
    optimizer.zero_grad()
    logits, loss, _ = model(x, targets=y)
    loss.backward()
    optimizer.step()

model.eval()

# Let's inspect the logits at each position when passing full x
with torch.no_grad():
    logits, _, _ = model(x)
    print(f"\nFull x forward shape: {logits.shape}")
    for pos in range(len(tokens) - 1):
        target_tok = tokens[pos + 1]
        top1_tok = logits[0, pos].argmax().item()
        target_str = tokenizer.id_to_token(target_tok)
        top1_str = tokenizer.id_to_token(top1_tok)
        top1_logit = logits[0, pos, top1_tok].item()
        target_logit = logits[0, pos, target_tok].item()
        match = (target_tok == top1_tok)
        print(f"Pos {pos:2d} ({repr(tokenizer.id_to_token(tokens[pos])):<12}): target={target_tok} ({repr(target_str):<12}), pred={top1_tok} ({repr(top1_str):<12}), match={match}, target_logit={target_logit:.2f}, top1_logit={top1_logit:.2f}")

# Now let's test passing ONLY prompt prefix: tokens[:9]
prompt_x = torch.tensor([tokens[:9]], dtype=torch.long, device=device)
with torch.no_grad():
    prompt_logits, _, _ = model(prompt_x)
    print(f"\nPrompt_x forward shape: {prompt_logits.shape}")
    last_pos = prompt_logits.shape[1] - 1
    top1_tok = prompt_logits[0, last_pos].argmax().item()
    target_tok = tokens[9]
    print(f"At end of prompt ('Ġgreen'): target={target_tok} ({repr(tokenizer.id_to_token(target_tok))}), pred={top1_tok} ({repr(tokenizer.id_to_token(top1_tok))})")
