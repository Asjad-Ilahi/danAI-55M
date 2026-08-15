import json
from pathlib import Path
from tokenizers import Tokenizer

tokenizer_file = Path("tokenizer/tokenizer.json")
if not tokenizer_file.exists():
    print("Tokenizer file not found!")
    exit(1)

tokenizer = Tokenizer.from_file(str(tokenizer_file))

test_sentences = [
    "Once upon a time, there was a young girl named Alice.",
    "The capital city of France is Paris.",
    "def fibonacci(n):",
    "Photosynthesis is the process by which plants convert light energy.",
    "Artificial intelligence has changed the way humans interact with technology."
]

print("=== TOKENIZER ENCODE / DECODE ROUNDTRIP TEST ===")
for text in test_sentences:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    is_lossless = (text.strip() == decoded.strip())
    tokens = [tokenizer.id_to_token(i) for i in encoded.ids]
    
    words = text.split()
    chars = len(text)
    n_tokens = len(encoded.ids)
    tokens_per_word = n_tokens / max(1, len(words))
    chars_per_token = chars / max(1, n_tokens)
    
    print(f"\nOriginal: '{text}'")
    print(f"Decoded:  '{decoded}'")
    print(f"Match:    {is_lossless}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Tokens:   {tokens}")
    print(f"Stats:    {n_tokens} tokens | {tokens_per_word:.2f} tok/word | {chars_per_token:.2f} char/tok")

# Tokenizer Vocab Inspection
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
print(f"\n=== TOKENIZER VOCAB SUMMARY ===")
print(f"Vocab Size: {vocab_size}")

# Check special tokens
eos_id = tokenizer.token_to_id("<eos>")
bos_id = tokenizer.token_to_id("<bos>")
unk_id = tokenizer.token_to_id("<unk>")
pad_id = tokenizer.token_to_id("<pad>")
print(f"Special Token IDs: <eos>={eos_id}, <bos>={bos_id}, <unk>={unk_id}, <pad>={pad_id}")
