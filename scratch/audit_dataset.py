import glob
from pathlib import Path
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

tokenizer_file = Path("tokenizer/tokenizer.json")
tokenizer = Tokenizer.from_file(str(tokenizer_file))

shard_files = sorted(list(Path("data/shards/train").glob("shard_*.bin")))
shard_files = [f for f in shard_files if not f.name.endswith("_seg.bin")]

print(f"Found {len(shard_files)} shard files.")

total_tokens = 0
token_counts = Counter()
documents = []
current_doc_tokens = []

for shard_file in shard_files:
    tokens = np.fromfile(str(shard_file), dtype=np.uint16)
    total_tokens += len(tokens)
    token_counts.update(tokens)
    
    # Extract documents based on <eos> token (id = 0)
    for token in tokens:
        if token == 0:  # <eos>
            if current_doc_tokens:
                documents.append(current_doc_tokens)
                current_doc_tokens = []
        else:
            current_doc_tokens.append(token)

if current_doc_tokens:
    documents.append(current_doc_tokens)

print(f"Total tokens analyzed: {total_tokens:,}")
print(f"Total documents extracted: {len(documents):,}")

doc_lengths = [len(d) for d in documents]
avg_doc_len = np.mean(doc_lengths) if doc_lengths else 0
median_doc_len = np.median(doc_lengths) if doc_lengths else 0

print(f"Average document length: {avg_doc_len:.1f} tokens")
print(f"Median document length: {median_doc_len:.1f} tokens")

# Duplicate document check
doc_hashes = Counter(tuple(d[:50]) for d in documents if len(d) >= 10)
duplicate_count = sum(c - 1 for c in doc_hashes.values() if c > 1)
dup_rate = (duplicate_count / max(1, len(documents))) * 100
print(f"Duplicate document rate (first 50 tokens): {dup_rate:.2f}% ({duplicate_count:,} duplicates)")

# Top 50 most frequent tokens
print("\n=== TOP 50 MOST FREQUENT TOKENS ===")
print(f"{'Rank':<5} {'ID':<6} {'Token':<20} {'Count':<12} {'Pct':<8}")
print("-" * 55)

for rank, (tok_id, count) in enumerate(token_counts.most_common(50), 1):
    tok_str = repr(tokenizer.id_to_token(int(tok_id)))
    pct = (count / total_tokens) * 100
    print(f"{rank:<5} {tok_id:<6} {tok_str:<20} {count:<12,} {pct:<6.2f}%")

# Sample 10 documents
print("\n=== SAMPLE DOCUMENTS FROM SHARDS ===")
sample_indices = np.linspace(0, len(documents) - 1, 10, dtype=int)
for i, idx in enumerate(sample_indices, 1):
    doc_toks = documents[idx][:150]  # First 150 tokens
    decoded_text = tokenizer.decode(doc_toks)
    print(f"\n--- SAMPLE {i} (Doc #{idx}, Length {len(documents[idx])} tokens) ---")
    print(decoded_text[:400] + ("..." if len(decoded_text) > 400 else ""))
