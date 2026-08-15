"""
Script to decode and inspect clean text samples from binary shard files.
"""

import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Load first 2,000 tokens from train shard 0
shard_path = "data/shards/train/shard_00000.bin"
tokens = np.fromfile(shard_path, dtype=np.uint16, count=2000)

decoded_text = tokenizer.decode(tokens.tolist())

print("=" * 60)
print(f"DECODED TEXT SAMPLE FROM {shard_path}:")
print("=" * 60)
print(decoded_text[:1500])
print("=" * 60)
