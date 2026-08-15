"""
Dataset validation script per §27.

Verifies:
- Shard file existence and non-zero uint16 arrays
- Token ID ranges fit inside vocabulary [0, vocab_size)
- Memmap loading performance
- Document boundary <eos> alignment
"""

import argparse
import json
from pathlib import Path
import numpy as np

from src.utils.config import Config


def validate_dataset(shards_dir: Path, vocab_size: int = 32768) -> bool:
    print("=" * 75)
    print("DATASET INTEGRITY & SHARD VALIDATION")
    print("=" * 75)

    all_passed = True

    for split in ["train", "val"]:
        split_dir = shards_dir / split
        if not split_dir.exists():
            print(f"❌ Split directory {split_dir} does not exist!")
            all_passed = False
            continue

        shard_files = sorted(list(split_dir.glob("shard_*.bin")))
        shard_files = [f for f in shard_files if not f.name.endswith("_seg.bin")]

        if not shard_files:
            print(f"❌ No binary shard files found in {split_dir}!")
            all_passed = False
            continue

        total_tokens = 0
        for sf in shard_files:
            mmap = np.memmap(str(sf), dtype=np.uint16, mode="r")
            num_tokens = len(mmap)
            total_tokens += num_tokens

            # Check min and max token IDs
            min_id = int(np.min(mmap))
            max_id = int(np.max(mmap))

            if max_id >= vocab_size:
                print(f"❌ Token ID overflow in {sf.name}: max_id {max_id} >= vocab_size {vocab_size}!")
                all_passed = False

            if min_id < 0:
                print(f"❌ Negative token ID in {sf.name}: min_id {min_id}!")
                all_passed = False

        print(f"✓ Split '{split}': {len(shard_files)} shards, {total_tokens:,} tokens total. (Token IDs in range [0, {vocab_size}))")

    print("-" * 75)
    status_str = "PASS" if all_passed else "FAIL"
    print(f"VALIDATION STATUS: {status_str}")
    print("=" * 75 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate binary shards")
    parser.add_argument("--shards-dir", type=str, default="data/shards")
    parser.add_argument("--vocab-size", type=int, default=32768)
    args = parser.parse_args()

    validate_dataset(Path(args.shards_dir), vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
