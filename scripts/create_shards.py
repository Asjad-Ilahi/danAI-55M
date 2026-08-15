"""
Binary shard creation script per §10, §6, & §12.

Reads cleaned docs from data/cleaned/ → tokenizes with tokenizer/ →
applies document-level 95/5 train/val split → applies document-aware packing →
writes uint16 binary shards to data/processed/{train,validation}/.
"""

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from src.data.packing import pack_documents
from src.data.shard_writer import ShardWriter


def create_shards(
    cleaned_dir: Path,
    processed_dir: Path,
    tokenizer_path: Path,
    seq_len: int = 1024,
    val_ratio: float = 0.05,
    seed: int = 42,
):
    cleaned_file = cleaned_dir / "cleaned_docs.jsonl"
    if not cleaned_file.exists():
        raise FileNotFoundError(f"Cleaned file {cleaned_file} not found. Run scripts/prepare_data.py first.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<unk>") or 0

    vocab_size = tokenizer.get_vocab_size()

    # Load cleaned docs
    docs = []
    with open(cleaned_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if text.strip():
                    docs.append(text)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(docs):,} cleaned documents.")

    # 95/5 Document-level split (§12)
    random.seed(seed)
    random.shuffle(docs)

    num_val = max(1, int(len(docs) * val_ratio))
    val_docs = docs[:num_val]
    train_docs = docs[num_val:]

    print(f"Document-level split ({100*(1-val_ratio):.0f}/{100*val_ratio:.0f}): Train = {len(train_docs):,}, Val = {len(val_docs):,}")

    for split_name, split_docs in [("train", train_docs), ("validation", val_docs)]:
        print(f"\nProcessing {split_name} split ({len(split_docs):,} docs)...")

        # Tokenize documents
        tokenized_docs = []
        for text in split_docs:
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            if not ids or ids[-1] != eos_id:
                ids.append(eos_id)
            tokenized_docs.append(ids)

        writer = ShardWriter(
            output_dir=processed_dir,
            split=split_name,
            max_tokens_per_shard=10_000_000,
            vocab_size=vocab_size,
        )

        for packed_tokens, packed_segments in pack_documents(tokenized_docs, max_seq_len=seq_len, eos_token_id=eos_id):
            writer.add_packed_sequence(packed_tokens, packed_segments)

        summary = writer.close()
        print(f"Completed {split_name}: {summary['total_tokens']:,} tokens across {summary['total_shards']} shards.")


def main():
    parser = argparse.ArgumentParser(description="Create uint16 binary token shards")
    parser.add_argument("--cleaned-dir", type=str, default="data/cleaned")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    create_shards(
        Path(args.cleaned_dir),
        Path(args.processed_dir),
        Path(args.tokenizer_dir),
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
