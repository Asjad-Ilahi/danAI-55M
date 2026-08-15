"""
Train BPE Tokenizer from scratch with 32,768 vocabulary size per §18.

Reads representative multi-source document samples from data/raw/ or dataset stream,
trains BPE tokenizer, and saves to tokenizer/tokenizer.json and tokenizer_config.json.
"""

import argparse
import json
from pathlib import Path
from typing import List, Iterator

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors


SPECIAL_TOKENS = ["<eos>", "<pad>", "<unk>", "<bos>"]
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"


def get_text_files(data_dir: Path) -> List[Path]:
    """Find all text/jsonl files in directory."""
    files = list(data_dir.glob("**/*.jsonl")) + list(data_dir.glob("**/*.txt")) + list(data_dir.glob("**/*.md"))
    return files


def file_iterator(files: List[Path]) -> Iterator[str]:
    """Yield documents from jsonl or text files."""
    for file_path in files:
        if file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        continue
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    yield content


def train_tokenizer(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32768,
    min_frequency: int = 2,
) -> Tokenizer:
    """Train Byte-level BPE tokenizer with 32,768 vocabulary size."""
    files = get_text_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No text files found in {data_dir}. Run prepare_dataset.py first.")

    print(f"Found {len(files)} raw/cleaned document files in {data_dir}")

    # ByteLevel BPE
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"Training BPE tokenizer (vocab_size={vocab_size:,})...")
    tokenizer.train_from_iterator(file_iterator(files), trainer=trainer)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token_id": tokenizer.token_to_id(EOS_TOKEN),
        "pad_token_id": tokenizer.token_to_id(PAD_TOKEN),
        "unk_token_id": tokenizer.token_to_id(UNK_TOKEN),
        "bos_token_id": tokenizer.token_to_id(BOS_TOKEN),
    }

    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Saved tokenizer to {tokenizer_json_path} (vocab_size: {tokenizer.get_vocab_size():,})")
    return tokenizer


def print_statistics(tokenizer: Tokenizer) -> None:
    vocab_size = tokenizer.get_vocab_size()
    print("\n" + "=" * 60)
    print("TOKENIZER SUMMARY & REPORT")
    print("=" * 60)
    print(f"  Vocab Size:     {vocab_size:,}")
    print(f"  Special Tokens: {SPECIAL_TOKENS}")
    for st in SPECIAL_TOKENS:
        print(f"    {st:<10} ID: {tokenizer.token_to_id(st)}")

    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "def train_model(config, dataset):\n    return model.fit(dataset)",
        "Integral formulation: \\int_{0}^{\\infty} x^2 e^{-x} dx = 2",
    ]

    print("\n  Sample Tokenizations:")
    for sample in samples:
        encoding = tokenizer.encode(sample)
        print(f"  Text: {sample!r}")
        print(f"    IDs ({len(encoding.ids)}): {encoding.ids[:10]}...\n")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train 32K BPE Tokenizer from scratch")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw/cleaned text directory")
    parser.add_argument("--output-dir", type=str, default="tokenizer", help="Path to save tokenizer artifacts")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default 32768)")
    args = parser.parse_args()

    tokenizer = train_tokenizer(Path(args.data_dir), Path(args.output_dir), vocab_size=args.vocab_size)
    print_statistics(tokenizer)


if __name__ == "__main__":
    main()
