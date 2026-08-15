"""
Perfect Multi-Domain Dataset Builder Script (25M Tokens).

Domains:
1. General Knowledge & Academic (FineWeb-Edu) -> 6,000,000 tokens -> data/raw/general_knowledge_clean.jsonl
2. Stories & Narrative Prose (TinyStories) -> 5,000,000 tokens -> data/raw/stories_clean.jsonl
3. Math & Reasoning (MetaMathQA / GSM8K) -> 5,000,000 tokens -> data/raw/math_clean.jsonl
4. Python Code & Scripts (Python-Edu) -> 5,000,000 tokens -> data/raw/python_clean.jsonl
5. HTML & Web Markup/Code (The-Stack / Code Instructions) -> 4,000,000 tokens -> data/raw/html_clean.jsonl

Total: 25,000,000 clean tokens (95% train / 5% val)
"""

import json
import shutil
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from src.data.cleaner import clean_text


def build_perfect_dataset():
    print("=" * 75)
    print("  BUILDING 25M TOKEN PERFECT MULTI-DOMAIN DATASET")
    print("  (6M General, 5M Stories, 5M Math, 5M Python, 4M HTML/Web)")
    print("=" * 75)

    tokenizer_path = Path("tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = 0

    raw_dir = Path("data/raw")
    shards_train_dir = Path("data/shards/train")
    shards_val_dir = Path("data/shards/val")

    # Clean existing data directories
    for d in [raw_dir, shards_train_dir, shards_val_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "General Knowledge (FineWeb-Edu)",
            "hf_path": "HuggingFaceTB/smollm-corpus",
            "name_kw": "fineweb-edu-dedup",
            "text_fn": lambda x: x.get("text", ""),
            "target_tokens": 6_000_000,
            "is_code": False,
            "raw_filename": "general_knowledge_clean.jsonl",
        },
        {
            "name": "Stories & Narrative (TinyStories)",
            "hf_path": "roneneldan/TinyStories",
            "name_kw": None,
            "text_fn": lambda x: x.get("text", ""),
            "target_tokens": 5_000_000,
            "is_code": False,
            "raw_filename": "stories_clean.jsonl",
        },
        {
            "name": "Math & Reasoning (MetaMathQA / GSM8K)",
            "hf_path": "meta-math/MetaMathQA",
            "name_kw": None,
            "text_fn": lambda x: f"Question: {x.get('query', '')}\nAnswer: {x.get('response', '')}",
            "fallback_hf_path": "gsm8k",
            "fallback_name_kw": "main",
            "fallback_text_fn": lambda x: f"Question: {x.get('question', '')}\nAnswer: {x.get('answer', '')}",
            "target_tokens": 5_000_000,
            "is_code": False,
            "raw_filename": "math_clean.jsonl",
        },
        {
            "name": "Python Code (Python-Edu)",
            "hf_path": "HuggingFaceTB/smollm-corpus",
            "name_kw": "python-edu",
            "text_fn": lambda x: x.get("text", ""),
            "target_tokens": 5_000_000,
            "is_code": True,
            "raw_filename": "python_clean.jsonl",
        },
        {
            "name": "HTML & Web Code (The-Stack / Alpaca Code)",
            "hf_path": "bigcode/the-stack-smol-xs",
            "name_kw": None,
            "data_dir": "data/html",
            "text_fn": lambda x: x.get("content", ""),
            "fallback_hf_path": "iamtarun/python_code_instructions_18k_alpaca",
            "fallback_name_kw": None,
            "fallback_text_fn": lambda x: f"<!-- HTML Web Snippet -->\n{x.get('output', '')}",
            "target_tokens": 4_000_000,
            "is_code": True,
            "raw_filename": "html_clean.jsonl",
        },
    ]

    all_train_tokens = []
    all_val_tokens = []
    total_start = time.time()

    for spec in sources:
        print(f"\n---> Collecting {spec['target_tokens']:,} tokens for: {spec['name']}")
        start_t = time.time()
        seen_domain_hashes = set()

        load_kwargs = {"split": "train", "streaming": True}
        if spec.get("name_kw"):
            load_kwargs["name"] = spec["name_kw"]
        if spec.get("data_dir"):
            load_kwargs["data_dir"] = spec["data_dir"]

        try:
            ds = load_dataset(spec["hf_path"], **load_kwargs)
            text_fn = spec["text_fn"]
        except Exception as e:
            if spec.get("fallback_hf_path"):
                print(f"     Primary source failed ({e}). Using fallback: {spec['fallback_hf_path']}")
                fb_kwargs = {"split": "train", "streaming": True}
                if spec.get("fallback_name_kw"):
                    fb_kwargs["name"] = spec["fallback_name_kw"]
                ds = load_dataset(spec["fallback_hf_path"], **fb_kwargs)
                text_fn = spec["fallback_text_fn"]
            else:
                raise e

        source_tokens = 0
        raw_filepath = raw_dir / spec["raw_filename"]
        doc_count = 0

        with open(raw_filepath, "w", encoding="utf-8") as raw_file:
            for doc in ds:
                raw_text = text_fn(doc)
                if not raw_text or len(raw_text) < 15:
                    continue

                cleaned = clean_text(raw_text, is_code=spec["is_code"])
                if not cleaned or len(cleaned) < 15:
                    continue

                # Deduplication check per domain
                if spec["is_code"]:
                    doc_hash = hash(cleaned[20:200]) if len(cleaned) > 200 else hash(cleaned)
                else:
                    doc_hash = hash(cleaned[:150])

                if doc_hash in seen_domain_hashes:
                    continue
                seen_domain_hashes.add(doc_hash)

                encoded = tokenizer.encode(cleaned)
                ids = encoded.ids
                if not ids or len(ids) < 5:
                    continue

                ids.append(eos_id)

                # Save raw jsonl document
                raw_file.write(json.dumps({"text": cleaned, "domain": spec["name"]}, ensure_ascii=False) + "\n")

                # Split 95% train / 5% val
                if doc_count % 20 == 0:
                    all_val_tokens.extend(ids)
                else:
                    all_train_tokens.extend(ids)

                doc_count += 1
                source_tokens += len(ids)

                if source_tokens >= spec["target_tokens"]:
                    break

        elapsed = time.time() - start_t
        print(f"     [SUCCESS] {spec['name']}: {source_tokens:,} clean tokens ({doc_count:,} docs) in {elapsed:.1f}s")
        print(f"     Saved raw clean file to: {raw_filepath}")

    print("\n" + "=" * 75)
    print(f"  TOTAL COLLECTED TOKENS: Train = {len(all_train_tokens):,} | Val = {len(all_val_tokens):,}")
    print(f"  GRAND TOTAL: {len(all_train_tokens) + len(all_val_tokens):,} TOKENS")
    print("=" * 75)

    # Write Binary Shards
    def write_shards(tokens_list, out_dir, prefix="shard", shard_size=5_000_000):
        out_dir.mkdir(parents=True, exist_ok=True)
        num_shards = (len(tokens_list) + shard_size - 1) // shard_size

        for i in range(num_shards):
            chunk = tokens_list[i * shard_size : (i + 1) * shard_size]
            bin_path = out_dir / f"{prefix}_{i:05d}.bin"
            meta_path = out_dir / f"{prefix}_{i:05d}.json"

            arr = np.array(chunk, dtype=np.uint16)
            with open(bin_path, "wb") as f:
                f.write(arr.tobytes())

            meta = {
                "shard_id": i,
                "num_tokens": len(chunk),
                "dtype": "uint16",
                "bytes": bin_path.stat().st_size,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Wrote {bin_path.name}: {len(chunk):,} tokens ({bin_path.stat().st_size / (1024**2):.2f} MB)")

    print("\nWriting Binary Shards (Train)...")
    write_shards(all_train_tokens, shards_train_dir, prefix="shard")

    print("\nWriting Binary Shards (Val)...")
    write_shards(all_val_tokens, shards_val_dir, prefix="shard")

    print(f"\nSUCCESS: Perfect 25M token dataset created in {time.time() - total_start:.1f}s!")


if __name__ == "__main__":
    build_perfect_dataset()
