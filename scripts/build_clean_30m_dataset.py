"""
30M Token Fresh Multi-Domain Dataset Builder script.

Mixture (Total: 30,000,000 Tokens):
- 20M Tokens: General Knowledge & Q&A (FineWeb-Edu fresh skip 300k + Cosmopedia-v2 Q&A) -> data/raw/general_qa_clean.jsonl
- 5M Tokens:  Python Coding (flytech + CodeAlpaca + python-edu) -> data/raw/coding_clean.jsonl
- 5M Tokens:  Mathematics & Arithmetic (SimpleMath + SwallowMath + GSM8k) -> data/raw/math_clean.jsonl

Outputs:
- Raw readable clean JSONL files in data/raw/
- Tokenized binary uint16 shards in data/shards/train (28.5M) and data/shards/val (1.5M)
"""

import json
import re
import shutil
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from src.data.cleaner import clean_text


def format_python_code(instruction: str, output: str) -> str:
    """Format instruction + output into clean executable Python source with module docstring."""
    match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if match:
        code_body = match.group(1).strip()
    else:
        lines = [l for l in output.split("\n") if not l.strip().startswith("```")]
        code_body = "\n".join(lines).strip()

    inst = instruction.strip()
    if inst and not inst.endswith('.'):
        inst += '.'

    if code_body.startswith('"""') or code_body.startswith("'''"):
        return code_body
    else:
        return f'"""\n{inst}\n"""\n{code_body}'


def extract_text_from_doc(doc: dict, is_code: bool = False) -> str:
    """Extract string content dynamically regardless of dataset field schema."""
    if not isinstance(doc, dict):
        return str(doc)
    
    q_val = doc.get("instruction") or doc.get("question") or doc.get("problem") or doc.get("prompt")
    a_val = doc.get("output") or doc.get("answer") or doc.get("solution") or doc.get("response")

    if is_code and q_val and a_val:
        return format_python_code(str(q_val), str(a_val))

    if q_val is not None and a_val is not None:
        q_str = str(q_val).strip()
        a_str = str(a_val).strip()
        if q_str.endswith("="):
            return f"{q_str} {a_str}"
        return f"Question: {q_str}\n\nAnswer:\n{a_str}"

    for k in ["text", "content", "code"]:
        if k in doc and isinstance(doc[k], str) and len(doc[k].strip()) > 0:
            return doc[k].strip()
            
    if q_val is not None:
        return str(q_val).strip()
    elif a_val is not None:
        return str(a_val).strip()
        
    str_vals = [str(v).strip() for v in doc.values() if isinstance(v, (str, int, float)) and len(str(v).strip()) > 0]
    return "\n\n".join(str_vals)


def build_30m_dataset():
    print("=" * 75)
    print("  BUILDING 30M FRESH TOKEN DATASET (20M GENERAL/Q&A, 5M CODING, 5M MATH)")
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

    # Clean existing directories
    for d in [raw_dir, shards_train_dir, shards_val_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "General Knowledge & Q&A (Fresh FineWeb-Edu + Cosmopedia-v2)",
            "sources": [
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "fineweb-edu-dedup",
                    "skip_docs": 40000,  # 100% fresh, unseen web/science articles (>32M tokens past prior training)
                },
                {
                    "hf_path": "HuggingFaceTB/cosmopedia-v2",
                    "name_kw": None,
                    "skip_docs": 0,
                },
            ],
            "target_tokens": 20_000_000,
            "raw_filename": "general_qa_clean.jsonl",
            "is_code": False,
        },
        {
            "name": "Python Coding (flytech + CodeAlpaca + python-edu)",
            "sources": [
                {
                    "hf_path": "flytech/python-codes-25k",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "sahil2801/CodeAlpaca-20k",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "python-edu",
                    "skip_docs": 0,
                },
            ],
            "target_tokens": 5_000_000,
            "raw_filename": "coding_clean.jsonl",
            "is_code": True,
        },
        {
            "name": "Mathematics & Arithmetic (SimpleMath + SwallowMath + GSM8K)",
            "sources": [
                {
                    "hf_path": "ProCreations/SimpleMath",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "tokyotech-llm/swallow-math",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "gsm8k",
                    "name_kw": "main",
                    "skip_docs": 0,
                },
            ],
            "target_tokens": 5_000_000,
            "raw_filename": "math_clean.jsonl",
            "is_code": False,
        },
    ]

    all_tokens = []
    seen_hashes = set()
    total_start = time.time()

    for spec in sources:
        print(f"\n---> Collecting {spec['target_tokens']:,} tokens for: {spec['name']}")
        start_t = time.time()
        
        source_tokens = 0
        raw_filepath = raw_dir / spec["raw_filename"]
        
        with open(raw_filepath, "w", encoding="utf-8") as raw_file:
            for s_info in spec["sources"]:
                if source_tokens >= spec["target_tokens"]:
                    break
                skip_docs = s_info.get("skip_docs", 0)
                print(f"     Streaming from HF dataset: {s_info['hf_path']}" + (f" (skipping first {skip_docs:,} docs)" if skip_docs > 0 else ""))
                load_kwargs = {"split": "train", "streaming": True}
                if s_info.get("name_kw"):
                    load_kwargs["name"] = s_info["name_kw"]

                try:
                    ds = load_dataset(s_info["hf_path"], **load_kwargs)
                    if skip_docs > 0:
                        ds = ds.skip(skip_docs)
                except Exception as err:
                    print(f"     Warning: Could not load {s_info['hf_path']}: {err}")
                    continue

                for doc in ds:
                    raw_text = extract_text_from_doc(doc, is_code=spec.get("is_code", False))

                    if not raw_text or len(raw_text) < 5:
                        continue
                    
                    cleaned = clean_text(raw_text, is_code=spec.get("is_code", False))
                    if not cleaned or len(cleaned) < 5:
                        continue

                    # Hash deduplication
                    doc_hash = hash(cleaned[:120])
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)

                    encoded = tokenizer.encode(cleaned)
                    ids = encoded.ids
                    if not ids or len(ids) < 2:
                        continue

                    ids.append(eos_id)
                    all_tokens.extend(ids)
                    source_tokens += len(ids)

                    raw_file.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")

                    if source_tokens >= spec["target_tokens"]:
                        break
        
        elapsed = time.time() - start_t
        print(f"     Finished {spec['name']}: Collected {source_tokens:,} tokens in {elapsed:.1f}s")
        print(f"     Saved raw clean corpus to {raw_filepath}")

    print(f"\n" + "=" * 75)
    print(f"  TOTAL COLLECTED CLEAN TOKENS: {len(all_tokens):,}")
    print(f"=" * 75)

    # Split into train (95%) and validation (5%)
    val_size = int(len(all_tokens) * 0.05)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    print(f"Train Tokens: {len(train_tokens):,} | Val Tokens: {len(val_tokens):,}")

    # Write binary shards
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
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            
            print(f"  Wrote {bin_path.name}: {len(chunk):,} tokens ({bin_path.stat().st_size / 1e6:.2f} MB)")

    print("\nWriting Binary Shards (Train)...")
    write_shards(train_tokens, shards_train_dir, prefix="shard")

    print("\nWriting Binary Shards (Val)...")
    write_shards(val_tokens, shards_val_dir, prefix="shard")

    print(f"\nSUCCESS: 30M token fresh dataset built in {time.time() - total_start:.1f}s!")


if __name__ == "__main__":
    build_30m_dataset()
