"""
50M Token Multi-Domain Pretraining Corpus Builder Script.

Target Corpus Mixture (Total: 50,000,000 Tokens):
- 40% FineWeb-Edu (20,000,000 tokens)      -> data/raw/fineweb_edu_clean.jsonl
- 16% Cosmopedia v2 (8,000,000 tokens)     -> data/raw/cosmopedia_v2_clean.jsonl
- 14% High-Quality Code (7,000,000 tokens) -> data/raw/coding_clean.jsonl
- 10% Mathematics (5,000,000 tokens)       -> data/raw/math_clean.jsonl
- 10% High-Quality Q&A (5,000,000 tokens)  -> data/raw/qa_clean.jsonl
- 6%  Wikipedia (3,000,000 tokens)         -> data/raw/wikipedia_clean.jsonl
- 4%  Stories (2,000,000 tokens)           -> data/raw/stories_clean.jsonl

Outputs:
- Raw clean readable JSONL files in data/raw/
- Tokenized binary uint16 shards in data/shards/train (47.5M tokens) and data/shards/val (2.5M tokens)
- Comprehensive manifest in data/manifest.json
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from src.data.cleaner import clean_text, is_valid_quality, strip_html_tags


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


def extract_text_from_doc(doc: dict, is_code: bool = False, domain_type: str = "general") -> str:
    """Extract text from dataset document dictionary according to domain schema."""
    if not isinstance(doc, dict):
        return str(doc)

    if domain_type == "code":
        q_val = doc.get("instruction") or doc.get("question") or doc.get("prompt")
        a_val = doc.get("output") or doc.get("answer") or doc.get("code") or doc.get("solution") or doc.get("text")
        if q_val and a_val:
            return format_python_code(str(q_val), str(a_val))
        elif a_val:
            return str(a_val).strip()
        elif doc.get("code"):
            return str(doc["code"]).strip()
        elif doc.get("text") and len(str(doc["text"]).strip()) > 20:
            return str(doc["text"]).strip()

    if domain_type == "qa":
        q_val = doc.get("instruction") or doc.get("question") or doc.get("prompt")
        a_val = doc.get("output") or doc.get("answer") or doc.get("response")
        if q_val and a_val:
            clean_q = strip_html_tags(str(q_val)).strip()
            clean_a = strip_html_tags(str(a_val)).strip()
            return f"Question:\n{clean_q}\n\nAnswer:\n{clean_a}"

    if domain_type == "math":
        prob = doc.get("problem") or doc.get("question") or doc.get("instruction")
        sol = doc.get("solution") or doc.get("answer") or doc.get("output")
        if prob and sol:
            clean_p = strip_html_tags(str(prob)).strip()
            clean_s = strip_html_tags(str(sol)).strip()
            return f"Problem:\n{clean_p}\n\nSolution:\n{clean_s}"

    # Explicit text content columns
    for k in ["text", "content", "code", "article"]:
        if k in doc and isinstance(doc[k], str) and len(doc[k].strip()) > 0:
            # Prevent picking up pure metadata blob_id or path if content is missing
            if k == "text" and any(meta_k in doc for meta_k in ["blob_id", "repo_name"]) and "def " not in doc[k] and "class " not in doc[k] and len(doc[k]) < 200:
                continue
            return doc[k].strip()

    # Explicit Q&A fallback
    q_val = doc.get("instruction") or doc.get("question") or doc.get("prompt")
    a_val = doc.get("output") or doc.get("answer") or doc.get("response")
    if q_val and a_val:
        clean_q = strip_html_tags(str(q_val)).strip()
        clean_a = strip_html_tags(str(a_val)).strip()
        return f"Question:\n{clean_q}\n\nAnswer:\n{clean_a}"
    elif a_val and len(str(a_val).strip()) > 10:
        return strip_html_tags(str(a_val)).strip()

    # DO NOT fall back to stringifying doc metadata keys (blob_id, repo_name, path)!
    return ""


def compute_doc_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for exact document deduplication."""
    norm = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(norm.encode('utf-8')).hexdigest()


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_archive_hashes(archive_dir: Path) -> set[str]:
    """Scan previous raw jsonl datasets and return a set of all previously consumed SHA-256 hashes."""
    archive_hashes = set()
    raw_archive = archive_dir / "raw"
    if not raw_archive.exists():
        return archive_hashes

    print(f"---> Loading past document hashes from archive: {raw_archive}...", flush=True)
    for jsonl_file in raw_archive.glob("*.jsonl"):
        print(f"     Reading archive file: {jsonl_file.name}...", flush=True)
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        archive_hashes.add(compute_doc_hash(text))
                except Exception:
                    continue

    print(f"✓ Loaded {len(archive_hashes):,} unique document hashes from archive.", flush=True)
    return archive_hashes


def build_50m_corpus():
    print("=" * 80)
    print("  BUILDING NEW 60,000,000 TOKEN MULTI-DOMAIN PRETRAINING CORPUS")
    print("=" * 80)

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
    archive_dir = Path("data/archive_30m")

    raw_dir.mkdir(parents=True, exist_ok=True)
    shards_train_dir.mkdir(parents=True, exist_ok=True)
    shards_val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load archive hashes for 100% fresh data guarantee
    seen_hashes = load_archive_hashes(archive_dir)    # 2. Specification of domain mixture
    domain_specs = [
        {
            "name": "FineWeb-Edu",
            "target_tokens": 20_000_000,
            "raw_filename": "fineweb_edu_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "sources": [
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "fineweb-edu-dedup",
                    "skip_docs": 10_000,  # Automatic deduplication against 170k archive hashes
                }
            ],
        },
        {
            "name": "Cosmopedia v2",
            "target_tokens": 8_000_000,
            "raw_filename": "cosmopedia_v2_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "sources": [
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "cosmopedia-v2",
                    "skip_docs": 100_000,
                }
            ],
        },
        {
            "name": "High-Quality Code",
            "target_tokens": 7_000_000,
            "raw_filename": "coding_clean.jsonl",
            "is_code": True,
            "domain_type": "code",
            "sources": [
                {
                    "hf_path": "flytech/python-codes-25k",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "iamtarun/python_code_instructions_18k_alpaca",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "sahil2801/CodeAlpaca-20k",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "nomic-ai/gpt4all_prompt_generations",
                    "name_kw": None,
                    "skip_docs": 0,
                },
            ],
        },
        {
            "name": "Mathematics",
            "target_tokens": 5_000_000,
            "raw_filename": "math_clean.jsonl",
            "is_code": False,
            "domain_type": "math",
            "sources": [
                {
                    "hf_path": "tokyotech-llm/swallow-math",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "ProCreations/SimpleMath",
                    "name_kw": None,
                    "skip_docs": 0,
                },
            ],
        },
        {
            "name": "High-Quality Q&A",
            "target_tokens": 5_000_000,
            "raw_filename": "qa_clean.jsonl",
            "is_code": False,
            "domain_type": "qa",
            "sources": [
                {
                    "hf_path": "tatsu-lab/alpaca",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "cosmopedia-v2",
                    "skip_docs": 50_000,
                },
            ],
        },
        {
            "name": "Wikipedia",
            "target_tokens": 13_000_000,
            "raw_filename": "wikipedia_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "sources": [
                {
                    "hf_path": "wikimedia/wikipedia",
                    "name_kw": "20231101.en",
                    "skip_docs": 0,
                },
            ],
        },
        {
            "name": "Stories",
            "target_tokens": 2_000_000,
            "raw_filename": "stories_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "sources": [
                {
                    "hf_path": "roneneldan/TinyStories",
                    "name_kw": None,
                    "skip_docs": 0,
                },
            ],
        },
    ]

    all_train_docs_tokens = []  # List of list[int] for train docs
    all_val_docs_tokens = []    # List of list[int] for val docs

    manifest_domains = []
    total_start_time = time.time()
    grand_total_tokens = 0

    for spec in domain_specs:
        domain_name = spec["name"]
        target_tokens = spec["target_tokens"]
        raw_filepath = raw_dir / spec["raw_filename"]
        is_code = spec["is_code"]
        domain_type = spec["domain_type"]

        print(f"\n---> Collecting {target_tokens:,} tokens for domain: [{domain_name}]")
        domain_start_t = time.time()

        domain_collected_tokens = 0
        raw_doc_count = 0
        accepted_doc_count = 0
        rejected_duplicate_count = 0
        rejected_quality_count = 0

        with open(raw_filepath, "w", encoding="utf-8") as raw_file:
            for s_info in spec["sources"]:
                if domain_collected_tokens >= target_tokens:
                    break

                hf_path = s_info["hf_path"]
                skip_docs = s_info.get("skip_docs", 0)
                print(f"     Streaming dataset: {hf_path}" + (f" (skipping first {skip_docs:,} docs)" if skip_docs > 0 else ""))

                load_kwargs = {"split": "train", "streaming": True}
                if s_info.get("name_kw"):
                    load_kwargs["name"] = s_info["name_kw"]

                try:
                    ds = load_dataset(hf_path, **load_kwargs)
                    if skip_docs > 0:
                        ds = ds.skip(skip_docs)
                except Exception as err:
                    print(f"     Warning: Could not load {hf_path}: {err}")
                    continue

                for doc in ds:
                    raw_doc_count += 1
                    raw_text = extract_text_from_doc(doc, is_code=is_code, domain_type=domain_type)

                    if not raw_text or len(raw_text) < 10:
                        rejected_quality_count += 1
                        continue

                    # 1. Cleaning
                    cleaned = clean_text(raw_text, is_code=is_code)

                    # 2. Strict Quality Filtering
                    if not is_valid_quality(cleaned, is_code=is_code):
                        rejected_quality_count += 1
                        continue

                    # 3. Exact & Cross-Source Deduplication
                    d_hash = compute_doc_hash(cleaned)
                    if d_hash in seen_hashes:
                        rejected_duplicate_count += 1
                        continue
                    seen_hashes.add(d_hash)

                    # 4. Tokenization & EOS append
                    encoded = tokenizer.encode(cleaned)
                    ids = encoded.ids
                    if not ids or len(ids) < 5:
                        rejected_quality_count += 1
                        continue

                    ids.append(eos_id)
                    doc_tokens = len(ids)

                    # 5. Discrete Train/Val split (5% val, 95% train by document hash)
                    # Use hash mod 100 to deterministically isolate val documents without overlap
                    is_val = (int(d_hash[:8], 16) % 100) < 5
                    if is_val:
                        all_val_docs_tokens.append(ids)
                    else:
                        all_train_docs_tokens.append(ids)

                    domain_collected_tokens += doc_tokens
                    accepted_doc_count += 1
                    grand_total_tokens += doc_tokens

                    # Write raw clean doc
                    raw_file.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")

                    if accepted_doc_count % 1000 == 0:
                        print(f"     Progress [{domain_name}]: {domain_collected_tokens:,} / {target_tokens:,} tokens ({domain_collected_tokens/target_tokens*100:.1f}%)", flush=True)

                    if domain_collected_tokens >= target_tokens:
                        break

        domain_elapsed = time.time() - domain_start_t
        print(f"     Finished [{domain_name}]: Collected {domain_collected_tokens:,} tokens in {domain_elapsed:.1f}s")
        print(f"     Accepted docs: {accepted_doc_count:,} | Duplicates rejected: {rejected_duplicate_count:,} | Quality rejected: {rejected_quality_count:,}")

        manifest_domains.append({
            "domain_name": domain_name,
            "raw_filename": spec["raw_filename"],
            "target_tokens": target_tokens,
            "collected_tokens": domain_collected_tokens,
            "token_percentage": round((domain_collected_tokens / 60_000_000) * 100, 2),
            "raw_doc_count": raw_doc_count,
            "accepted_doc_count": accepted_doc_count,
            "rejected_duplicate_count": rejected_duplicate_count,
            "rejected_quality_count": rejected_quality_count,
        })

    # Flatten train and val tokens
    flatten_train_tokens = [t for doc in all_train_docs_tokens for t in doc]
    flatten_val_tokens = [t for doc in all_val_docs_tokens for t in doc]

    print("\n" + "=" * 80)
    print(f"  TOTAL COLLECTED CORPUS TOKENS: {grand_total_tokens:,}")
    print(f"  Train Tokens: {len(flatten_train_tokens):,} | Val Tokens: {len(flatten_val_tokens):,}")
    print("=" * 80)

    # Helper to write uint16 binary shards
    def write_shards(tokens_list, out_dir, prefix="shard", shard_size=5_000_000):
        shards_info = []
        num_shards = (len(tokens_list) + shard_size - 1) // shard_size
        print(f"\nWriting {len(tokens_list):,} tokens into {num_shards} shard(s) at {out_dir}...")

        for i in range(num_shards):
            chunk = tokens_list[i * shard_size : (i + 1) * shard_size]
            bin_path = out_dir / f"{prefix}_{i:05d}.bin"
            meta_path = out_dir / f"{prefix}_{i:05d}.json"

            arr = np.array(chunk, dtype=np.uint16)
            with open(bin_path, "wb") as f:
                f.write(arr.tobytes())

            meta = {
                "shard_index": i,
                "num_tokens": len(chunk),
                "dtype": "uint16",
                "bytes": bin_path.stat().st_size,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            sha256_checksum = compute_file_sha256(bin_path)
            shards_info.append({
                "filename": bin_path.name,
                "num_tokens": len(chunk),
                "bytes": bin_path.stat().st_size,
                "sha256": sha256_checksum,
            })
            print(f"  Wrote {bin_path.name}: {len(chunk):,} tokens ({bin_path.stat().st_size / 1e6:.2f} MB)")

        return shards_info

    train_shards = write_shards(flatten_train_tokens, shards_train_dir, prefix="shard", shard_size=5_000_000)
    val_shards = write_shards(flatten_val_tokens, shards_val_dir, prefix="shard", shard_size=5_000_000)

    total_elapsed = time.time() - total_start_time

    # Generate Manifest
    manifest = {
        "dataset_name": "50M Token Multi-Domain SLM Pretraining Corpus",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_corpus_tokens": grand_total_tokens,
        "train_tokens": len(flatten_train_tokens),
        "val_tokens": len(flatten_val_tokens),
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": eos_id,
        "total_build_time_seconds": round(total_elapsed, 1),
        "domain_mixture": manifest_domains,
        "shards": {
            "train": train_shards,
            "val": val_shards,
        },
    }

    manifest_path = Path("data/manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved dataset manifest to {manifest_path}")
    print(f"SUCCESS: 50M token multi-domain corpus built in {total_elapsed:.1f}s!")


if __name__ == "__main__":
    build_50m_corpus()
