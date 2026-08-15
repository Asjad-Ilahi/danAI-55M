"""
100M Token Multi-Domain Pretraining Corpus Builder Script.

Target Corpus Mixture (Total: 100,000,000 Tokens):
- 25% Science & Textbooks (25,000,000 tokens) -> data_100m/raw/science_clean.jsonl
- 25% General Knowledge (25,000,000 tokens)    -> data_100m/raw/general_knowledge_clean.jsonl
- 15% Multi-Language Code (15,000,000 tokens)  -> data_100m/raw/coding_clean.jsonl
- 15% Structured Q&A (15,000,000 tokens)       -> data_100m/raw/qa_clean.jsonl
- 10% Mathematics & Arithmetic (10,000,000 tok)-> data_100m/raw/math_clean.jsonl
- 10% Stories (10,000,000 tokens)              -> data_100m/raw/stories_clean.jsonl

Key Features:
1. Strict global SHA-256 deduplication against all past datasets (30M archive + 60M corpus).
2. Explicit streaming offsets (skip_docs) to avoid previously consumed streaming positions.
3. Dedicated foundational arithmetic generator (1+1=2, column arithmetic, times tables, word math).
4. Full multi-language code support (Python, JavaScript, HTML, Go, Java, C++, SQL, Rust).
5. uint16 binary tokenization with segment masking (_seg.bin) for document-aware causal attention.
"""

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from src.data.cleaner import clean_text, is_valid_quality, strip_html_tags


def compute_doc_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text for exact document deduplication."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_all_past_hashes() -> Set[str]:
    """Scan all past raw dataset files or load cached hash index to guarantee 100% fresh, unseen documents."""
    archive_hashes = set()
    gz_index = Path("data/historical_doc_hashes.json.gz")

    print("=" * 80)
    print("  LOADING PAST DOCUMENT HASHES FOR 100% FRESHNESS GUARANTEE")
    print("=" * 80)

    if gz_index.exists():
        import gzip
        print(f"  Loading precomputed historical hash registry: {gz_index}...", flush=True)
        try:
            with gzip.open(gz_index, "rt", encoding="utf-8") as f:
                loaded = json.load(f)
                archive_hashes.update(loaded)
            print(f"✓ Total unique historical document hashes loaded from registry: {len(archive_hashes):,}\n")
            return archive_hashes
        except Exception as e:
            print(f"  Warning: failed to load {gz_index} ({e}), falling back to disk scan.")

    paths_to_scan = [
        Path("data/archive_30m/raw"),
        Path("data/raw"),
        Path("data_100m/raw"),
    ]

    for p in paths_to_scan:
        if not p.exists():
            continue
        for jsonl_file in p.glob("*.jsonl"):
            print(f"  Scanning archive file: {jsonl_file}...", flush=True)
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

    print(f"✓ Total unique historical document hashes loaded: {len(archive_hashes):,}\n")
    return archive_hashes


# ==============================================================================
#  FOUNDATIONAL ARITHMETIC & NUMBER SENSE GENERATOR
# ==============================================================================

def generate_foundational_math_corpus(target_tokens: int, tokenizer: Tokenizer) -> List[str]:
    """
    Generate rich, varied foundational arithmetic documents:
    - Single & double digit basic arithmetic: 1+1=2, 5*5=25, 10-3=7, etc.
    - Times tables (1-20) with repeated addition explanations.
    - Step-by-step column addition, borrowing subtraction, long multiplication.
    - Word arithmetic ("If Alice has 1 apple and Bob gives her 1 apple...").
    - Order of operations (PEMDAS / BODMAS) step-by-step.
    - Fractions, decimals, percentages, and basic algebra basics.
    """
    print(f"---> Generating foundational arithmetic corpus (target: {target_tokens:,} tokens)...")
    docs = []
    current_tokens = 0
    rng = random.Random(42)

    # 1. Times tables 1 to 20
    for n in range(1, 21):
        lines = [f"# Multiplication Table for {n}\n"]
        for m in range(1, 13):
            prod = n * m
            repeated = " + ".join([str(n)] * m) if m <= 6 else f"{n} repeated {m} times"
            lines.append(f"{n} × {m} = {prod} (because {repeated} = {prod})")
            lines.append(f"Question: What is {n} multiplied by {m}?")
            lines.append(f"Answer: {n} × {m} = {prod}.\n")
        doc_str = "\n".join(lines)
        docs.append(doc_str)
        current_tokens += len(tokenizer.encode(doc_str).ids)

    # 2. Comprehensive 1+1 through 20+20 arithmetic facts
    fact_templates = [
        "What is {a} + {b}? The answer is {ans}.",
        "Question: Calculate {a} + {b}.\nAnswer: {a} + {b} = {ans}.",
        "If you have {a} items and add {b} more, you have {ans} in total: {a} + {b} = {ans}.",
        "Problem: {a} + {b} = ?\nSolution: Adding {b} to {a} gives {ans}.",
        "Basic arithmetic fact: {a} + {b} = {ans}.",
        "Math fact: {a} plus {b} equals {ans} ({a} + {b} = {ans}).",
    ]
    for a in range(1, 21):
        for b in range(1, 21):
            ans = a + b
            lines = [f"# Addition Facts for {a} + {b}\n"]
            for tmpl in fact_templates:
                lines.append(tmpl.format(a=a, b=b, ans=ans))
            lines.append(f"Verification: Start at {a} on the number line, count forward {b} steps to land on {ans}.")
            doc_str = "\n".join(lines)
            docs.append(doc_str)
            current_tokens += len(tokenizer.encode(doc_str).ids)

    # 3. Subtraction facts
    sub_templates = [
        "What is {a} - {b}? The answer is {ans}.",
        "Question: What is {a} minus {b}?\nAnswer: {a} - {b} = {ans}.",
        "If you have {a} objects and take away {b}, you are left with {ans}: {a} - {b} = {ans}.",
        "Subtraction is the inverse of addition: since {b} + {ans} = {a}, we know {a} - {b} = {ans}.",
    ]
    for a in range(1, 30):
        for b in range(1, a + 1):
            ans = a - b
            lines = [f"# Subtraction Facts: {a} - {b}\n"]
            for tmpl in sub_templates:
                lines.append(tmpl.format(a=a, b=b, ans=ans))
            doc_str = "\n".join(lines)
            docs.append(doc_str)
            current_tokens += len(tokenizer.encode(doc_str).ids)

    # 4. Step-by-step multi-digit addition with carry explanation
    while current_tokens < target_tokens:
        batch_lines = []
        doc_type = rng.choice(["column_add", "column_sub", "word_math", "pemdas", "fractions", "division"])

        if doc_type == "column_add":
            x = rng.randint(10, 999)
            y = rng.randint(10, 999)
            res = x + y
            batch_lines.append(f"Problem: Calculate {x} + {y} step by step.\n")
            batch_lines.append("Step-by-step Solution:")
            batch_lines.append(f"1. Line up the numbers vertically by place value:")
            batch_lines.append(f"     {x:4d}")
            batch_lines.append(f"   + {y:4d}")
            batch_lines.append(f"   ------")
            batch_lines.append(f"     {res:4d}")
            batch_lines.append(f"2. Add the ones column: {x % 10} + {y % 10} = {(x%10)+(y%10)}.")
            batch_lines.append(f"3. Add the tens column: {(x//10)%10} + {(y//10)%10} with any carry.")
            batch_lines.append(f"4. Add the hundreds column: {x//100} + {y//100} with any carry.")
            batch_lines.append(f"\nFinal Answer: {x} + {y} = {res}.")

        elif doc_type == "column_sub":
            x = rng.randint(20, 999)
            y = rng.randint(1, x)
            res = x - y
            batch_lines.append(f"Problem: What is {x} - {y}?\n")
            batch_lines.append(f"Solution:\nTo subtract {y} from {x}:")
            batch_lines.append(f"We can verify using addition: {res} + {y} = {x}.")
            batch_lines.append(f"Therefore, {x} - {y} = {res}.")

        elif doc_type == "word_math":
            names = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "George", "Hannah"]
            items = ["apples", "books", "marbles", "pencils", "stickers", "candies", "coins", "stamps"]
            p1, p2 = rng.sample(names, 2)
            item = rng.choice(items)
            n1 = rng.randint(1, 50)
            n2 = rng.randint(1, 50)
            tot = n1 + n2
            batch_lines.append(f"Word Problem: {p1} has {n1} {item}. {p2} gives {p1} {n2} more {item}. How many {item} does {p1} have now?\n")
            batch_lines.append("Solution:")
            batch_lines.append(f"- Initial count of {item} = {n1}")
            batch_lines.append(f"- Additional count given = {n2}")
            batch_lines.append(f"- Total = {n1} + {n2} = {tot}")
            batch_lines.append(f"Answer: {p1} now has {tot} {item}.")

        elif doc_type == "pemdas":
            a = rng.randint(1, 15)
            b = rng.randint(1, 10)
            c = rng.randint(1, 10)
            res = a + (b * c)
            batch_lines.append(f"Problem: Evaluate the expression {a} + {b} × {c}.\n")
            batch_lines.append("Solution:")
            batch_lines.append("According to the order of operations (PEMDAS / BODMAS):")
            batch_lines.append(f"1. Multiplication is evaluated before addition: {b} × {c} = {b * c}.")
            batch_lines.append(f"2. Then add {a}: {a} + {b * c} = {res}.")
            batch_lines.append(f"Final Answer: {res}.")

        elif doc_type == "fractions":
            d1 = rng.choice([2, 3, 4, 5, 6, 8, 10])
            n1 = rng.randint(1, d1 - 1)
            dec = round(n1 / d1, 4)
            pct = round((n1 / d1) * 100, 2)
            batch_lines.append(f"Math Concept: Converting Fraction {n1}/{d1} to Decimals and Percentages.\n")
            batch_lines.append(f"- Fraction: {n1}/{d1}")
            batch_lines.append(f"- Decimal representation: {n1} ÷ {d1} = {dec}")
            batch_lines.append(f"- Percentage representation: {dec} × 100% = {pct}%")
            batch_lines.append(f"Therefore, {n1}/{d1} is equal to {dec} or {pct}%.")

        elif doc_type == "division":
            divisor = rng.randint(2, 12)
            quotient = rng.randint(1, 20)
            dividend = divisor * quotient
            batch_lines.append(f"Question: What is {dividend} divided by {divisor}?\n")
            batch_lines.append("Answer:")
            batch_lines.append(f"{dividend} ÷ {divisor} = {quotient}.")
            batch_lines.append(f"Explanation: Since {divisor} × {quotient} = {dividend}, dividing {dividend} by {divisor} gives {quotient}.")

        doc_str = "\n".join(batch_lines)
        docs.append(doc_str)
        current_tokens += len(tokenizer.encode(doc_str).ids)

    print(f"✓ Generated {len(docs):,} foundational arithmetic documents ({current_tokens:,} tokens).")
    return docs


# ==============================================================================
#  TEXT EXTRACTION & CLEANING PER DOMAIN
# ==============================================================================

def extract_domain_text(doc: dict, domain_type: str) -> str:
    """Extract clean domain text from various Hugging Face schema structures."""
    if not isinstance(doc, dict):
        return str(doc)

    if domain_type == "code":
        # CodeFeedback schema: ['query', 'answer', 'lang']
        q = doc.get("query") or doc.get("instruction") or doc.get("prompt")
        a = doc.get("answer") or doc.get("response") or doc.get("output") or doc.get("code")
        lang = doc.get("lang") or doc.get("language") or ""
        if q and a:
            clean_q = strip_html_tags(str(q)).strip()
            clean_a = str(a).strip()
            header = f"# Programming Problem ({lang.upper() if lang else 'Code'})\n{clean_q}\n\n# Solution:\n"
            return f"{header}{clean_a}"
        elif a:
            return str(a).strip()
        elif doc.get("text"):
            return str(doc["text"]).strip()

    elif domain_type == "math":
        # MathInstruct / Orca Math / SimpleMath
        q = doc.get("question") or doc.get("problem") or doc.get("instruction")
        a = doc.get("answer") or doc.get("solution") or doc.get("output") or doc.get("response")
        if q and a:
            clean_q = strip_html_tags(str(q)).strip()
            clean_a = strip_html_tags(str(a)).strip()
            return f"Problem:\n{clean_q}\n\nSolution:\n{clean_a}"
        elif doc.get("text"):
            return str(doc["text"]).strip()

    elif domain_type == "science":
        # SciQ / Cosmopedia OpenStax / KhanAcademy
        if "correct_answer" in doc and "question" in doc:
            # SciQ format
            q = strip_html_tags(str(doc["question"])).strip()
            ans = strip_html_tags(str(doc["correct_answer"])).strip()
            support = strip_html_tags(str(doc.get("support", ""))).strip()
            out = f"Question: {q}\nAnswer: {ans}"
            if support:
                out += f"\n\nScientific Explanation:\n{support}"
            return out
        elif doc.get("text"):
            return str(doc["text"]).strip()

    elif domain_type == "qa":
        # SmolTalk messages format or OpenOrca
        if "messages" in doc and isinstance(doc["messages"], list):
            turns = []
            for m in doc["messages"]:
                role = m.get("role", "User").capitalize()
                content = strip_html_tags(str(m.get("content", ""))).strip()
                if content:
                    turns.append(f"{role}: {content}")
            return "\n\n".join(turns)
        elif "question" in doc and "response" in doc:
            q = strip_html_tags(str(doc["question"])).strip()
            r = strip_html_tags(str(doc["response"])).strip()
            return f"Question:\n{q}\n\nAnswer:\n{r}"
        elif doc.get("text"):
            return str(doc["text"]).strip()

    elif domain_type == "general":
        # Wikipedia / TinyStories / General
        if "title" in doc and "text" in doc:
            title = str(doc["title"]).strip()
            text = str(doc["text"]).strip()
            return f"# {title}\n\n{text}"
        elif doc.get("text"):
            return str(doc["text"]).strip()

    # Generic fallback
    for k in ["text", "content", "article", "response", "solution"]:
        if k in doc and isinstance(doc[k], str) and len(doc[k].strip()) > 10:
            return doc[k].strip()

    return ""


# ==============================================================================
#  MAIN 100M BUILDER ROUTINE
# ==============================================================================

def build_100m_corpus():
    print("=" * 80)
    print("  BUILDING 100,000,000 TOKEN MULTI-DOMAIN PRETRAINING CORPUS")
    print("=" * 80)

    tokenizer_path = Path("tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = 0

    raw_dir = Path("data_100m/raw")
    shards_train_dir = Path("data_100m/shards/train")
    shards_val_dir = Path("data_100m/shards/val")

    raw_dir.mkdir(parents=True, exist_ok=True)
    shards_train_dir.mkdir(parents=True, exist_ok=True)
    shards_val_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all previous hashes to guarantee 100% fresh data
    seen_hashes = load_all_past_hashes()

    # 2. Multi-Domain Specifications (Total: 100,000,000 Tokens)
    domain_specs = [
        # Domain 1: Mathematics & Foundational Arithmetic (10M tokens)
        {
            "name": "Mathematics & Arithmetic",
            "target_tokens": 10_000_000,
            "raw_filename": "math_clean.jsonl",
            "is_code": False,
            "domain_type": "math",
            "synthetic_generator": True,
            "synthetic_tokens": 3_500_000,  # 3.5M dedicated arithmetic (1+1=2, step-by-step proofs)
            "sources": [
                {
                    "hf_path": "TIGER-Lab/MathInstruct",
                    "name_kw": None,
                    "skip_docs": 10_000,
                },
                {
                    "hf_path": "microsoft/orca-math-word-problems-200k",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "ProCreations/SimpleMath",
                    "name_kw": None,
                    "skip_docs": 10_000,
                },
            ],
        },
        # Domain 2: Science & Textbooks (25M tokens)
        {
            "name": "Science & Textbooks",
            "target_tokens": 25_000_000,
            "raw_filename": "science_clean.jsonl",
            "is_code": False,
            "domain_type": "science",
            "synthetic_generator": False,
            "sources": [
                {
                    "hf_path": "HuggingFaceTB/cosmopedia",
                    "name_kw": "openstax",
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/cosmopedia",
                    "name_kw": "khanacademy",
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/cosmopedia",
                    "name_kw": "auto_math_text",
                    "skip_docs": 0,
                },
                {
                    "hf_path": "allenai/sciq",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/cosmopedia",
                    "name_kw": "web_samples_v1",
                    "skip_docs": 50_000,
                },
            ],
        },
        # Domain 3: General Knowledge & World Facts (25M tokens)
        {
            "name": "General Knowledge & World Facts",
            "target_tokens": 25_000_000,
            "raw_filename": "general_knowledge_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "synthetic_generator": False,
            "sources": [
                {
                    "hf_path": "wikimedia/wikipedia",
                    "name_kw": "20231101.en",
                    "skip_docs": 150_000,  # Explicitly skip docs examined in earlier runs
                },
            ],
        },
        # Domain 4: Multi-Language Coding (15M tokens)
        {
            "name": "Multi-Language Coding",
            "target_tokens": 15_000_000,
            "raw_filename": "coding_clean.jsonl",
            "is_code": True,
            "domain_type": "code",
            "synthetic_generator": False,
            "sources": [
                {
                    "hf_path": "m-a-p/CodeFeedback-Filtered-Instruction",
                    "name_kw": None,
                    "skip_docs": 0,
                },
                {
                    "hf_path": "HuggingFaceTB/smollm-corpus",
                    "name_kw": "python-edu",
                    "skip_docs": 20_000,
                },
            ],
        },
        # Domain 5: Structured Q&A & Reasoning (15M tokens)
        {
            "name": "Structured Q&A & Reasoning",
            "target_tokens": 15_000_000,
            "raw_filename": "qa_clean.jsonl",
            "is_code": False,
            "domain_type": "qa",
            "synthetic_generator": False,
            "sources": [
                {
                    "hf_path": "HuggingFaceTB/smoltalk",
                    "name_kw": "all",
                    "skip_docs": 0,
                },
                {
                    "hf_path": "Open-Orca/OpenOrca",
                    "name_kw": None,
                    "skip_docs": 50_000,
                },
            ],
        },
        # Domain 6: Stories & Narrative (10M tokens)
        {
            "name": "Stories & Narrative",
            "target_tokens": 10_000_000,
            "raw_filename": "stories_clean.jsonl",
            "is_code": False,
            "domain_type": "general",
            "synthetic_generator": False,
            "sources": [
                {
                    "hf_path": "roneneldan/TinyStories",
                    "name_kw": None,
                    "skip_docs": 250_000,  # Unseen stories offset
                },
            ],
        },
    ]

    all_train_docs_tokens: List[List[int]] = []
    all_val_docs_tokens: List[List[int]] = []

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
            # 1. Check if synthetic generator is configured
            if spec.get("synthetic_generator"):
                synth_target = spec.get("synthetic_tokens", 3_000_000)
                synth_docs = generate_foundational_math_corpus(synth_target, tokenizer)
                for doc_text in synth_docs:
                    raw_doc_count += 1
                    cleaned = clean_text(doc_text, is_code=is_code)
                    d_hash = compute_doc_hash(cleaned)
                    if d_hash in seen_hashes:
                        rejected_duplicate_count += 1
                        continue
                    seen_hashes.add(d_hash)

                    encoded = tokenizer.encode(cleaned)
                    ids = encoded.ids
                    ids.append(eos_id)
                    doc_tokens = len(ids)

                    is_val = (int(d_hash[:8], 16) % 100) < 5
                    if is_val:
                        all_val_docs_tokens.append(ids)
                    else:
                        all_train_docs_tokens.append(ids)

                    domain_collected_tokens += doc_tokens
                    accepted_doc_count += 1
                    grand_total_tokens += doc_tokens
                    raw_file.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")

            # 2. Stream from configured Hugging Face sources
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
                    raw_text = extract_domain_text(doc, domain_type=domain_type)

                    if not raw_text or len(raw_text) < 15:
                        rejected_quality_count += 1
                        continue

                    cleaned = clean_text(raw_text, is_code=is_code)
                    if not is_valid_quality(cleaned, is_code=is_code):
                        rejected_quality_count += 1
                        continue

                    d_hash = compute_doc_hash(cleaned)
                    if d_hash in seen_hashes:
                        rejected_duplicate_count += 1
                        continue
                    seen_hashes.add(d_hash)

                    encoded = tokenizer.encode(cleaned)
                    ids = encoded.ids
                    if not ids or len(ids) < 5:
                        rejected_quality_count += 1
                        continue

                    ids.append(eos_id)
                    doc_tokens = len(ids)

                    is_val = (int(d_hash[:8], 16) % 100) < 5
                    if is_val:
                        all_val_docs_tokens.append(ids)
                    else:
                        all_train_docs_tokens.append(ids)

                    domain_collected_tokens += doc_tokens
                    accepted_doc_count += 1
                    grand_total_tokens += doc_tokens

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
            "token_percentage": round((domain_collected_tokens / 100_000_000) * 100, 2),
            "raw_doc_count": raw_doc_count,
            "accepted_doc_count": accepted_doc_count,
            "rejected_duplicate_count": rejected_duplicate_count,
            "rejected_quality_count": rejected_quality_count,
        })

    # Flatten train and val tokens and create segment IDs for document-aware masking
    print("\n" + "=" * 80)
    print(f"  TOTAL COLLECTED 100M CORPUS TOKENS: {grand_total_tokens:,}")
    print("=" * 80)

    def write_shards_with_segments(docs_tokens: List[List[int]], out_dir: Path, prefix="shard", shard_size=5_000_000):
        out_dir.mkdir(parents=True, exist_ok=True)
        shards_info = []

        flat_tokens = []
        flat_segments = []

        for doc_idx, doc in enumerate(docs_tokens):
            seg_id = doc_idx % 65535  # uint16 range
            flat_tokens.extend(doc)
            flat_segments.extend([seg_id] * len(doc))

        total_toks = len(flat_tokens)
        num_shards = (total_toks + shard_size - 1) // shard_size
        print(f"\nWriting {total_toks:,} tokens into {num_shards} shard(s) with segment boundaries at {out_dir}...")

        for i in range(num_shards):
            tok_chunk = flat_tokens[i * shard_size : (i + 1) * shard_size]
            seg_chunk = flat_segments[i * shard_size : (i + 1) * shard_size]

            bin_path = out_dir / f"{prefix}_{i:05d}.bin"
            seg_path = out_dir / f"{prefix}_{i:05d}_seg.bin"
            meta_path = out_dir / f"{prefix}_{i:05d}.json"

            tok_arr = np.array(tok_chunk, dtype=np.uint16)
            with open(bin_path, "wb") as f:
                f.write(tok_arr.tobytes())

            seg_arr = np.array(seg_chunk, dtype=np.uint16)
            with open(seg_path, "wb") as f:
                f.write(seg_arr.tobytes())

            meta = {
                "shard_index": i,
                "num_tokens": len(tok_chunk),
                "dtype": "uint16",
                "bytes": bin_path.stat().st_size,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            sha256_checksum = compute_file_sha256(bin_path)
            shards_info.append({
                "filename": bin_path.name,
                "num_tokens": len(tok_chunk),
                "bytes": bin_path.stat().st_size,
                "sha256": sha256_checksum,
            })
            print(f"  Wrote {bin_path.name} & {seg_path.name}: {len(tok_chunk):,} tokens ({bin_path.stat().st_size / 1e6:.2f} MB)")

        return shards_info, total_toks

    train_shards, total_train_tokens = write_shards_with_segments(all_train_docs_tokens, shards_train_dir, prefix="shard", shard_size=5_000_000)
    val_shards, total_val_tokens = write_shards_with_segments(all_val_docs_tokens, shards_val_dir, prefix="shard", shard_size=5_000_000)

    total_elapsed = time.time() - total_start_time

    # Generate Manifest
    manifest = {
        "dataset_name": "100M Token Multi-Domain SLM Continuation Corpus",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_corpus_tokens": grand_total_tokens,
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token_id": eos_id,
        "total_build_time_seconds": round(total_elapsed, 1),
        "domain_mixture": manifest_domains,
        "shards": {
            "train": train_shards,
            "val": val_shards,
        },
    }

    manifest_path = Path("data_100m/manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved dataset manifest to {manifest_path}")
    print(f"SUCCESS: 100M token multi-domain corpus built in {total_elapsed:.1f}s!")


if __name__ == "__main__":
    build_100m_corpus()
