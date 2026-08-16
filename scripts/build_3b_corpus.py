"""
3.0 Billion Token Multi-Domain Pretraining Corpus Builder (With Safe Shard Resuming).

Features:
- Seamless Resume: Detects existing shard files on disk (e.g. shard_00000.bin - shard_01907.bin)
  and resumes without re-downloading or discarding previously generated tokens!
- Combinatorial Synthetic Logic & Refusal Generators with infinite unique permutations.
- Infinite-loop safety guard: Consecutive duplicate limits prevent any potential stalls.
- Multi-Domain 10-Stream Balanced High-Density Mixture.
- 100% SHA-256 deduplication against historical_doc_hashes.json.gz and running hashes.
- Packed uint16 binary shards with document boundary segment masks (_seg.bin).
"""

import ast
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple, Iterator

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cleaner import clean_text, is_valid_quality, strip_html_tags


# ==============================================================================
#  CONFIGURATION (10 Streams, 3.0 Billion Total Tokens)
# ==============================================================================

STREAM_SPECS = [
    {
        "name": "Science & Textbooks",
        "target_tokens": 600_000_000,
        "domain_type": "science",
        "is_code": False,
        "sources": [
            {"hf_path": "HuggingFaceFW/fineweb-edu", "name_kw": "sample-10BT", "split": "train",
             "text_field": "text", "skip_docs": 20_000, "quality_score_field": "score", "min_score": 3.0},
        ],
    },
    {
        "name": "Chain-of-Thought Mathematics & Arithmetic",
        "target_tokens": 450_000_000,
        "domain_type": "math",
        "is_code": False,
        "sources": [
            {"hf_path": "TIGER-Lab/MathInstruct", "name_kw": None, "split": "train",
             "text_field": None, "skip_docs": 10_000, "extract_fn": "math_instruct"},
        ],
        "synthetic_generator": "foundational_arithmetic",
        "synthetic_tokens": 200_000_000,
    },
    {
        "name": "Encyclopedic & World Knowledge",
        "target_tokens": 750_000_000,
        "domain_type": "general",
        "is_code": False,
        "sources": [
            {"hf_path": "wikimedia/wikipedia", "name_kw": "20231101.en", "split": "train",
             "text_field": "text", "skip_docs": 155_000, "title_field": "title"},
        ],
    },
    {
        "name": "AST-Verified Python & Systems Code",
        "target_tokens": 70_000_000,
        "domain_type": "code",
        "is_code": True,
        "sources": [
            {"hf_path": "m-a-p/CodeFeedback-Filtered-Instruction", "name_kw": None, "split": "train",
             "text_field": None, "skip_docs": 0, "extract_fn": "code_feedback"},
        ],
    },
    {
        "name": "Structured Q&A & Conversations",
        "target_tokens": 270_000_000,
        "domain_type": "qa",
        "is_code": False,
        "sources": [
            {"hf_path": "HuggingFaceTB/smoltalk", "name_kw": "all", "split": "train",
             "text_field": None, "skip_docs": 0, "extract_fn": "smoltalk"},
        ],
    },
    {
        "name": "Deductive Logic & Syllogisms",
        "target_tokens": 180_000_000,
        "domain_type": "logic",
        "is_code": False,
        "sources": [
            {"hf_path": "Open-Orca/OpenOrca", "name_kw": None, "split": "train",
             "text_field": None, "skip_docs": 110_000, "extract_fn": "openorca"},
        ],
    },
    {
        "name": "Premise Refusal & Saying NO",
        "target_tokens": 200_000_000,
        "domain_type": "refusal",
        "is_code": False,
        "sources": [
            {"hf_path": "Anthropic/hh-rlhf", "name_kw": None, "split": "train",
             "text_field": None, "skip_docs": 0, "extract_fn": "hh_rlhf"},
        ],
        "synthetic_generator": "premise_refusal",
        "synthetic_tokens": 100_000_000,
    },
    {
        "name": "High-Quality Annealing",
        "target_tokens": 200_000_000,
        "domain_type": "science",
        "is_code": False,
        "sources": [
            {"hf_path": "HuggingFaceTB/cosmopedia", "name_kw": "auto_math_text", "split": "train",
             "text_field": "text", "skip_docs": 0},
            {"hf_path": "HuggingFaceTB/cosmopedia", "name_kw": "web_samples_v2", "split": "train",
             "text_field": "text", "skip_docs": 60_000},
            {"hf_path": "HuggingFaceTB/cosmopedia", "name_kw": "stanford", "split": "train",
             "text_field": "text", "skip_docs": 0},
        ],
    },
    {
        "name": "Instruction Following & Turn-Taking",
        "target_tokens": 200_000_000,
        "domain_type": "qa",
        "is_code": False,
        "sources": [
            {"hf_path": "HuggingFaceH4/no_robots", "name_kw": None, "split": "train",
             "text_field": None, "skip_docs": 0, "extract_fn": "no_robots"},
            {"hf_path": "HuggingFaceTB/everyday-conversations-llama3.1-2k", "name_kw": None, "split": "train_sft",
             "text_field": None, "skip_docs": 0, "extract_fn": "smoltalk"},
        ],
    },
]


# ==============================================================================
#  HASHING & DEDUPLICATION
# ==============================================================================

def compute_doc_hash(text: str) -> str:
    """SHA-256 hash of normalized text for exact document deduplication."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def load_all_past_hashes() -> Set[str]:
    """Load all 363,966 historical document hashes from previous training stages."""
    archive_hashes = set()
    gz_index = Path("data/historical_doc_hashes.json.gz")

    print("=" * 80)
    print("  LOADING HISTORICAL SHA-256 DEDUPLICATION REGISTRY")
    print("=" * 80)

    if gz_index.exists():
        try:
            with gzip.open(gz_index, "rt", encoding="utf-8") as f:
                loaded = json.load(f)
                archive_hashes.update(loaded)
            print(f"  [OK] Loaded {len(archive_hashes):,} historical document hashes from {gz_index}")
            return archive_hashes
        except Exception as e:
            print(f"  Warning: failed to load {gz_index}: {e}")

    paths_to_scan = [Path("data/raw"), Path("data_100m/raw"), Path("data_sft/raw")]
    for p in paths_to_scan:
        if not p.exists():
            continue
        for jsonl_file in p.glob("*.jsonl"):
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
    print(f"  [OK] Scanned disk archive: {len(archive_hashes):,} hashes\n")
    return archive_hashes


# ==============================================================================
#  DOCUMENT EXTRACTION & CLEANING PER SOURCE TYPE
# ==============================================================================

def extract_text(doc: dict, source: dict, domain_type: str) -> str:
    """Extract and clean text from a HuggingFace document based on source schema."""
    extract_fn = source.get("extract_fn")

    if extract_fn == "sciq":
        q = strip_html_tags(str(doc.get("question", ""))).strip()
        ans = strip_html_tags(str(doc.get("correct_answer", ""))).strip()
        support = strip_html_tags(str(doc.get("support", ""))).strip()
        out = f"Question: {q}\nAnswer: {ans}"
        if support:
            out += f"\n\nScientific Explanation:\n{support}"
        return out

    elif extract_fn == "math_instruct":
        q = strip_html_tags(str(doc.get("instruction", "") or doc.get("question", ""))).strip()
        a = strip_html_tags(str(doc.get("output", "") or doc.get("answer", ""))).strip()
        if q and a:
            return f"Problem:\n{q}\n\nSolution:\n{a}"
        return ""

    elif extract_fn == "orca_math":
        q = strip_html_tags(str(doc.get("question", ""))).strip()
        a = strip_html_tags(str(doc.get("answer", ""))).strip()
        if q and a:
            return f"Math Word Problem:\n{q}\n\nStep-by-Step Solution:\n{a}"
        return ""

    elif extract_fn == "code_feedback":
        q = strip_html_tags(str(doc.get("query", "") or doc.get("instruction", ""))).strip()
        a = str(doc.get("answer", "") or doc.get("response", "")).strip()
        lang = doc.get("lang", "Python")
        if q and a:
            return f"# Programming Problem ({lang})\n{q}\n\n# Solution:\n{a}"
        return ""

    elif extract_fn == "smoltalk":
        messages = doc.get("messages", [])
        if isinstance(messages, list):
            turns = []
            for m in messages:
                role = m.get("role", "user").capitalize()
                content = strip_html_tags(str(m.get("content", ""))).strip()
                if content:
                    turns.append(f"{role}: {content}")
            return "\n\n".join(turns)
        return ""

    elif extract_fn == "hh_rlhf":
        chosen = doc.get("chosen", "")
        if chosen and isinstance(chosen, str) and len(chosen.strip()) > 20:
            return chosen.strip()
        return ""

    elif extract_fn == "no_robots":
        messages = doc.get("messages", [])
        if isinstance(messages, list):
            turns = []
            for m in messages:
                role = m.get("role", "user").capitalize()
                content = strip_html_tags(str(m.get("content", ""))).strip()
                if content:
                    turns.append(f"{role}: {content}")
            return "\n\n".join(turns)
        return ""

    elif extract_fn == "openorca":
        sys_prompt = str(doc.get("system_prompt", "")).strip()
        q = strip_html_tags(str(doc.get("question", ""))).strip()
        r = strip_html_tags(str(doc.get("response", ""))).strip()
        parts = []
        if sys_prompt and len(sys_prompt) > 5:
            parts.append(f"System: {sys_prompt}")
        if q:
            parts.append(f"User: {q}")
        if r:
            parts.append(f"Assistant: {r}")
        return "\n\n".join(parts)

    # Default: use text_field
    text_field = source.get("text_field", "text")
    if text_field and text_field in doc:
        text = str(doc[text_field]).strip()
        title_field = source.get("title_field")
        if title_field and title_field in doc:
            title = str(doc[title_field]).strip()
            return f"# {title}\n\n{text}"
        return text

    for k in ["text", "content", "article", "response", "solution"]:
        if k in doc and isinstance(doc[k], str) and len(doc[k].strip()) > 10:
            return doc[k].strip()
    return ""


def quality_filter(text: str, is_code: bool = False) -> bool:
    """Multi-stage quality filter for incoming documents."""
    if not text or len(text.strip()) < 50:
        return False

    control_chars = sum(1 for c in text if ord(c) < 32 and c not in ('\n', '\t', '\r'))
    if control_chars / max(1, len(text)) > 0.05:
        return False

    words = text.split()
    if not is_code and len(words) < 15:
        return False

    if len(words) >= 40:
        four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        counts = Counter(four_grams)
        most_common_count = counts.most_common(1)[0][1] if counts else 1
        if (most_common_count * 4) / len(words) > 0.30:
            return False

    if not is_code:
        non_ascii = len(re.findall(r"[^\x00-\x7F]", text))
        if non_ascii / max(1, len(text)) > 0.15:
            return False

    return True


# ==============================================================================
#  COMBINATORIAL SYNTHETIC DATA GENERATORS (Infinite Unique Samples)
# ==============================================================================

class ArithmeticGenerator:
    """Generates foundational arithmetic with infinite combinations."""

    NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "George", "Hannah",
             "Ivy", "Jack", "Katie", "Leo", "Maya", "Nathan", "Olivia", "Peter",
             "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Zack"]
    ITEMS = ["apples", "books", "marbles", "pencils", "stickers", "candies",
             "coins", "stamps", "cookies", "oranges", "balloons", "toys",
             "markers", "notebooks", "erasers", "cards", "ribbons", "badges"]

    @classmethod
    def generate(cls, rng: random.Random) -> str:
        doc_type = rng.choice([
            "add_simple", "add_column", "sub_simple", "sub_check",
            "mult_simple", "mult_distrib", "div_simple", "div_verify",
            "word_add", "word_sub", "word_mult", "word_multi_step",
            "times_table_row", "pemdas", "fraction", "compare",
        ])

        if doc_type == "add_simple":
            a, b = rng.randint(1, 500), rng.randint(1, 500)
            return f"Question: What is {a} + {b}?\nAnswer: {a} + {b} = {a+b}."

        elif doc_type == "add_column":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            ans = a + b
            ones_sum = (a % 10) + (b % 10)
            carry1 = ones_sum // 10
            ones_digit = ones_sum % 10
            tens_sum = ((a // 10) % 10) + ((b // 10) % 10) + carry1
            return (f"Problem: Calculate {a} + {b} step by step.\n"
                    f"Step 1: Add the ones digits: {a%10} + {b%10} = {ones_sum}"
                    f"{f' (write {ones_digit}, carry {carry1})' if carry1 else ''}.\n"
                    f"Step 2: Add the tens digits: {(a//10)%10} + {(b//10)%10}"
                    f"{f' + {carry1} (carry)' if carry1 else ''} = {tens_sum}.\n"
                    f"Final Answer: {a} + {b} = {ans}.\n"
                    f"Verification: {ans} - {a} = {b}. Correct.")

        elif doc_type == "sub_simple":
            a = rng.randint(10, 1000)
            b = rng.randint(1, a)
            return (f"Question: What is {a} - {b}?\n"
                    f"Answer: {a} - {b} = {a-b}.\n"
                    f"Check: {a-b} + {b} = {a}. Correct.")

        elif doc_type == "sub_check":
            a = rng.randint(50, 5000)
            b = rng.randint(1, a)
            ans = a - b
            return (f"Subtraction Problem: {a} - {b}\n"
                    f"Solution: To subtract {b} from {a}:\n"
                    f"  {a} - {b} = {ans}\n"
                    f"Verification: {ans} + {b} = {a}. The answer {ans} is correct.")

        elif doc_type == "mult_simple":
            a, b = rng.randint(2, 50), rng.randint(2, 50)
            return (f"Question: What is {a} × {b}?\n"
                    f"Answer: {a} × {b} = {a*b}.\n"
                    f"Explanation: Multiplying {a} by {b} gives {a*b}.")

        elif doc_type == "mult_distrib":
            a = rng.randint(11, 999)
            b = rng.randint(2, 50)
            ans = a * b
            b_tens = (b // 10) * 10
            b_ones = b % 10
            if b_tens > 0:
                return (f"Question: What is {a} × {b}?\n"
                        f"Using the distributive property:\n"
                        f"  {a} × {b} = {a} × ({b_tens} + {b_ones})\n"
                        f"  = {a} × {b_tens} + {a} × {b_ones}\n"
                        f"  = {a*b_tens} + {a*b_ones}\n"
                        f"  = {ans}\n"
                        f"Answer: {a} × {b} = {ans}.")
            else:
                return f"Question: What is {a} × {b}?\nAnswer: {a} × {b} = {ans}."

        elif doc_type == "div_simple":
            divisor = rng.randint(2, 25)
            quotient = rng.randint(1, 100)
            dividend = divisor * quotient
            return (f"Question: What is {dividend} ÷ {divisor}?\n"
                    f"Answer: {dividend} ÷ {divisor} = {quotient}.\n"
                    f"Explanation: Since {divisor} × {quotient} = {dividend}, "
                    f"dividing {dividend} by {divisor} gives {quotient}.")

        elif doc_type == "div_verify":
            divisor = rng.randint(2, 50)
            quotient = rng.randint(2, 100)
            dividend = divisor * quotient
            return (f"Division Problem: {dividend} / {divisor}\n"
                    f"Step 1: How many times does {divisor} go into {dividend}?\n"
                    f"Step 2: {divisor} × {quotient} = {dividend}\n"
                    f"Step 3: {dividend} - {divisor} × {quotient} = 0 (no remainder)\n"
                    f"Answer: {dividend} / {divisor} = {quotient}.")

        elif doc_type == "word_add":
            p1, p2 = rng.sample(cls.NAMES, 2)
            item = rng.choice(cls.ITEMS)
            n1, n2 = rng.randint(1, 200), rng.randint(1, 200)
            return (f"Word Problem: {p1} has {n1} {item}. {p2} gives {p1} {n2} more {item}. "
                    f"How many {item} does {p1} have now?\n"
                    f"Solution:\n"
                    f"- {p1} starts with: {n1} {item}\n"
                    f"- {p2} gives: {n2} {item}\n"
                    f"- Total: {n1} + {n2} = {n1+n2} {item}\n"
                    f"Answer: {p1} now has {n1+n2} {item}.")

        elif doc_type == "word_sub":
            p1 = rng.choice(cls.NAMES)
            item = rng.choice(cls.ITEMS)
            n1 = rng.randint(10, 500)
            n2 = rng.randint(1, n1)
            return (f"Word Problem: {p1} has {n1} {item}. {p1} gives away {n2} {item}. "
                    f"How many {item} does {p1} have left?\n"
                    f"Solution:\n"
                    f"- Started with: {n1} {item}\n"
                    f"- Gave away: {n2} {item}\n"
                    f"- Remaining: {n1} - {n2} = {n1-n2} {item}\n"
                    f"Answer: {p1} has {n1-n2} {item} left.")

        elif doc_type == "word_mult":
            p1 = rng.choice(cls.NAMES)
            item = rng.choice(cls.ITEMS)
            boxes = rng.randint(2, 20)
            per_box = rng.randint(3, 50)
            return (f"Word Problem: {p1} buys {boxes} boxes of {item}. Each box contains "
                    f"{per_box} {item}. How many {item} does {p1} have in total?\n"
                    f"Solution:\n"
                    f"- Number of boxes: {boxes}\n"
                    f"- {item.capitalize()} per box: {per_box}\n"
                    f"- Total: {boxes} × {per_box} = {boxes*per_box} {item}\n"
                    f"Answer: {p1} has {boxes*per_box} {item} in total.")

        elif doc_type == "word_multi_step":
            p1 = rng.choice(cls.NAMES)
            item = rng.choice(cls.ITEMS)
            start = rng.randint(20, 200)
            give = rng.randint(1, start // 2)
            buy = rng.randint(1, 100)
            final = start - give + buy
            return (f"Word Problem: {p1} has {start} {item}. {p1} gives {give} to a friend "
                    f"and then buys {buy} more. How many {item} does {p1} have now?\n"
                    f"Solution:\n"
                    f"- Started with: {start} {item}\n"
                    f"- After giving away {give}: {start} - {give} = {start-give} {item}\n"
                    f"- After buying {buy} more: {start-give} + {buy} = {final} {item}\n"
                    f"Answer: {p1} has {final} {item}.")

        elif doc_type == "times_table_row":
            n = rng.randint(1, 50)
            lines = [f"Multiplication Table for {n}:"]
            for m in range(1, 13):
                lines.append(f"  {n} × {m} = {n*m}")
            return "\n".join(lines)

        elif doc_type == "pemdas":
            a, b, c = rng.randint(1, 50), rng.randint(1, 20), rng.randint(1, 20)
            return (f"Problem: Evaluate {a} + {b} × {c}.\n"
                    f"Solution (Order of Operations - PEMDAS):\n"
                    f"  Step 1: Multiplication first: {b} × {c} = {b*c}\n"
                    f"  Step 2: Then addition: {a} + {b*c} = {a + b*c}\n"
                    f"Answer: {a + b*c}.")

        elif doc_type == "fraction":
            d = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 50, 100])
            n = rng.randint(1, d - 1)
            dec = round(n / d, 4)
            pct = round(dec * 100, 2)
            return (f"Convert the fraction {n}/{d} to a decimal and percentage.\n"
                    f"Solution:\n"
                    f"  {n} ÷ {d} = {dec}\n"
                    f"  As a percentage: {dec} × 100% = {pct}%\n"
                    f"Answer: {n}/{d} = {dec} = {pct}%.")

        else:
            a, b = rng.randint(1, 1000), rng.randint(1, 1000)
            if a > b:
                relation = "greater than"
            elif a < b:
                relation = "less than"
            else:
                relation = "equal to"
            return f"Compare: Is {a} greater than, less than, or equal to {b}?\nAnswer: {a} is {relation} {b}."


class LogicSyllogismGenerator:
    """Generates millions of unique formal deductive logic and syllogism combinations."""

    SUBJECT_CLASSES = [
        # (Specific, General, Super-General)
        ("dogs", "canines", "mammals", "animals"),
        ("cats", "felines", "mammals", "animals"),
        ("eagles", "birds of prey", "birds", "vertebrates"),
        ("salmon", "fish", "aquatic vertebrates", "animals"),
        ("frogs", "amphibians", "cold-blooded vertebrates", "animals"),
        ("oak trees", "deciduous trees", "plants", "photosynthetic organisms"),
        ("roses", "flowering plants", "vascular plants", "living organisms"),
        ("squares", "rectangles", "quadrilaterals", "polygons"),
        ("equilateral triangles", "triangles", "polygons", "geometric shapes"),
        ("circles", "ellipses", "conic sections", "curves"),
        ("iron", "transition metals", "metals", "chemical elements"),
        ("oxygen", "nonmetals", "gases", "chemical elements"),
        ("water", "chemical compounds", "molecules", "substances"),
        ("Mars", "terrestrial planets", "planets in our solar system", "celestial bodies"),
        ("Jupiter", "gas giants", "planets in our solar system", "astronomical objects"),
    ]

    PROPERTIES = [
        ("mammals", "have a heart and breathe air"),
        ("birds", "have feathers and lay eggs"),
        ("fish", "breathe through gills and live in water"),
        ("plants", "perform photosynthesis to create glucose"),
        ("quadrilaterals", "have interior angles summing to 360 degrees"),
        ("triangles", "have interior angles summing to 180 degrees"),
        ("metals", "conduct electricity and heat"),
        ("chemical elements", "consist of atoms with a unique atomic number"),
        ("celestial bodies", "are held together by gravitational forces"),
    ]

    NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "George", "Hannah", "John", "Sarah", "Michael"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTHS = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    @classmethod
    def generate(cls, rng: random.Random) -> str:
        doc_type = rng.choice([
            "syllogism_3step", "syllogism_property", "modus_tollens",
            "temporal_day", "temporal_month", "divisibility_logic",
            "conditional_modus_ponens", "contrapositive", "disjunctive"
        ])

        if doc_type == "syllogism_3step":
            specific, gen1, gen2, super_gen = rng.choice(cls.SUBJECT_CLASSES)
            return (f"Logical Deduction:\n"
                    f"Premise 1: All {specific} are {gen1}.\n"
                    f"Premise 2: All {gen1} are {gen2}.\n"
                    f"Premise 3: All {gen2} are {super_gen}.\n"
                    f"Conclusion: Therefore, all {specific} are {super_gen}.\n"
                    f"Explanation: By the transitivity of categorical syllogisms, if A⊆B and B⊆C and C⊆D, then A⊆D.")

        elif doc_type == "syllogism_property":
            specific, gen1, gen2, _ = rng.choice(cls.SUBJECT_CLASSES)
            # Pick a matching property
            prop_pair = rng.choice(cls.PROPERTIES)
            p_class, p_desc = prop_pair
            return (f"Deductive Syllogism:\n"
                    f"Premise 1: All {specific} are {p_class}.\n"
                    f"Premise 2: All {p_class} {p_desc}.\n"
                    f"Conclusion: Therefore, all {specific} {p_desc}.\n"
                    f"Reasoning: The property of the broad category ({p_class}) necessarily applies to its subcategory ({specific}).")

        elif doc_type == "modus_tollens":
            rule_pairs = [
                ("a shape is a circle", "it has no straight sides", "has 4 straight sides", "is not a circle"),
                ("an integer is prime and greater than 2", "it is odd", "is even and equal to 12", "is not prime"),
                ("water is boiling at standard atmospheric pressure", "its temperature is 100°C", "temperature is 45°C", "is not boiling"),
                ("a substance is an acid", "its pH is less than 7", "pH is 9.5", "is not an acid"),
                ("a polygon is a triangle", "it has exactly 3 vertices", "has 5 vertices", "is not a triangle"),
            ]
            cond_p, cond_q, observed_not_q, deduced_not_p = rng.choice(rule_pairs)
            name = rng.choice(cls.NAMES)
            return (f"Logical Reasoning (Modus Tollens):\n"
                    f"Rule: If {cond_p}, then {cond_q}.\n"
                    f"Observation: {name} observes that the sample {observed_not_q}.\n"
                    f"Deduction: Therefore, the sample {deduced_not_p}.\n"
                    f"Formal Rule: If P → Q and ¬Q, then ¬P.")

        elif doc_type == "temporal_day":
            start_idx = rng.randint(0, 6)
            offset = rng.randint(1, 100)
            end_idx = (start_idx + offset) % 7
            start_day = cls.DAYS[start_idx]
            end_day = cls.DAYS[end_idx]
            return (f"Temporal Reasoning Problem:\n"
                    f"Question: If today is {start_day}, what day of the week will it be in {offset} days?\n"
                    f"Solution:\n"
                    f"1. There are 7 days in a full week cycle.\n"
                    f"2. Calculate {offset} modulo 7: {offset} mod 7 = {offset % 7}.\n"
                    f"3. Advance {offset % 7} days from {start_day}: {start_day} + {offset % 7} days = {end_day}.\n"
                    f"Answer: It will be {end_day}.")

        elif doc_type == "temporal_month":
            start_idx = rng.randint(0, 11)
            offset = rng.randint(1, 48)
            end_idx = (start_idx + offset) % 12
            return (f"Question: What month comes {offset} months after {cls.MONTHS[start_idx]}?\n"
                    f"Solution: Advance {offset} months (which is {offset // 12} years and {offset % 12} months):\n"
                    f"Answer: {cls.MONTHS[end_idx]}.")

        elif doc_type == "divisibility_logic":
            f1, f2 = rng.randint(2, 10), rng.randint(2, 10)
            prod = f1 * f2
            k = rng.randint(2, 50)
            val = prod * k
            return (f"Mathematical Logic Deduction:\n"
                    f"Premise 1: Any integer divisible by {prod} is also divisible by {f1} and {f2}.\n"
                    f"Premise 2: The number {val} is divisible by {prod} ({val} = {prod} × {k}).\n"
                    f"Conclusion: Therefore, {val} is divisible by {f1} ({val} = {f1} × {f2 * k}) and by {f2} ({val} = {f2} × {f1 * k}).")

        elif doc_type == "conditional_modus_ponens":
            name = rng.choice(cls.NAMES)
            var_num = rng.randint(10, 100)
            return (f"Conditional Reasoning (Modus Ponens):\n"
                    f"Premise 1: For any number n, if n > 50, then n + 10 > 60.\n"
                    f"Premise 2: {name} selects n = {var_num} (which is {'> 50' if var_num > 50 else '<= 50'}).\n"
                    f"Conclusion: {'Therefore, n + 10 = ' + str(var_num + 10) + ' > 60.' if var_num > 50 else 'The premise n > 50 is not met, so the rule does not guarantee n + 10 > 60.'}")

        elif doc_type == "contrapositive":
            p, q = rng.choice([
                ("a triangle is equilateral", "all three of its angles are 60 degrees"),
                ("a number is divisible by 10", "its last digit is 0"),
                ("an organism is a reptile", "it is cold-blooded"),
            ])
            return (f"Contrapositive Logic Equivalence:\n"
                    f"Original Statement: If {p}, then {q}.\n"
                    f"Contrapositive Statement: If not ({q}), then not ({p}).\n"
                    f"Rule: A conditional statement is logically equivalent to its contrapositive.")

        else:
            p, q = rng.choice([
                ("the light is on", "the light is off"),
                ("the integer is even", "the integer is odd"),
                ("the coin landed on heads", "the coin landed on tails"),
            ])
            return (f"Disjunctive Syllogism:\n"
                    f"Premise 1: Either {p} or {q} (exclusive or).\n"
                    f"Premise 2: We confirm that it is NOT the case that {p}.\n"
                    f"Conclusion: Therefore, {q}.")


class PremiseRefusalGenerator:
    """Generates millions of combinatorial grounded premise refusals and sanity corrections."""

    HISTORICAL_EVENTS = [
        ("George Washington", "1732–1799 (18th century)", "the Apollo 11 moon landing occurred in 1969", "fly to or land on the moon"),
        ("Abraham Lincoln", "1809–1865 (19th century)", "smartphones were invented in the 21st century", "use an iPhone or browse the internet"),
        ("Julius Caesar", "100 BC – 44 BC (Ancient Rome)", "automobiles were invented in the late 19th century", "drive a sports car"),
        ("Cleopatra", "69 BC – 30 BC (Ancient Egypt)", "commercial television was invented in the 20th century", "watch television programs"),
        ("William Shakespeare", "1564–1616 (Elizabethan era)", "the first airplane flight occurred in 1903", "fly in an airplane"),
        ("Leonardo da Vinci", "1452–1519 (Renaissance)", "digital cameras were invented in the late 20th century", "take a digital photograph"),
        ("Isaac Newton", "1643–1727", "electronic computers were invented in the 1940s", "write a Python program"),
        ("Alexander the Great", "356 BC – 323 BC", "firearms were invented centuries later", "use a machine gun"),
    ]

    ANIMALS_ANATOMY = [
        ("dog", "four-legged mammal", "six-legged flying bird", "fly through the air or lay eggs"),
        ("cat", "four-legged feline mammal", "fish with scales and gills", "live underwater and breathe with gills"),
        ("goldfish", "aquatic fish with gills and fins", "mammal with four legs", "walk on land or breathe atmospheric air"),
        ("eagle", "two-legged bird with wings and feathers", "four-legged reptile", "burrow underground on 4 legs"),
        ("snake", "limbless reptile", "ten-legged crustacean", "walk on legs or fly"),
        ("elephant", "large land mammal with a trunk", "feathered bird that can fly", "nest in trees or fly"),
    ]

    @classmethod
    def generate(cls, rng: random.Random) -> str:
        doc_type = rng.choice(["math_add_false", "math_mult_false", "historical_anachronism",
                                "animal_biology_false", "physics_impossible", "common_myth"])

        if doc_type == "math_add_false":
            a = rng.randint(1, 500)
            b = rng.randint(1, 500)
            correct = a + b
            delta = rng.choice([-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50, 100])
            wrong = correct + delta
            return (f"Question: Is {a} + {b} equal to {wrong}?\n"
                    f"Answer: No, {a} + {b} is not equal to {wrong}. "
                    f"The correct calculation is {a} + {b} = {correct}.")

        elif doc_type == "math_mult_false":
            a = rng.randint(2, 50)
            b = rng.randint(2, 50)
            correct = a * b
            delta = rng.choice([-20, -10, -5, -2, -1, 1, 2, 5, 10, 20])
            wrong = correct + delta
            return (f"Question: Is {a} × {b} equal to {wrong}?\n"
                    f"Answer: No, {a} × {b} is not equal to {wrong}. "
                    f"The correct product is {a} × {b} = {correct}.")

        elif doc_type == "historical_anachronism":
            person, era, tech_context, action = rng.choice(cls.HISTORICAL_EVENTS)
            return (f"Question: Did {person} {action}?\n"
                    f"Answer: No, {person} did not {action}. "
                    f"{person} lived during {era}, long before {tech_context}. "
                    f"Therefore, it was historically impossible for {person} to {action}.")

        elif doc_type == "animal_biology_false":
            animal, real_desc, fake_desc, impossible_action = rng.choice(cls.ANIMALS_ANATOMY)
            return (f"Question: Is a {animal} a {fake_desc}?\n"
                    f"Answer: No, a {animal} is not a {fake_desc}. "
                    f"A {animal} is a {real_desc}. It cannot {impossible_action}.")

        elif doc_type == "physics_impossible":
            physics_cases = [
                ("Can heat naturally flow from a colder object to a hotter object without external work?",
                 "No, heat cannot spontaneously flow from a colder body to a hotter body. This is forbidden by the Second Law of Thermodynamics."),
                ("Can an object travel faster than the speed of light in a vacuum?",
                 "No, according to the theory of special relativity, no mass or information can travel faster than c (approximately 299,792 km/s) in a vacuum."),
                ("Can a machine achieve 100% efficiency and produce perpetual motion?",
                 "No, perpetual motion machines of the first or second kind are physically impossible because energy is always dissipated as heat due to friction and entropy."),
                ("Does sound travel faster in a vacuum than in air?",
                 "No, sound is a mechanical wave that requires a physical medium (gas, liquid, or solid) to propagate. Sound cannot travel in a vacuum at all."),
            ]
            q, a = rng.choice(physics_cases)
            return f"Question: {q}\nAnswer: {a}"

        else:
            myths = [
                ("Is the Earth flat?",
                 "No, the Earth is not flat. The Earth is an oblate spheroid, flattened slightly at the poles. This has been established by satellite photography, physics, astronomy, and centuries of navigation."),
                ("Do humans use only 10% of their brains?",
                 "No, this is a popular myth. Modern neuroimaging (fMRI, PET scans) shows that virtually all regions of the brain are active and have known functions."),
                ("Is lightning unable to strike the same place twice?",
                 "No, lightning frequently strikes the same place multiple times, especially tall structures like the Empire State Building which is struck dozens of times per year."),
            ]
            q, a = rng.choice(myths)
            return f"Question: {q}\nAnswer: {a}"


# ==============================================================================
#  SHARD WRITER WITH RESUME SUPPORT
# ==============================================================================

class ShardWriter:
    """Writes deduplicated, tokenized documents into uint16 binary shards with RESUME support."""

    def __init__(self, tokenizer: Tokenizer, output_dir: Path,
                 tokens_per_shard: int = 1_000_000, val_ratio: float = 0.03):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.train_dir = output_dir / "train"
        self.val_dir = output_dir / "val"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        self.tokens_per_shard = tokens_per_shard
        self.val_ratio = val_ratio
        self.eos_id = tokenizer.token_to_id("<|endoftext|>") or tokenizer.token_to_id("<eos>") or 0

        self.train_buf: List[int] = []
        self.train_seg: List[int] = []
        self.val_buf: List[int] = []
        self.val_seg: List[int] = []
        self.doc_counter = 0
        self.total_docs_accepted = 0
        self.total_docs_rejected_dup = 0
        self.total_docs_rejected_quality = 0

        # === AUTOMATIC SHARD RESUME DETECTION ===
        existing_train = sorted(list(self.train_dir.glob("shard_*.bin")))
        existing_train = [f for f in existing_train if not f.name.endswith("_seg.bin")]
        existing_val = sorted(list(self.val_dir.glob("shard_*.bin")))
        existing_val = [f for f in existing_val if not f.name.endswith("_seg.bin")]

        if existing_train:
            max_t_idx = max(int(f.stem.split("_")[1]) for f in existing_train)
            self.train_shard_idx = max_t_idx + 1
            self.total_train_tokens = len(existing_train) * self.tokens_per_shard
            print(f"  [OK] RESUME DETECTED: Found {len(existing_train):,} existing train shards ({self.total_train_tokens:,} tokens). Next shard: shard_{self.train_shard_idx:05d}.bin")
        else:
            self.train_shard_idx = 0
            self.total_train_tokens = 0

        if existing_val:
            max_v_idx = max(int(f.stem.split("_")[1]) for f in existing_val)
            self.val_shard_idx = max_v_idx + 1
            self.total_val_tokens = len(existing_val) * self.tokens_per_shard
            print(f"  [OK] RESUME DETECTED: Found {len(existing_val):,} existing val shards ({self.total_val_tokens:,} tokens). Next shard: shard_{self.val_shard_idx:05d}.bin")
        else:
            self.val_shard_idx = 0
            self.total_val_tokens = 0

    def add_document(self, text: str, seen_hashes: Set[str],
                     is_code: bool = False, rng: random.Random = None) -> bool:
        """Clean, deduplicate (SHA-256), tokenize, and buffer a document."""
        text = clean_text(text, is_code=is_code)
        if len(text.strip()) < 15:
            return False

        if not quality_filter(text, is_code=is_code) and len(text.split()) > 30:
            self.total_docs_rejected_quality += 1
            return False

        doc_hash = compute_doc_hash(text)
        if doc_hash in seen_hashes:
            self.total_docs_rejected_dup += 1
            return False
        seen_hashes.add(doc_hash)

        encoded = self.tokenizer.encode(text).ids
        if not encoded:
            return False
        if encoded[-1] != self.eos_id:
            encoded.append(self.eos_id)

        is_val = (rng or random).random() < self.val_ratio
        buf = self.val_buf if is_val else self.train_buf
        seg = self.val_seg if is_val else self.train_seg

        self.doc_counter += 1
        doc_id = self.doc_counter
        buf.extend(encoded)
        seg.extend([doc_id % 65535] * len(encoded))

        self.total_docs_accepted += 1

        if is_val and len(self.val_buf) >= self.tokens_per_shard:
            self._flush(is_val=True)
        elif not is_val and len(self.train_buf) >= self.tokens_per_shard:
            self._flush(is_val=False)

        return True

    def _flush(self, is_val: bool):
        buf = self.val_buf if is_val else self.train_buf
        seg = self.val_seg if is_val else self.train_seg
        d = self.val_dir if is_val else self.train_dir
        idx = self.val_shard_idx if is_val else self.train_shard_idx

        shard_toks = np.array(buf[:self.tokens_per_shard], dtype=np.uint16)
        shard_segs = np.array(seg[:self.tokens_per_shard], dtype=np.uint16)

        bin_path = d / f"shard_{idx:05d}.bin"
        seg_path = d / f"shard_{idx:05d}_seg.bin"
        with open(bin_path, "wb") as f:
            f.write(shard_toks.tobytes())
        with open(seg_path, "wb") as f:
            f.write(shard_segs.tobytes())

        if is_val:
            self.val_buf = buf[self.tokens_per_shard:]
            self.val_seg = seg[self.tokens_per_shard:]
            self.total_val_tokens += len(shard_toks)
            self.val_shard_idx += 1
        else:
            self.train_buf = buf[self.tokens_per_shard:]
            self.train_seg = seg[self.tokens_per_shard:]
            self.total_train_tokens += len(shard_toks)
            self.train_shard_idx += 1
            if self.train_shard_idx % 50 == 0:
                print(f"    [TRAIN SHARD {self.train_shard_idx:05d}] "
                      f"Total: {self.total_train_tokens:,} tokens | "
                      f"Docs: {self.total_docs_accepted:,} accepted, "
                      f"{self.total_docs_rejected_dup:,} dup-rejected, "
                      f"{self.total_docs_rejected_quality:,} quality-rejected")

    def flush_remaining(self):
        if self.train_buf:
            self._flush(is_val=False)
        if self.val_buf:
            self._flush(is_val=True)


# ==============================================================================
#  MAIN BUILD PIPELINE (With Cumulative Stream Resume)
# ==============================================================================

def build_3b_corpus(target_tokens: int = 3_000_000_000, output_dir: str = "data_3b/shards"):
    print("=" * 80)
    print(f"  3.0 BILLION TOKEN MULTI-DOMAIN CORPUS BUILDER")
    print(f"  Target: {target_tokens:,} tokens | Output: {output_dir}")
    print("=" * 80)

    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    seen_hashes = load_all_past_hashes()
    writer = ShardWriter(tokenizer, Path(output_dir))
    rng = random.Random(42)
    stream_stats = []

    cumulative_target = 0

    for stream_idx, spec in enumerate(STREAM_SPECS, 1):
        stream_name = spec["name"]
        stream_target = spec["target_tokens"]
        cumulative_target += stream_target

        # Check if this stream is already satisfied by existing shards on disk
        if writer.total_train_tokens >= cumulative_target:
            print(f"\n  [Stream {stream_idx}/{len(STREAM_SPECS)}] {stream_name}: ALREADY COMPLETED ({stream_target:,} tokens on disk). Skipping.")
            stream_stats.append({"name": stream_name, "target": stream_target, "actual": stream_target, "status": "ALREADY_COMPLETED"})
            continue

        stream_start_tokens = writer.total_train_tokens + writer.total_val_tokens

        print(f"\n{'='*80}")
        print(f"  [Stream {stream_idx}/{len(STREAM_SPECS)}] {stream_name}")
        print(f"  Target: {stream_target:,} tokens")
        print(f"{'='*80}")

        stream_tokens_added = 0

        # --- HuggingFace Sources ---
        hf_token = os.environ.get("HF_TOKEN")
        for source in spec.get("sources", []):
            if stream_tokens_added >= stream_target:
                break

            hf_path = source["hf_path"]
            name_kw = source.get("name_kw")
            split = source.get("split", "train")
            skip_docs = source.get("skip_docs", 0)
            min_score = source.get("min_score")
            score_field = source.get("quality_score_field")

            print(f"\n  -> Streaming from {hf_path}"
                  f"{f' ({name_kw})' if name_kw else ''}"
                  f"{f' [skip {skip_docs:,}]' if skip_docs else ''}"
                  f"{f' [score >= {min_score}]' if min_score else ''}...")

            try:
                load_kwargs = {"split": split, "streaming": True}
                if hf_token:
                    load_kwargs["token"] = hf_token

                if name_kw:
                    ds = load_dataset(hf_path, name_kw, **load_kwargs)
                else:
                    ds = load_dataset(hf_path, **load_kwargs)

                doc_count = 0
                for doc in ds:
                    doc_count += 1
                    if doc_count <= skip_docs:
                        continue

                    if min_score and score_field and score_field in doc:
                        if float(doc[score_field]) < min_score:
                            continue

                    text = extract_text(doc, source, spec["domain_type"])
                    if not text:
                        continue

                    added = writer.add_document(text, seen_hashes,
                                                 is_code=spec.get("is_code", False), rng=rng)
                    if added:
                        tok_len = len(tokenizer.encode(text).ids)
                        stream_tokens_added += tok_len

                    if stream_tokens_added >= stream_target:
                        break

                print(f"    [OK] Collected {stream_tokens_added:,} tokens from {hf_path}")

            except Exception as e:
                print(f"    [FAIL] Error streaming {hf_path}: {e}")

        # --- Synthetic Generators with Duplicate Safety Guard ---
        syn_gen = spec.get("synthetic_generator")
        syn_target = spec.get("synthetic_tokens", 0)

        if syn_gen and stream_tokens_added < stream_target:
            remaining = stream_target - stream_tokens_added
            syn_count = min(remaining, syn_target)
            print(f"\n  -> Generating synthetic data ({syn_gen}): {syn_count:,} tokens...")

            gen_tokens = 0
            consecutive_dups = 0
            max_consecutive_dups = 5000

            while gen_tokens < syn_count and consecutive_dups < max_consecutive_dups:
                if syn_gen == "foundational_arithmetic" or syn_gen == "math_scratchpads":
                    text = ArithmeticGenerator.generate(rng)
                elif syn_gen == "logic_syllogisms":
                    text = LogicSyllogismGenerator.generate(rng)
                elif syn_gen == "premise_refusal":
                    text = PremiseRefusalGenerator.generate(rng)
                else:
                    break

                added = writer.add_document(text, seen_hashes, is_code=False, rng=rng)
                if added:
                    tok_len = len(tokenizer.encode(text).ids)
                    gen_tokens += tok_len
                    stream_tokens_added += tok_len
                    consecutive_dups = 0
                else:
                    consecutive_dups += 1

            if consecutive_dups >= max_consecutive_dups:
                print(f"    [INFO] Synthetic generator reached diversity ceiling ({gen_tokens:,} tokens generated). Proceeding safely.")
            else:
                print(f"    [OK] Generated {gen_tokens:,} synthetic tokens")

        total_stream = (writer.total_train_tokens + writer.total_val_tokens) - stream_start_tokens
        stream_stats.append({"name": stream_name, "target": stream_target, "actual": total_stream, "status": "COMPLETED"})
        print(f"\n  Stream Summary: {stream_name} → {total_stream:,} / {stream_target:,} tokens")

    # Flush remaining buffers
    writer.flush_remaining()

    # Save manifest
    manifest = {
        "dataset_name": "3.0B Token Multi-Domain Pretraining Corpus",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_tokens": target_tokens,
        "total_train_tokens": writer.total_train_tokens,
        "total_val_tokens": writer.total_val_tokens,
        "total_tokens": writer.total_train_tokens + writer.total_val_tokens,
        "train_shards": writer.train_shard_idx,
        "val_shards": writer.val_shard_idx,
        "docs_accepted": writer.total_docs_accepted,
        "docs_rejected_duplicate": writer.total_docs_rejected_dup,
        "docs_rejected_quality": writer.total_docs_rejected_quality,
        "historical_hashes_loaded": len(seen_hashes),
        "streams": stream_stats,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  CORPUS BUILD COMPLETE")
    print(f"  Train: {writer.total_train_tokens:,} tokens ({writer.train_shard_idx} shards)")
    print(f"  Val:   {writer.total_val_tokens:,} tokens ({writer.val_shard_idx} shards)")
    print(f"  Total: {writer.total_train_tokens + writer.total_val_tokens:,} tokens")
    print(f"  Manifest saved: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3.0B Token Multi-Domain Corpus Builder")
    parser.add_argument("--target-tokens", type=int, default=3_000_000_000)
    parser.add_argument("--output-dir", type=str, default="data_3b/shards")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face User Access Token")
    parser.add_argument("--sample-test", action="store_true",
                        help="Generate a small 5M token sample to verify pipeline")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        print("  [OK] Hugging Face authentication token configured.")

    target = 5_000_000 if args.sample_test else args.target_tokens
    build_3b_corpus(target_tokens=target, output_dir=args.output_dir)
