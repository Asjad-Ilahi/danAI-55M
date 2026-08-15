import sys
from datasets import load_dataset
from tokenizers import Tokenizer

print("Testing Hugging Face Dataset Sources for 30M Dataset...")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# 1. Math Dataset Test
print("\n[1/4] Testing Math Dataset (cosmopedia-v2 / stanford)...")
try:
    ds_math = load_dataset("HuggingFaceTB/cosmopedia-v2", data_files="stanford/*.parquet", split="train", streaming=True)
    sample_math = next(iter(ds_math))
    text_math = sample_math.get("text", "")
    print(f"  Math Sample ({len(tokenizer.encode(text_math).ids)} tokens): {text_math[:120]}...")
except Exception as e:
    print(f"  Math Error: {e}")

# 2. Python Code Test
print("\n[2/4] Testing Python Code Dataset (iamtarun/python_code_instructions_18k_alpaca or CodeSearchNet)...")
try:
    ds_py = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    sample_py = next(iter(ds_py))
    text_py = sample_py.get("instruction", "") + "\n" + sample_py.get("output", "")
    print(f"  Python Sample ({len(tokenizer.encode(text_py).ids)} tokens): {text_py[:120]}...")
except Exception as e:
    print(f"  Python Error: {e}")

# 3. HTML Code Test
print("\n[3/4] Testing HTML / Web Code Dataset (bigcode/the-stack-smol)...")
try:
    ds_html = load_dataset("bigcode/the-stack-smol", data_dir="data/html", split="train", streaming=True)
    sample_html = next(iter(ds_html))
    text_html = sample_html.get("content", "")
    print(f"  HTML Sample ({len(tokenizer.encode(text_html).ids)} tokens): {text_html[:120]}...")
except Exception as e:
    print(f"  HTML Error: {e}")

# 4. General Knowledge Test
print("\n[4/4] Testing General Knowledge Dataset (FineWeb-Edu)...")
try:
    ds_gen = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    sample_gen = next(iter(ds_gen))
    text_gen = sample_gen.get("text", "")
    print(f"  General Sample ({len(tokenizer.encode(text_gen).ids)} tokens): {text_gen[:120]}...")
except Exception as e:
    print(f"  General Error: {e}")

print("\nDataset test script completed.")
