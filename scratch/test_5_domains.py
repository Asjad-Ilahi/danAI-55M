"""
Test all 5 domain dataset sources and verify cleaning output.
"""
from datasets import load_dataset
from src.data.cleaner import clean_text

print("==========================================================================")
print("1. GENERAL KNOWLEDGE (FineWeb-Edu)")
ds_gen = iter(load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True))
item_gen = next(ds_gen)["text"]
clean_gen = clean_text(item_gen, is_code=False)
print("SAMPLE PROSE:\n", clean_gen[:200])

print("\n==========================================================================")
print("2. STORIES (TinyStories)")
ds_story = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
item_story = next(ds_story)["text"]
clean_story = clean_text(item_story, is_code=False)
print("SAMPLE STORY:\n", clean_story[:200])

print("\n==========================================================================")
print("3. MATH & REASONING (MetaMathQA / GSM8K / Cosmopedia Q&A)")
try:
    ds_math = iter(load_dataset("meta-math/MetaMathQA", split="train", streaming=True))
    m_item = next(ds_math)
    item_math = f"Question: {m_item['query']}\nAnswer: {m_item['response']}"
except Exception as e:
    print("MetaMathQA failed, falling back to GSM8K:", e)
    ds_math = iter(load_dataset("gsm8k", "main", split="train", streaming=True))
    m_item = next(ds_math)
    item_math = f"Question: {m_item['question']}\nAnswer: {m_item['answer']}"

clean_math = clean_text(item_math, is_code=False)
print("SAMPLE MATH:\n", clean_math[:200])

print("\n==========================================================================")
print("4. PYTHON CODE (Python-Edu)")
ds_py = iter(load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True))
item_py = next(ds_py)["text"]
clean_py = clean_text(item_py, is_code=True)
print("SAMPLE PYTHON (indentation preserved!):\n", clean_py[:200])

print("\n==========================================================================")
print("5. HTML / WEB CODE (CodeSearchNet HTML / Web / Python Instructions)")
try:
    ds_html = iter(load_dataset("bigcode/the-stack-smol-xs", data_dir="data/html", split="train", streaming=True))
    item_html = next(ds_html)["content"]
except Exception as e:
    print("the-stack-smol-xs html fallback:", e)
    ds_html = iter(load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True))
    h_item = next(ds_html)
    item_html = f"Instruction: {h_item['instruction']}\nCode:\n{h_item['output']}"

clean_html = clean_text(item_html, is_code=True)
print("SAMPLE HTML/WEB CODE:\n", clean_html[:200])
