from datasets import load_dataset

print("1. Testing TinyStories (Stories)...")
ds_stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
s_sample = next(iter(ds_stories))["text"]
print("  Sample:", repr(s_sample[:120]))

print("\n2. Testing FineWeb-Edu / SmolLM (General Knowledge & Science)...")
ds_edu = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
e_sample = next(iter(ds_edu))["text"]
print("  Sample:", repr(e_sample[:120]))

print("\n3. Testing Python Code...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
c_item = next(iter(ds_code))
c_sample = f"# {c_item['instruction']}\n{c_item['output']}"
print("  Sample:", repr(c_sample[:120]))

print("\n4. Testing GSM8K Math...")
ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)
m_item = next(iter(ds_math))
m_sample = f"Question: {m_item['question']}\nAnswer: {m_item['answer']}"
print("  Sample:", repr(m_sample[:120]))

print("\nALL 4 DATASET SOURCES VERIFIED SUCCESSFULLY!")
