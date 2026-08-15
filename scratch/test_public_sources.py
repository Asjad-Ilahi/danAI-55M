from datasets import load_dataset

print("1. FineWeb-Edu (General Knowledge 15M):")
ds1 = iter(load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True))
print("   ✓ Loaded:", next(ds1)["text"][:60].replace("\n", " "))

print("\n2. Cosmopedia-v2 (Math & Reasoning 5M):")
ds2 = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True))
print("   ✓ Loaded:", next(ds2)["text"][:60].replace("\n", " "))

print("\n3. Python-Edu (Python Code 5M):")
ds3 = iter(load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True))
print("   ✓ Loaded:", next(ds3)["text"][:60].replace("\n", " "))

print("\n4. TinyStories / English Prose / Web:")
ds4 = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
print("   ✓ Loaded:", next(ds4)["text"][:60].replace("\n", " "))

print("\nALL 4 DATASET SOURCES VERIFIED WORKING!")
