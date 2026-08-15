from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

full_sentence = "Once upon a time, in a bright green forest, there lived a friendly little squirrel named Sammy."
prompt = "Once upon a time, in a bright green"

full_ids = tokenizer.encode(full_sentence).ids
prompt_ids = tokenizer.encode(prompt).ids

print("Full sentence IDs:")
for i in full_ids:
    print(f"  {i}: {repr(tokenizer.id_to_token(i))}")

print("\nPrompt IDs:")
for i in prompt_ids:
    print(f"  {i}: {repr(tokenizer.id_to_token(i))}")

print(f"\nDo prompt IDs match prefix of full sentence IDs? {full_ids[:len(prompt_ids)] == prompt_ids}")
