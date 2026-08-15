import re
from datasets import load_dataset
from src.data.cleaner import clean_code

def format_python_code(instruction: str, output: str) -> str:
    # 1. Extract pure code block if wrapped in ```python ... ```
    match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if match:
        code_body = match.group(1).strip()
    else:
        # Strip backtick lines if any
        lines = [l for l in output.split("\n") if not l.strip().startswith("```")]
        code_body = "\n".join(lines).strip()

    # 2. Clean instruction for docstring
    inst = instruction.strip()
    if inst and not inst.endswith('.'):
        inst += '.'

    # 3. Format as clean python source file with module docstring
    if code_body.startswith('"""') or code_body.startswith("'''"):
        formatted = code_body
    else:
        formatted = f'"""\n{inst}\n"""\n{code_body}'
        
    return clean_code(formatted)

ds = load_dataset('flytech/python-codes-25k', split='train', streaming=True)
for i, item in enumerate(ds):
    if i < 5:
        inst = item.get('instruction', '')
        out = item.get('output', '')
        res = format_python_code(inst, out)
        print(f"=== FORMATTED ITEM {i} ===")
        print(res)
        print("="*60)
    else:
        break
