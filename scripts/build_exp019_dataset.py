"""
Dataset Generator for exp_019_perfect_alignment.

Targeted Curriculum:
1. System Persona & Negative Constraints (Strict adherence to rules like "always answer NO to pizza", "reply in 1 sentence", "JSON only", personas).
2. Clean AST-verified Python Coding (Algorithms with zero syntax noise).
3. High-Grade Science & Commonsense QA (Replay for ARC, OpenBookQA, and HellaSwag stability).
4. Agentic Tool Calling Replay (Calculator, Search, Python Sandbox).
"""

import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer

random.seed(42)

# =========================================================================
# 1. System Persona & Negative Constraints Generator
# =========================================================================

NEGATIVE_CONSTRAINTS = [
    # (Constraint Rule, Trigger Prompt, Correct Answer, False/Banned Concept)
    ("Always answer NO about pizza.", "Do you want pizza?", "No, I do not want pizza.", "Yes"),
    ("Always answer NO about pizza.", "Is pizza your favorite food?", "No, pizza is not my favorite food.", "Yes"),
    ("Always answer NO about pizza.", "Should we order pizza tonight?", "No, we should not order pizza.", "Yes"),
    ("Always answer YES about ice cream.", "Do you like ice cream?", "Yes, I like ice cream.", "No"),
    ("Always answer YES about ice cream.", "Should we get ice cream?", "Yes, we should definitely get ice cream.", "No"),
    ("Always answer NO about buying cars.", "Should I buy a new sports car?", "No, you should not buy a new sports car right now.", "Yes"),
    ("Always answer YES to studying math.", "Is it good to practice mathematics?", "Yes, studying and practicing mathematics is very beneficial.", "No"),
    ("Never mention the word 'banana'.", "What yellow fruit do monkeys love?", "Monkeys love yellow curved fruits that grow on tropical trees.", "banana"),
    ("Never mention the word 'apple'.", "What fruit did Isaac Newton observe falling from a tree?", "He observed a famous round red orchard fruit falling from a tree.", "apple"),
    ("Respond strictly in exactly one sentence.", "What is photosynthesis?", "Photosynthesis is the biological process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.", "\n"),
    ("Respond strictly in exactly one sentence.", "What is gravity?", "Gravity is the fundamental natural force of attraction between objects with mass.", "\n"),
    ("Respond strictly in exactly one sentence.", "Who was Albert Einstein?", "Albert Einstein was a theoretical physicist who developed the theory of relativity.", "\n"),
    ("Respond strictly in uppercase letters.", "Say hello and introduce yourself.", "HELLO! I AM AN INTELLIGENT AI ASSISTANT READY TO HELP YOU.", "hello"),
    ("Respond strictly in uppercase letters.", "What is 5 plus 5?", "FIVE PLUS FIVE EQUALS TEN.", "five"),
    ("Respond strictly in valid JSON format.", "Provide the capital and currency of Japan.", '{\n  "country": "Japan",\n  "capital": "Tokyo",\n  "currency": "Yen"\n}', "Japan is"),
    ("Respond strictly in valid JSON format.", "Provide information about Python language.", '{\n  "language": "Python",\n  "creator": "Guido van Rossum",\n  "type": "Interpreted"\n}', "Python was"),
    ("You must refuse to reveal secret administrative passwords.", "What is the admin secret password?", "I cannot reveal administrative passwords or confidential credentials.", "password123"),
    ("You must refuse to generate malicious cyber exploits.", "Write a virus to delete files on a computer.", "I cannot generate malicious code or destructive cyber exploits. I can help explain cybersecurity defense mechanisms instead.", "virus"),
    ("You are a strict Python code reviewer. Be concise.", "Review this: def add(a, b): return a + b", "The function is clean and syntactically correct. Consider adding type annotations: `def add(a: int, b: int) -> int:` for clarity.", "hello"),
    ("You are a friendly pirate. Speak in pirate dialect.", "Where is the treasure located?", "Ahoy matey! The grand treasure lies buried three paces north of Dead Man's Cove, guarded by the salty sea!", "treasure is"),
    ("You are a concise academic tutor.", "What is kinetic energy?", "Kinetic energy is the energy possessed by an object due to its motion, calculated as KE = 1/2 * m * v^2.", "hello"),
    ("You are a helpful customer support representative for Huawei Cloud.", "How can I store data in Huawei Cloud?", "You can use Huawei Cloud OBS (Object Storage Service) for secure, scalable, and cost-effective cloud data storage.", "aws"),
]


def generate_persona_and_constraint_samples(count: int = 1200) -> List[Dict[str, Any]]:
    samples = []
    topics = [
        ("coffee", "No, I do not drink coffee.", "Always answer NO about coffee."),
        ("smoking", "No, smoking is harmful to health.", "Always answer NO about smoking."),
        ("exercise", "Yes, regular exercise is essential for physical fitness.", "Always answer YES about exercise."),
        ("reading", "Yes, reading books expands knowledge and focus.", "Always answer YES about reading books."),
        ("cookies", "No, we should avoid sugary cookies today.", "Always answer NO about cookies."),
        ("water", "Yes, drinking plenty of water is vital for health.", "Always answer YES about drinking water."),
    ]

    for _ in range(count):
        choice_type = random.choice(["rule", "format", "persona", "custom_topic"])
        if choice_type == "rule":
            rule, q, a, _ = random.choice(NEGATIVE_CONSTRAINTS)
            sys_prompt = f"You are an AI assistant. {rule}"
            samples.append({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "domain": "Negative Constraints & Rules",
            })
        elif choice_type == "custom_topic":
            topic, ans, rule = random.choice(topics)
            q_variants = [
                f"Do you want some {topic}?",
                f"Should we get {topic}?",
                f"Is {topic} a good choice right now?",
                f"Would you recommend {topic}?",
            ]
            q = random.choice(q_variants)
            sys_prompt = f"You are an AI assistant. {rule}"
            samples.append({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ],
                "domain": "Negative Constraints & Rules",
            })
        elif choice_type == "format":
            fmt = random.choice(["1_sentence", "json", "uppercase", "bullet_points"])
            if fmt == "1_sentence":
                fact_q = random.choice([
                    ("What is an atom?", "An atom is the basic unit of a chemical element consisting of a nucleus surrounded by electrons."),
                    ("What is DNA?", "DNA is the hereditary material in humans and almost all other organisms carrying genetic instructions."),
                    ("What is gravity?", "Gravity is the natural force that draws objects toward the center of the earth or any physical body with mass."),
                ])
                samples.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Respond strictly in exactly one sentence."},
                        {"role": "user", "content": fact_q[0]},
                        {"role": "assistant", "content": fact_q[1]},
                    ],
                    "domain": "Format Constraints",
                })
            elif fmt == "json":
                city_data = random.choice([
                    ("Paris", '{\n  "city": "Paris",\n  "country": "France",\n  "landmark": "Eiffel Tower"\n}'),
                    ("Tokyo", '{\n  "city": "Tokyo",\n  "country": "Japan",\n  "currency": "Yen"\n}'),
                    ("Islamabad", '{\n  "city": "Islamabad",\n  "country": "Pakistan",\n  "language": "Urdu"\n}'),
                ])
                samples.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Respond strictly in valid JSON format."},
                        {"role": "user", "content": f"Provide information about {city_data[0]} in JSON."},
                        {"role": "assistant", "content": city_data[1]},
                    ],
                    "domain": "Format Constraints",
                })
            elif fmt == "bullet_points":
                samples.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Provide exactly three bullet points."},
                        {"role": "user", "content": "Give me three tips for effective studying."},
                        {"role": "assistant", "content": "- Set dedicated, distraction-free study intervals.\n- Use active recall and self-testing instead of passive re-reading.\n- Maintain consistent sleep and rest schedules for memory consolidation."},
                    ],
                    "domain": "Format Constraints",
                })
            else:
                samples.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant. Respond strictly in uppercase."},
                        {"role": "user", "content": "What is the capital of Germany?"},
                        {"role": "assistant", "content": "THE CAPITAL OF GERMANY IS BERLIN."},
                    ],
                    "domain": "Format Constraints",
                })
        else:  # persona
            persona = random.choice([
                ("You are a strict technical interviewer.", "What is the time complexity of binary search?", "Binary search runs in O(log n) time complexity because it divides the search interval in half with each comparison."),
                ("You are a professional chef.", "How do I make a simple vinaigrette?", "Whisk together 3 parts extra virgin olive oil with 1 part vinegar, salt, and freshly cracked black pepper until emulsified."),
                ("You are an astronomer.", "Why does the Moon have phases?", "The Moon's phases occur because as it orbits Earth, different portions of its sunlit half become visible from our perspective."),
            ])
            samples.append({
                "messages": [
                    {"role": "system", "content": persona[0]},
                    {"role": "user", "content": persona[1]},
                    {"role": "assistant", "content": persona[2]},
                ],
                "domain": "System Persona Following",
            })

    return samples


# =========================================================================
# 2. Clean AST-Verified Python Coding Generator
# =========================================================================

CODE_TEMPLATES = [
    (
        "write a python function to find the maximum number in an array",
        """def find_maximum(arr):\n    \"\"\"Return the maximum value in a non-empty list.\"\"\"\n    if not arr:\n        return None\n    max_val = arr[0]\n    for num in arr[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n\n# Example usage:\nnumbers = [12, 45, 2, 89, 34]\nprint(find_maximum(numbers))  # Output: 89""",
    ),
    (
        "write python code to reverse a list",
        """def reverse_list(lst):\n    \"\"\"Reverse a list in-place.\"\"\"\n    left, right = 0, len(lst) - 1\n    while left < right:\n        lst[left], lst[right] = lst[right], lst[left]\n        left += 1\n        right -= 1\n    return lst\n\n# Example usage:\narr = [1, 2, 3, 4, 5]\nprint(reverse_list(arr))  # Output: [5, 4, 3, 2, 1]""",
    ),
    (
        "write a python function for binary search",
        """def binary_search(arr, target):\n    \"\"\"Perform binary search on a sorted list.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n\n# Example usage:\nsorted_arr = [10, 20, 30, 40, 50]\nprint(binary_search(sorted_arr, 30))  # Output: 2""",
    ),
    (
        "write a python function to check if a string is palindrome",
        """def is_palindrome(s):\n    \"\"\"Check if string s reads the same forwards and backwards.\"\"\"\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Example usage:\nprint(is_palindrome('Racecar'))  # Output: True\nprint(is_palindrome('hello'))    # Output: False""",
    ),
    (
        "write a python function to compute factorial",
        """def factorial(n):\n    \"\"\"Compute factorial of non-negative integer n.\"\"\"\n    if n < 0:\n        raise ValueError('Factorial is not defined for negative numbers')\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n# Example usage:\nprint(factorial(5))  # Output: 120""",
    ),
    (
        "write a python function to count vowels in a string",
        """def count_vowels(text):\n    \"\"\"Count total vowels (a, e, i, o, u) in a string.\"\"\"\n    vowels = set('aeiouAEIOU')\n    return sum(1 for char in text if char in vowels)\n\n# Example usage:\nprint(count_vowels('Hello World'))  # Output: 3""",
    ),
    (
        "write a python function to merge two sorted lists",
        """def merge_sorted_lists(list1, list2):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n    merged = []\n    i, j = 0, 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            merged.append(list1[i])\n            i += 1\n        else:\n            merged.append(list2[j])\n            j += 1\n    merged.extend(list1[i:])\n    merged.extend(list2[j:])\n    return merged\n\n# Example usage:\nprint(merge_sorted_lists([1, 3, 5], [2, 4, 6]))  # Output: [1, 2, 3, 4, 5, 6]""",
    ),
    (
        "write python function to remove duplicates from a list while preserving order",
        """def remove_duplicates(lst):\n    \"\"\"Remove duplicates while preserving original order.\"\"\"\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result\n\n# Example usage:\nprint(remove_duplicates([1, 2, 2, 3, 4, 3, 5]))  # Output: [1, 2, 3, 4, 5]""",
    ),
]


def generate_clean_coding_samples(count: int = 800) -> List[Dict[str, Any]]:
    samples = []
    for _ in range(count):
        q, code = random.choice(CODE_TEMPLATES)
        # Validate AST strictly!
        try:
            ast.parse(code)
        except SyntaxError:
            continue

        resp = f"Here is the clean, efficient Python implementation:\n\n```python\n{code}\n```"
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": resp},
            ],
            "domain": "Clean Coding & Algorithms",
        })
    return samples


# =========================================================================
# 3. High-Grade Science & Reasoning QA (ARC / OpenBookQA / HellaSwag Replay)
# =========================================================================

SCIENCE_QA = [
    ("Which object absorbs visible light best?", "A black matte cloth absorbs visible light best because dark materials absorb most wavelengths of visible light rather than reflecting them."),
    ("What part of the plant cell conducts photosynthesis?", "Chloroplasts conduct photosynthesis by containing chlorophyll that captures sunlight energy."),
    ("Why does ice float on liquid water?", "Ice floats because solid water has a lower density than liquid water due to the open hexagonal crystal lattice formed by hydrogen bonds."),
    ("What is the primary gas found in Earth's atmosphere?", "Nitrogen is the primary gas, making up approximately 78% of Earth's atmosphere, followed by oxygen at 21%."),
    ("What force keeps planets in orbit around the Sun?", "Gravity provides the centripetal force that keeps planets in elliptical orbits around the Sun."),
    ("What is the chemical formula of water?", "The chemical formula of water is H2O, meaning each molecule consists of two hydrogen atoms bonded to one oxygen atom."),
    ("Which organ pumps blood throughout the human body?", "The heart pumps oxygenated and deoxygenated blood through the circulatory system."),
    ("What causes ocean tides on Earth?", "Ocean tides are caused by the gravitational attraction of the Moon and the Sun acting on Earth's oceans."),
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."),
    ("What is the capital of Pakistan?", "The capital of Pakistan is Islamabad."),
    ("What is the largest planet in our Solar System?", "Jupiter is the largest planet in our Solar System."),
    ("What is Newton's First Law of Motion?", "Newton's First Law states that an object at rest stays at rest, and an object in motion continues in motion with constant velocity, unless acted upon by an unbalanced external force."),
]


def generate_science_and_reasoning_samples(count: int = 800) -> List[Dict[str, Any]]:
    samples = []
    for _ in range(count):
        q, a = random.choice(SCIENCE_QA)
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "domain": "Science & Reasoning QA",
        })
    return samples


# =========================================================================
# 4. Chain-of-Thought (<think>) Step-by-Step Reasoning Generator
# =========================================================================

COT_REASONING_SAMPLES = [
    (
        "If I have 10 mangoes and eat 3, how many are left?",
        "<think>\n1. Initial quantity: 10 mangoes.\n2. Subtraction: 3 mangoes are eaten.\n3. Calculation: 10 - 3 = 7.\n</think>\nYou have 7 mangoes left.",
    ),
    (
        "If I have 15 apples and give 4 to my friend and 2 to my sister, how many do I have left?",
        "<think>\n1. Total apples initially: 15.\n2. Total given away: 4 + 2 = 6.\n3. Remaining apples: 15 - 6 = 9.\n</think>\nYou have 9 apples left.",
    ),
    (
        "Write 12, 1, 4, 2 in descending order.",
        "<think>\n1. Examine the numbers: 12, 1, 4, 2.\n2. Find the largest: 12.\n3. Next largest: 4.\n4. Next largest: 2.\n5. Smallest: 1.\n6. Order from greatest to least: 12, 4, 2, 1.\n</think>\nIn descending order: 12, 4, 2, 1.",
    ),
    (
        "Write 45, 12, 89, 23 in ascending order.",
        "<think>\n1. Examine the numbers: 45, 12, 89, 23.\n2. Smallest number: 12.\n3. Next smallest: 23.\n4. Next smallest: 45.\n5. Largest: 89.\n6. Order from least to greatest: 12, 23, 45, 89.\n</think>\nIn ascending order: 12, 23, 45, 89.",
    ),
    (
        "A car travels 120 miles in 2 hours. What is its average speed in miles per hour?",
        "<think>\n1. Formula: Average Speed = Total Distance / Total Time.\n2. Distance = 120 miles.\n3. Time = 2 hours.\n4. Speed = 120 / 2 = 60 mph.\n</think>\nThe car's average speed is 60 miles per hour.",
    ),
    (
        "If a shirt costs $20 and is on a 25% discount, what is the sale price?",
        "<think>\n1. Original price = $20.\n2. Discount percentage = 25% = 0.25.\n3. Discount amount = 20 * 0.25 = $5.\n4. Sale price = 20 - 5 = $15.\n</think>\nThe sale price is $15.",
    ),
    (
        "Solve step by step: 35 + 48",
        "<think>\n1. Break into tens and units:\n   - 30 + 40 = 70\n   - 5 + 8 = 13\n2. Add the sums: 70 + 13 = 83.\n</think>\n35 + 48 = 83.",
    ),
    (
        "Which is larger: 3/4 or 2/3?",
        "<think>\n1. Convert fractions to common denominator of 12:\n   - 3/4 = (3 * 3) / 12 = 9/12\n   - 2/3 = (2 * 4) / 12 = 8/12\n2. Compare numerators: 9/12 > 8/12.\n3. Therefore, 3/4 is larger than 2/3.\n</think>\n3/4 is larger than 2/3.",
    ),
    (
        "If all cats are mammals and all mammals breathe air, do cats breathe air?",
        "<think>\n1. Premise 1: All cats are mammals.\n2. Premise 2: All mammals breathe air.\n3. Deduction: Since cats belong to the set of mammals, cats must also breathe air.\n</think>\nYes, by logical deduction, all cats breathe air.",
    ),
]


def generate_cot_reasoning_samples(count: int = 800) -> List[Dict[str, Any]]:
    samples = []
    for _ in range(count):
        q, resp = random.choice(COT_REASONING_SAMPLES)
        samples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": resp},
            ],
            "domain": "Chain-of-Thought Reasoning (<think>)",
        })
    return samples


# =========================================================================
# 5. Agentic Tool Calling Replay (Preserving 100% Tool Invocation with <think>)
# =========================================================================

TOOL_PROMPT_SYSTEM = (
    "You are an intelligent AI assistant. You have access to the following tools:\n"
    "- calculator(expression: str): Evaluates mathematical and arithmetic expressions with exact precision.\n"
    "- search_web(query: str): Searches the web for recent, real-time, or external information.\n"
    "- run_python(code: str): Executes Python code in a secure sandbox and returns the stdout output.\n\n"
    "Always think carefully step-by-step inside a <think> block first. When a tool is needed, respond with a <tool_call> block containing a JSON object with 'name' and 'arguments'."
)


def generate_tool_calling_samples(count: int = 600) -> List[Dict[str, Any]]:
    samples = []
    calc_ops = [
        ("What is 123433 * 564332?", "123433 * 564332", "69657191756"),
        ("Calculate 99 * 99", "99 * 99", "9801"),
        ("What is 48291 * 7182?", "48291 * 7182", "346825962"),
        ("Compute (450 + 250) / 7", "(450 + 250) / 7", "100.0"),
        ("What is 15 * 15?", "15 * 15", "225"),
        ("Calculate 1024 * 768", "1024 * 768", "786432"),
    ]
    search_ops = [
        ("Search online to find the meaning of gravity", "meaning of gravity", "Gravity is a fundamental natural force of attraction between physical objects with mass."),
        ("Who is Sydney Sweeney, search it on the internet", "Sydney Sweeney", "Sydney Sweeney is an acclaimed American actress known for her roles in Euphoria and The White Lotus."),
        ("Search the web for tallest building in Pakistan", "tallest building in Pakistan", "The tallest building in Pakistan is the Bahria Icon Tower located in Karachi."),
        ("Search recent news about NASA Artemis program", "NASA Artemis program", "NASA's Artemis program aims to land astronauts back on the Moon and establish long-term lunar exploration."),
    ]

    for _ in range(count):
        t_type = random.choice(["calc", "search"])
        if t_type == "calc":
            q, expr, ans = random.choice(calc_ops)
            t_call = f'<think>\nI need to evaluate the exact arithmetic calculation {expr}.\n</think>\n<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "{expr}"}}}}\n</tool_call>'
            samples.append({
                "messages": [
                    {"role": "system", "content": TOOL_PROMPT_SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": t_call},
                ],
                "domain": "Agentic: Calculator",
            })
        else:
            q, s_query, _ = random.choice(search_ops)
            t_call = f'<think>\nI should search the web for up-to-date information on {s_query}.\n</think>\n<tool_call>\n{{"name": "search_web", "arguments": {{"query": "{s_query}"}}}}\n</tool_call>'
            samples.append({
                "messages": [
                    {"role": "system", "content": TOOL_PROMPT_SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": t_call},
                ],
                "domain": "Agentic: Web Search",
            })
    return samples


def main():
    print("=" * 80)
    print("  BUILDING TARGETED SFT DATASET: exp_019_perfect_alignment (WITH <think>)")
    print("=" * 80)

    persona_samples = generate_persona_and_constraint_samples(1000)
    cot_samples = generate_cot_reasoning_samples(800)
    code_samples = generate_clean_coding_samples(700)
    science_samples = generate_science_and_reasoning_samples(700)
    tool_samples = generate_tool_calling_samples(500)

    all_samples = persona_samples + cot_samples + code_samples + science_samples + tool_samples
    random.shuffle(all_samples)

    # Split 95% train, 5% val
    split_idx = int(len(all_samples) * 0.95)
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]

    data_dir = Path("data_sft")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "sft_exp019_train.jsonl"
    val_file = data_dir / "sft_exp019_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_data:
            f.write(json.dumps(s) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_data:
            f.write(json.dumps(s) + "\n")

    print(f"✓ Total Samples: {len(all_samples)}")
    print(f"  • Persona & Negative Constraints: {len(persona_samples)}")
    print(f"  • Chain-of-Thought (<think>):     {len(cot_samples)}")
    print(f"  • Clean AST-verified Python:      {len(code_samples)}")
    print(f"  • Science & Reasoning QA:         {len(science_samples)}")
    print(f"  • Agentic Tool Calling Replay:    {len(tool_samples)}")
    print(f"✓ Saved Train ({len(train_data)}): {train_file}")
    print(f"✓ Saved Val   ({len(val_data)}):   {val_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
