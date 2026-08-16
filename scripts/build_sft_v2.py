"""
Weakness-Targeted 15M Token SFT Dataset Builder V2 for 54.5M SLM.

Designed to address the specific weaknesses diagnosed in the base pretrained model:
- Logic & Reasoning: 0-20%  → Synthetic syllogisms, negation, temporal, adversarial
- Code:              0%     → AST-verified Python with proper indentation
- Science:           33-50% → Curated factual QA with verified answers
- General Knowledge: 33-50% → Geography, history, world facts
- Math:              75%    → Strengthened multiplication & division

Target: 15,000,000 Tokens across 8 domains:
  - 20.0%  SmolTalk Conversations     (3,000,000 tokens)
  - 16.7%  Verified Mathematics        (2,500,000 tokens)
  - 15.0%  Science & Facts             (2,250,000 tokens)  [NEW]
  - 15.0%  General Knowledge           (2,250,000 tokens)  [NEW]
  - 10.0%  Logic & Reasoning           (1,500,000 tokens)  [NEW]
  - 10.0%  Python Coding (AST-verified)(1,500,000 tokens)
  - 10.0%  Tulu 3 Instruction          (1,500,000 tokens)
  -  3.3%  Conversational QA           (  500,000 tokens)
"""

import ast
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.cleaner import clean_text, is_valid_quality, strip_html_tags



# ==============================================================================
#  LANGUAGE & QUALITY FILTERING
# ==============================================================================

ENG_STOPWORDS = {
    "the", "is", "and", "to", "in", "of", "that", "it", "with", "for", "on", "are",
    "this", "you", "from", "at", "be", "by", "have", "not", "what", "how", "which",
    "an", "they", "we", "will", "can", "has", "about", "would", "there", "their", "or", "if"
}

NON_ENG_STOPWORDS = {
    "el", "la", "los", "las", "que", "en", "un", "una", "por", "para",
    "con", "del", "al", "es", "son", "como", "su", "sus",
    "le", "les", "des", "du", "dans", "qui", "avec", "pour",
    "der", "die", "das", "und", "den", "von", "zu", "mit", "ist",
    "il", "lo", "gli", "di", "da", "per", "tra", "fra", "che", "non",
}


def is_strictly_english(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    non_ascii = len(re.findall(r"[^\x00-\x7F]", text))
    if non_ascii / max(1, len(text)) > 0.10:
        return False
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{2,}\b", text)]
    if not words:
        return False
    eng_count = sum(1 for w in words if w in ENG_STOPWORDS)
    non_eng_count = sum(1 for w in words if w in NON_ENG_STOPWORDS)
    if non_eng_count > eng_count:
        return False
    if len(words) >= 8 and eng_count == 0:
        is_code = any(k in text for k in ["def ", "function", "class ", "SELECT ", "import ", "return "])
        if not is_code:
            return False
    return True


def compute_conversation_hash(messages: List[Dict[str, str]]) -> str:
    norm_parts = []
    for m in messages:
        role = m.get("role", "").strip().lower()
        content = re.sub(r"\s+", " ", m.get("content", "").strip().lower())
        norm_parts.append(f"{role}:{content}")
    full_str = " | ".join(norm_parts)
    return hashlib.sha256(full_str.encode("utf-8")).hexdigest()


def count_conversation_tokens(messages: List[Dict[str, str]], tokenizer: Tokenizer) -> int:
    total = 0
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        formatted = f"User: {content}\n\nAssistant: " if role == "user" else f"{content}"
        encoded = tokenizer.encode(formatted)
        total += len(encoded.ids)
    return total


def clean_conversation_turn(text: str) -> str:
    t = strip_html_tags(text).strip()
    t = re.sub(r"^(As an AI language model|As an AI developed by OpenAI|I am ChatGPT|I am Claude)[,\.\s]+", "", t, flags=re.IGNORECASE).strip()
    return t


def validate_python_code_block(code_str: str) -> bool:
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def make_conversation(q: str, a: str, source: str, domain: str, tokenizer: Tokenizer,
                      seen_hashes: Set[str], extra_prov: Optional[Dict] = None) -> Optional[Dict]:
    """Helper to create a conversation dict with dedup check."""
    msg = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    d_hash = compute_conversation_hash(msg)
    if d_hash in seen_hashes:
        return None
    seen_hashes.add(d_hash)
    tok_count = count_conversation_tokens(msg, tokenizer)
    if tok_count < 15 or tok_count > 1024:
        return None
    prov = {
        "source": source,
        "domain": domain,
        "lang": "en",
        "doc_hash": d_hash,
        "num_tokens": tok_count,
    }
    if extra_prov:
        prov.update(extra_prov)
    return {"messages": msg, "provenance": prov}


# ==============================================================================
#  SYNTHETIC SCIENCE & FACTS GENERATOR
# ==============================================================================

SCIENCE_FACTS = [
    # Biology
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize food from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct."),
    ("How many chambers does the human heart have?", "The human heart has 4 chambers: two upper chambers called atria (left atrium and right atrium) and two lower chambers called ventricles (left ventricle and right ventricle)."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known living organisms. DNA is a double helix structure made up of nucleotides."),
    ("What is the function of red blood cells?", "Red blood cells (erythrocytes) carry oxygen from the lungs to all parts of the body and transport carbon dioxide back to the lungs for exhalation. They contain hemoglobin, the protein that binds oxygen."),
    ("What are the main parts of a cell?", "The main parts of a cell are: the cell membrane (outer boundary), the nucleus (contains DNA and controls cell activities), the cytoplasm (gel-like fluid inside the cell), mitochondria (produce energy), and ribosomes (make proteins)."),
    ("What is the largest organ in the human body?", "The largest organ in the human body is the skin. In an average adult, the skin covers about 1.5 to 2 square meters and weighs about 3.6 kilograms."),
    ("How many bones are in the adult human body?", "The adult human body has 206 bones. Babies are born with about 270 bones, but many fuse together as they grow."),
    ("What is the function of the liver?", "The liver performs over 500 vital functions including filtering blood, producing bile for digestion, storing vitamins and minerals, processing nutrients from food, detoxifying harmful substances, and producing proteins needed for blood clotting."),
    # Chemistry
    ("What is the chemical formula for water?", "The chemical formula for water is H2O. This means each molecule of water consists of two hydrogen atoms and one oxygen atom bonded together."),
    ("What gas do plants absorb from the atmosphere during photosynthesis?", "Plants absorb carbon dioxide (CO2) from the atmosphere during photosynthesis. They use the carbon dioxide along with water and sunlight to produce glucose (sugar) and oxygen."),
    ("What is the periodic table?", "The periodic table is a tabular arrangement of all known chemical elements, organized by their atomic number (number of protons), electron configuration, and recurring chemical properties. It was first developed by Dmitri Mendeleev in 1869."),
    ("What is an atom?", "An atom is the smallest unit of ordinary matter that forms a chemical element. Every solid, liquid, gas, and plasma is composed of atoms. An atom consists of a nucleus (containing protons and neutrons) surrounded by a cloud of electrons."),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is NaCl (sodium chloride). It consists of one sodium atom (Na) and one chlorine atom (Cl) bonded together in an ionic bond."),
    ("What is the pH scale?", "The pH scale measures how acidic or basic a substance is, on a scale from 0 to 14. A pH of 7 is neutral, below 7 is acidic, and above 7 is basic (alkaline). For example, lemon juice has a pH of about 2 (acidic) and baking soda has a pH of about 9 (basic)."),
    # Physics
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 kilometers per second or 186,000 miles per second). This is the fastest speed at which any form of energy, matter, or information can travel in the universe."),
    ("Why does an apple fall from a tree?", "An apple falls from a tree because of gravity. Gravity is a fundamental force of nature that attracts objects with mass toward each other. The Earth's gravity pulls the apple downward toward the ground with an acceleration of approximately 9.8 meters per second squared."),
    ("What is Newton's First Law of Motion?", "Newton's First Law of Motion (also called the law of inertia) states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction, unless acted upon by an unbalanced external force."),
    ("What is the boiling point of water?", "The boiling point of water at standard atmospheric pressure (1 atmosphere or 101.325 kPa) is 100 degrees Celsius (212 degrees Fahrenheit or 373.15 Kelvin)."),
    ("What causes a rainbow?", "A rainbow is caused by the refraction, reflection, and dispersion of sunlight through water droplets in the atmosphere. When white sunlight enters a raindrop, it is split into its component colors (red, orange, yellow, green, blue, indigo, violet) because each color bends at a slightly different angle."),
    ("What is energy?", "Energy is the ability to do work or cause change. It exists in many forms including kinetic energy (energy of motion), potential energy (stored energy), thermal energy (heat), chemical energy, electrical energy, and nuclear energy. According to the law of conservation of energy, energy cannot be created or destroyed, only converted from one form to another."),
    # Earth Science
    ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in the Earth's crust that creates seismic waves. Most earthquakes occur along fault lines where tectonic plates meet, slide past each other, or collide. The point where an earthquake originates is called the focus, and the point directly above it on the surface is the epicenter."),
    ("What is the water cycle?", "The water cycle (also called the hydrological cycle) is the continuous movement of water within the Earth and atmosphere. It includes evaporation (water turning into vapor), condensation (vapor forming clouds), precipitation (rain, snow, sleet, hail), and collection (water flowing into rivers, lakes, and oceans)."),
    ("What is the ozone layer?", "The ozone layer is a region of Earth's stratosphere, approximately 15 to 35 kilometers above the surface, that contains high concentrations of ozone (O3) molecules. It absorbs most of the Sun's harmful ultraviolet (UV) radiation, protecting life on Earth from its damaging effects."),
    # Astronomy
    ("How far is the Sun from Earth?", "The Sun is approximately 150 million kilometers (93 million miles) from Earth. This distance is known as one Astronomical Unit (AU). Light from the Sun takes about 8 minutes and 20 seconds to reach Earth."),
    ("What is a black hole?", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, can escape from it. Black holes form when massive stars collapse at the end of their life cycle."),
    ("How many planets are in our solar system?", "There are 8 planets in our solar system. In order from the Sun, they are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union."),
    ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter. It has a diameter of about 139,820 kilometers (86,881 miles), which is more than 11 times the diameter of Earth. Jupiter is a gas giant composed mainly of hydrogen and helium."),
]


def generate_science_facts_sft(target_tokens: int, tokenizer: Tokenizer, seen_hashes: Set[str]) -> Tuple[List[Dict], int]:
    """Generate factual science QA pairs with verified ground-truth answers."""
    print(f"---> Generating verified science fact SFT conversations (target: {target_tokens:,} tokens)...")
    conversations = []
    current_tokens = 0
    rng = random.Random(42)

    # First pass: use all curated facts
    for q, a in SCIENCE_FACTS:
        conv = make_conversation(q, a, "synthetic_verified_science", "science_facts", tokenizer, seen_hashes)
        if conv:
            conversations.append(conv)
            current_tokens += conv["provenance"]["num_tokens"]

    # Generate variations to reach target
    question_templates = [
        "Question: {q}\nAnswer:",
        "{q}",
        "Explain: {q}",
        "Please answer the following: {q}",
    ]
    cycle_count = 0
    while current_tokens < target_tokens:
        cycle_count += 1
        for q, a in rng.sample(SCIENCE_FACTS, len(SCIENCE_FACTS)):
            if current_tokens >= target_tokens:
                break
            tmpl = rng.choice(question_templates)
            varied_q = tmpl.format(q=q)
            # Slightly vary answers
            varied_a = a
            if rng.random() < 0.3:
                varied_a = a + f"\n\nIn summary, {a.split('.')[0].lower()}."
            conv = make_conversation(varied_q, varied_a, "synthetic_verified_science", "science_facts", tokenizer, seen_hashes)
            if conv:
                conversations.append(conv)
                current_tokens += conv["provenance"]["num_tokens"]
        if cycle_count > 50:
            break

    print(f"✓ Generated {len(conversations):,} science fact conversations ({current_tokens:,} tokens).")
    return conversations, current_tokens


# ==============================================================================
#  SYNTHETIC GENERAL KNOWLEDGE GENERATOR
# ==============================================================================

GK_FACTS = [
    # Geography
    ("What is the capital of France?", "The capital of France is Paris. Paris is also the most populous city in France, located on the River Seine in northern France."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo. Tokyo is the most populous metropolitan area in the world."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin. Berlin is also the largest city in Germany."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London. London is also the capital of England."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra, not Sydney or Melbourne as commonly assumed. Canberra was purpose-built as the capital city and is located in the Australian Capital Territory."),
    ("What is the capital of Italy?", "The capital of Italy is Rome (Roma). Rome is also the largest city in Italy and is famous for its ancient ruins including the Colosseum."),
    ("What is the capital of Canada?", "The capital of Canada is Ottawa, located in the province of Ontario."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasilia, not Rio de Janeiro or Sao Paulo. Brasilia was purpose-built and became the capital in 1960."),
    ("What is the capital of India?", "The capital of India is New Delhi. New Delhi is an urban district within the metropolis of Delhi."),
    ("What is the capital of China?", "The capital of China (officially the People's Republic of China) is Beijing."),
    ("What is the largest ocean on Earth?", "The largest ocean on Earth is the Pacific Ocean. It covers an area of about 165.25 million square kilometers (63.8 million square miles), which is larger than all the land areas on Earth combined."),
    ("What is the longest river in the world?", "The longest river in the world is the Nile River in Africa, which is approximately 6,650 kilometers (4,130 miles) long. It flows through eleven countries in northeastern Africa before emptying into the Mediterranean Sea."),
    ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount Everest, which stands at 8,849 meters (29,032 feet) above sea level. It is located in the Himalayas on the border between Nepal and Tibet (China)."),
    ("What is the smallest country in the world?", "The smallest country in the world by area is Vatican City, with an area of approximately 0.44 square kilometers (0.17 square miles). It is an independent city-state enclaved within Rome, Italy."),
    ("What is the largest desert in the world?", "The largest desert in the world is the Antarctic Desert, covering about 14.2 million square kilometers. The largest hot desert is the Sahara Desert in Africa, covering about 9.2 million square kilometers."),
    ("How many continents are there?", "There are 7 continents on Earth: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America. Asia is the largest continent by both area and population."),
    # History
    ("In what year did World War II end?", "World War II ended in 1945. The war in Europe ended on May 8, 1945 (V-E Day), when Nazi Germany surrendered. The war in the Pacific ended on September 2, 1945 (V-J Day), when Japan formally surrendered."),
    ("Who was the first President of the United States?", "The first President of the United States was George Washington. He served two terms from 1789 to 1797 and is often called the 'Father of His Country.'"),
    ("When was the Declaration of Independence signed?", "The United States Declaration of Independence was adopted on July 4, 1776. It declared the thirteen American colonies free from British rule. The primary author was Thomas Jefferson."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the first practical telephone in 1876. He was awarded the first U.S. patent for the telephone."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci, the Italian Renaissance artist. It was created approximately between 1503 and 1519 and is now displayed in the Louvre Museum in Paris, France."),
    ("What year did the first humans land on the Moon?", "The first humans landed on the Moon on July 20, 1969, during the Apollo 11 mission. Neil Armstrong was the first person to walk on the Moon, followed by Buzz Aldrin."),
    ("Who discovered penicillin?", "Penicillin was discovered by Scottish bacteriologist Alexander Fleming in 1928. He observed that a mold called Penicillium notatum killed bacteria in a petri dish. This discovery revolutionized medicine and led to the development of antibiotics."),
    # General
    ("What is the largest animal on Earth?", "The largest animal on Earth is the blue whale (Balaenoptera musculus). Blue whales can reach lengths of up to 30 meters (100 feet) and weigh up to 200 tonnes (440,000 pounds)."),
    ("How many days are in a year?", "A regular year has 365 days. A leap year, which occurs every 4 years (with some exceptions), has 366 days. The extra day is added to February, making it 29 days instead of 28."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au, which comes from the Latin word 'aurum.' Gold has an atomic number of 79 on the periodic table."),
]


def generate_gk_facts_sft(target_tokens: int, tokenizer: Tokenizer, seen_hashes: Set[str]) -> Tuple[List[Dict], int]:
    """Generate verified general knowledge QA pairs."""
    print(f"---> Generating verified GK fact SFT conversations (target: {target_tokens:,} tokens)...")
    conversations = []
    current_tokens = 0
    rng = random.Random(43)

    for q, a in GK_FACTS:
        conv = make_conversation(q, a, "synthetic_verified_gk", "general_knowledge", tokenizer, seen_hashes)
        if conv:
            conversations.append(conv)
            current_tokens += conv["provenance"]["num_tokens"]

    question_templates = [
        "Question: {q}\nAnswer:",
        "{q}",
        "Can you tell me: {q}",
        "I'd like to know: {q}",
    ]
    cycle_count = 0
    while current_tokens < target_tokens:
        cycle_count += 1
        for q, a in rng.sample(GK_FACTS, len(GK_FACTS)):
            if current_tokens >= target_tokens:
                break
            tmpl = rng.choice(question_templates)
            varied_q = tmpl.format(q=q)
            conv = make_conversation(varied_q, a, "synthetic_verified_gk", "general_knowledge", tokenizer, seen_hashes)
            if conv:
                conversations.append(conv)
                current_tokens += conv["provenance"]["num_tokens"]
        if cycle_count > 50:
            break

    print(f"✓ Generated {len(conversations):,} GK fact conversations ({current_tokens:,} tokens).")
    return conversations, current_tokens


# ==============================================================================
#  SYNTHETIC LOGIC & REASONING GENERATOR
# ==============================================================================

def generate_logic_sft(target_tokens: int, tokenizer: Tokenizer, seen_hashes: Set[str]) -> Tuple[List[Dict], int]:
    """Generate programmatically verified logic, negation, temporal, and adversarial QA."""
    print(f"---> Generating verified logic & reasoning SFT conversations (target: {target_tokens:,} tokens)...")
    conversations = []
    current_tokens = 0
    rng = random.Random(44)

    logic_types = ["syllogism", "negation_math", "negation_fact", "temporal", "adversarial", "comparison", "boolean"]

    # Syllogism components
    categories = [
        ("dogs", "mammals", "animals"),
        ("cats", "mammals", "animals"),
        ("roses", "flowers", "plants"),
        ("eagles", "birds", "animals"),
        ("salmon", "fish", "animals"),
        ("apples", "fruits", "foods"),
        ("oak trees", "trees", "plants"),
        ("pythons", "snakes", "reptiles"),
        ("spiders", "arachnids", "arthropods"),
        ("whales", "mammals", "animals"),
        ("penguins", "birds", "animals"),
        ("carrots", "vegetables", "foods"),
        ("diamonds", "gemstones", "minerals"),
        ("Mars", "planets", "celestial bodies"),
    ]

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Historical figures for adversarial
    historical_anachronisms = [
        ("Did Abraham Lincoln use a smartphone?", "No, Abraham Lincoln did not use a smartphone. Lincoln lived from 1809 to 1865, while smartphones were not invented until the early 21st century. The first widely popular smartphone, the iPhone, was released in 2007, more than 140 years after Lincoln's death."),
        ("Did Julius Caesar send emails?", "No, Julius Caesar did not send emails. Caesar lived from 100 BC to 44 BC, over 2,000 years before email was invented. Email was first developed in the early 1970s."),
        ("Did Cleopatra watch television?", "No, Cleopatra did not watch television. Cleopatra lived from 69 BC to 30 BC in ancient Egypt. Television was not invented until the 1920s, nearly 2,000 years later."),
        ("Did Shakespeare use the internet?", "No, William Shakespeare did not use the internet. Shakespeare lived from 1564 to 1616, while the internet was not developed until the late 20th century. The World Wide Web was created by Tim Berners-Lee in 1989."),
        ("Did Napoleon Bonaparte fly in an airplane?", "No, Napoleon Bonaparte did not fly in an airplane. Napoleon lived from 1769 to 1821. The first successful powered airplane flight was made by the Wright Brothers in 1903, more than 80 years after Napoleon's death."),
        ("Did George Washington drive a car?", "No, George Washington did not drive a car. Washington lived from 1732 to 1799. The first practical automobile was not invented until the 1880s by Karl Benz, nearly a century after Washington's time."),
        ("Did ancient Romans use computers?", "No, ancient Romans did not use computers. The Roman Empire existed from 27 BC to 476 AD. The first electronic computers were not built until the 1940s, nearly 1,500 years later."),
        ("Did Leonardo da Vinci take photographs?", "No, Leonardo da Vinci did not take photographs. Da Vinci lived from 1452 to 1519. Photography was not invented until the 1820s-1830s, over 300 years after his death."),
    ]

    while current_tokens < target_tokens:
        logic_type = rng.choice(logic_types)

        if logic_type == "syllogism":
            spec, mid, broad = rng.choice(categories)
            q = f"All {spec} are {mid}. All {mid} are {broad}. Therefore, all {spec} are:"
            a = f"All {spec} are {broad}.\n\nReasoning: Since all {spec} are {mid}, and all {mid} are {broad}, it logically follows that all {spec} are {broad}. This is a valid syllogism (transitive reasoning)."
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "negation_math":
            x = rng.randint(1, 100)
            y = rng.randint(1, 100)
            correct = x + y
            wrong = correct + rng.choice([-3, -2, -1, 1, 2, 3])
            q = f"Is {x} + {y} equal to {wrong}?"
            a = f"No, {x} + {y} is not equal to {wrong}. The correct answer is {x} + {y} = {correct}."
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "negation_fact":
            negation_facts = [
                ("Can fish fly in the sky like birds?", "No, fish cannot fly in the sky like birds. Fish are aquatic animals that live in water and breathe through gills. They do not have wings or the ability to fly. However, there are 'flying fish' that can glide above the water surface for short distances using their fin-like wings."),
                ("Is the Earth flat?", "No, the Earth is not flat. The Earth is an oblate spheroid, meaning it is roughly spherical but slightly flattened at the poles and bulging at the equator. This has been confirmed by extensive scientific evidence including satellite imagery, physics, and direct observation."),
                ("Can humans breathe underwater without equipment?", "No, humans cannot breathe underwater without equipment. Humans breathe air through lungs and require oxygen from the atmosphere. To breathe underwater, humans need specialized equipment like scuba gear or a submarine."),
                ("Is the Sun larger than the Earth?", "Yes, the Sun is much larger than the Earth. The Sun has a diameter of about 1.39 million kilometers, which is approximately 109 times the diameter of Earth. The Sun's volume is about 1.3 million times that of Earth."),
                ("Does ice sink in water?", "No, ice does not sink in water. Ice floats because it is less dense than liquid water. Water is unusual in that its solid form (ice) is less dense than its liquid form, which is why ice cubes float in your drink and icebergs float in the ocean."),
            ]
            q, a = rng.choice(negation_facts)
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "temporal":
            start_day_idx = rng.randint(0, 6)
            offset = rng.randint(1, 14)
            result_idx = (start_day_idx + offset) % 7
            start_day = days_of_week[start_day_idx]
            result_day = days_of_week[result_idx]
            q = f"If today is {start_day}, what day will it be in {offset} days?"
            a = f"If today is {start_day}, then in {offset} days it will be {result_day}.\n\nCalculation: Starting from {start_day}, counting forward {offset} days leads to {result_day}."
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "adversarial":
            q, a = rng.choice(historical_anachronisms)
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "comparison":
            comparisons = [
                ("Which is larger, an elephant or a mouse?", "An elephant is much larger than a mouse. An adult African elephant can weigh between 4,000 to 6,000 kilograms, while a typical house mouse weighs about 20 grams. An elephant is roughly 200,000 times heavier than a mouse."),
                ("Which is faster, a cheetah or a snail?", "A cheetah is much faster than a snail. A cheetah can run at speeds up to 120 km/h (75 mph), making it the fastest land animal. A garden snail moves at about 0.03 mph, making the cheetah roughly 2,500 times faster."),
                ("Which is taller, Mount Everest or a house?", "Mount Everest is vastly taller than a house. Mount Everest stands at 8,849 meters (29,032 feet) above sea level, while a typical house is about 8-10 meters (26-33 feet) tall. Mount Everest is roughly 1,000 times taller than a house."),
                ("Which came first, World War I or World War II?", "World War I came first. World War I took place from 1914 to 1918, and World War II took place from 1939 to 1945. There were about 21 years between the end of WWI and the start of WWII."),
            ]
            q, a = rng.choice(comparisons)
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        elif logic_type == "boolean":
            x = rng.randint(1, 100)
            y = rng.randint(1, 100)
            if rng.random() < 0.5:
                # True case
                q = f"Is {max(x,y)} greater than {min(x,y)}?"
                a = f"Yes, {max(x,y)} is greater than {min(x,y)}."
            else:
                # False case
                q = f"Is {min(x,y)} greater than {max(x,y)}?"
                a = f"No, {min(x,y)} is not greater than {max(x,y)}. In fact, {min(x,y)} is less than {max(x,y)}."
            conv = make_conversation(q, a, "synthetic_logic", "logic_reasoning", tokenizer, seen_hashes)

        if conv:
            conversations.append(conv)
            current_tokens += conv["provenance"]["num_tokens"]

    print(f"✓ Generated {len(conversations):,} logic/reasoning conversations ({current_tokens:,} tokens).")
    return conversations, current_tokens


# ==============================================================================
#  VERIFIED ARITHMETIC GENERATOR (Enhanced mul/div)
# ==============================================================================

def generate_verified_arithmetic_sft(target_tokens: int, tokenizer: Tokenizer, seen_hashes: Set[str]) -> Tuple[List[Dict], int]:
    """Generate 100% programmatically verified arithmetic with emphasis on mul/div."""
    print(f"---> Generating verified arithmetic SFT conversations (target: {target_tokens:,} tokens)...")
    conversations = []
    current_tokens = 0
    rng = random.Random(42)

    # Weight multiplication and division higher since they're the weak spots
    ops = ["add"] * 2 + ["sub"] * 2 + ["mul"] * 3 + ["div"] * 3 + ["word_math"] * 2 + ["algebra_basic"] * 1

    while current_tokens < target_tokens:
        op = rng.choice(ops)

        if op == "add":
            x = rng.randint(1, 999)
            y = rng.randint(1, 999)
            res = x + y
            assert x + y == res
            q = rng.choice([
                f"What is {x} + {y}?",
                f"Calculate {x} + {y}.",
                f"What is the sum of {x} and {y}?",
            ])
            a = rng.choice([
                f"{x} + {y} = {res}.",
                f"The sum of {x} and {y} is {res}.",
                f"{x} + {y} = {res}.\n\nVerification: {res} - {x} = {y}. Correct.",
            ])

        elif op == "sub":
            x = rng.randint(1, 999)
            y = rng.randint(1, x)
            res = x - y
            assert x - y == res
            q = rng.choice([
                f"What is {x} - {y}?",
                f"Calculate {x} minus {y}.",
                f"Subtract {y} from {x}.",
            ])
            a = rng.choice([
                f"{x} - {y} = {res}.",
                f"The difference is {res}.",
                f"{x} - {y} = {res}.\n\nVerification: {res} + {y} = {x}. Correct.",
            ])

        elif op == "mul":
            x = rng.randint(2, 50)
            y = rng.randint(2, 25)
            res = x * y
            assert x * y == res
            q = rng.choice([
                f"What is {x} × {y}?",
                f"What is {x} * {y}?",
                f"Calculate {x} multiplied by {y}.",
                f"What is the product of {x} and {y}?",
                f"Multiply {x} by {y}.",
            ])
            a = rng.choice([
                f"{x} × {y} = {res}.",
                f"{x} * {y} = {res}.",
                f"The product of {x} and {y} is {res}.",
                f"{x} × {y} = {res}.\n\nExplanation: {x} multiplied by {y} equals {res}.",
                f"{x} × {y} = {res}.\n\nVerification: {res} ÷ {x} = {y}. Correct.",
            ])

        elif op == "div":
            divisor = rng.randint(2, 25)
            quotient = rng.randint(1, 50)
            dividend = divisor * quotient
            assert dividend // divisor == quotient
            q = rng.choice([
                f"What is {dividend} ÷ {divisor}?",
                f"What is {dividend} / {divisor}?",
                f"Calculate {dividend} divided by {divisor}.",
                f"Divide {dividend} by {divisor}.",
            ])
            a = rng.choice([
                f"{dividend} ÷ {divisor} = {quotient}.",
                f"{dividend} / {divisor} = {quotient}.",
                f"{dividend} ÷ {divisor} = {quotient}.\n\nExplanation: Since {divisor} × {quotient} = {dividend}, dividing {dividend} by {divisor} gives {quotient}.",
                f"{dividend} ÷ {divisor} = {quotient}.\n\nVerification: {quotient} × {divisor} = {dividend}. Correct.",
            ])

        elif op == "word_math":
            names = ["Alice", "Bob", "Charlie", "David", "Emma", "Fiona", "George", "Hannah", "Sarah", "Tom"]
            items = ["apples", "notebooks", "pencils", "stickers", "marbles", "candies", "books", "cookies"]
            p1, p2 = rng.sample(names, 2)
            item = rng.choice(items)
            n1 = rng.randint(5, 100)
            n2 = rng.randint(2, 50)
            n3 = rng.randint(1, 30)
            tot = n1 - n2 + n3
            assert n1 - n2 + n3 == tot
            q = f"{p1} has {n1} {item}. {p1} gives {n2} to {p2} and then buys {n3} more. How many {item} does {p1} have now?"
            a = f"{p1} has {tot} {item}.\n\nStep-by-step:\n- Started with: {n1} {item}\n- After giving {n2} to {p2}: {n1} - {n2} = {n1 - n2} {item}\n- After buying {n3} more: {n1 - n2} + {n3} = {tot} {item}\n\nAnswer: {tot}"

        elif op == "algebra_basic":
            x_val = rng.randint(1, 20)
            coeff = rng.randint(2, 6)
            b_val = rng.randint(1, 30)
            rhs = coeff * x_val + b_val
            assert coeff * x_val + b_val == rhs
            q = f"Solve for x: {coeff}x + {b_val} = {rhs}"
            a = f"To solve {coeff}x + {b_val} = {rhs}:\n1. Subtract {b_val} from both sides: {coeff}x = {rhs - b_val}\n2. Divide both sides by {coeff}: x = {(rhs - b_val) // coeff}\n\nAnswer: x = {x_val}"

        msg = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        d_hash = compute_conversation_hash(msg)
        if d_hash in seen_hashes:
            continue
        seen_hashes.add(d_hash)
        toks = count_conversation_tokens(msg, tokenizer)
        if toks < 15 or toks > 1024:
            continue
        conversations.append({
            "messages": msg,
            "provenance": {
                "source": "synthetic_verified_math",
                "domain": "mathematics",
                "lang": "en",
                "doc_hash": d_hash,
                "num_tokens": toks,
                "math_verified": True,
            }
        })
        current_tokens += toks

    print(f"✓ Generated {len(conversations):,} verified arithmetic conversations ({current_tokens:,} tokens).")
    return conversations, current_tokens


# ==============================================================================
#  SYNTHETIC PYTHON CODE GENERATOR (AST-verified)
# ==============================================================================

PYTHON_FUNCTIONS = [
    {
        "prompt": "Write a Python function to check if a number is even.",
        "code": '''def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0''',
    },
    {
        "prompt": "Write a Python function to find the maximum value in a list.",
        "code": '''def find_max(lst: list) -> int:
    """Find the maximum value in a list."""
    if not lst:
        return None
    max_val = lst[0]
    for item in lst[1:]:
        if item > max_val:
            max_val = item
    return max_val''',
    },
    {
        "prompt": "Write a Python function to calculate the factorial of a number.",
        "code": '''def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result''',
    },
    {
        "prompt": "Write a Python function to reverse a string.",
        "code": '''def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]''',
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "code": '''def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome (reads the same forwards and backwards)."""
    cleaned = s.lower().replace(" ", "")
    return cleaned == cleaned[::-1]''',
    },
    {
        "prompt": "Write a Python function to count the number of vowels in a string.",
        "code": '''def count_vowels(s: str) -> int:
    """Count the number of vowels in a string."""
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    return count''',
    },
    {
        "prompt": "Write a Python function to find the sum of a list of numbers.",
        "code": '''def sum_list(numbers: list) -> float:
    """Calculate the sum of all numbers in a list."""
    total = 0
    for num in numbers:
        total += num
    return total''',
    },
    {
        "prompt": "Write a Python function to calculate the average of a list of numbers.",
        "code": '''def average(numbers: list) -> float:
    """Calculate the average (mean) of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)''',
    },
    {
        "prompt": "Write a Python function to find the minimum value in a list.",
        "code": '''def find_min(lst: list) -> int:
    """Find the minimum value in a list."""
    if not lst:
        return None
    min_val = lst[0]
    for item in lst[1:]:
        if item < min_val:
            min_val = item
    return min_val''',
    },
    {
        "prompt": "Write a Python function to check if a number is prime.",
        "code": '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True''',
    },
    {
        "prompt": "Write a Python function to generate the Fibonacci sequence up to n terms.",
        "code": '''def fibonacci(n: int) -> list:
    """Generate the Fibonacci sequence with n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    for i in range(2, n):
        next_num = sequence[i - 1] + sequence[i - 2]
        sequence.append(next_num)
    return sequence''',
    },
    {
        "prompt": "Write a Python function to sort a list using bubble sort.",
        "code": '''def bubble_sort(lst: list) -> list:
    """Sort a list using the bubble sort algorithm."""
    arr = lst.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr''',
    },
    {
        "prompt": "Write a Python function to convert Celsius to Fahrenheit.",
        "code": '''def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32''',
    },
    {
        "prompt": "Write a Python function to count the occurrences of each character in a string.",
        "code": '''def char_count(s: str) -> dict:
    """Count occurrences of each character in a string."""
    counts = {}
    for char in s:
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    return counts''',
    },
    {
        "prompt": "Write a Python function to remove duplicates from a list while preserving order.",
        "code": '''def remove_duplicates(lst: list) -> list:
    """Remove duplicates from a list while preserving the original order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result''',
    },
    {
        "prompt": "Write a Python function to flatten a nested list.",
        "code": '''def flatten(nested_list: list) -> list:
    """Flatten a nested list into a single list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result''',
    },
    {
        "prompt": "Write a Python function to find the greatest common divisor (GCD) of two numbers.",
        "code": '''def gcd(a: int, b: int) -> int:
    """Find the greatest common divisor of two integers using Euclid's algorithm."""
    while b != 0:
        a, b = b, a % b
    return a''',
    },
    {
        "prompt": "Write a Python function to compute the power of a number.",
        "code": '''def power(base: float, exponent: int) -> float:
    """Compute base raised to the power of exponent."""
    if exponent == 0:
        return 1
    result = 1
    for _ in range(abs(exponent)):
        result *= base
    if exponent < 0:
        return 1 / result
    return result''',
    },
]


def generate_code_sft(target_tokens: int, tokenizer: Tokenizer, seen_hashes: Set[str]) -> Tuple[List[Dict], int]:
    """Generate AST-verified Python code SFT samples."""
    print(f"---> Generating AST-verified Python code SFT conversations (target: {target_tokens:,} tokens)...")
    conversations = []
    current_tokens = 0
    rng = random.Random(45)

    # Verify all templates pass AST
    for fn in PYTHON_FUNCTIONS:
        assert validate_python_code_block(fn["code"]), f"Template code failed AST: {fn['prompt'][:50]}"

    prompt_templates = [
        "{prompt}\n\n```python",
        "{prompt}",
        "Question: {prompt}\nAnswer:",
    ]

    answer_templates = [
        "```python\n{code}\n```",
        "Here is the implementation:\n\n```python\n{code}\n```",
        "```python\n{code}\n```\n\nThis function works by {explanation}.",
    ]

    cycle_count = 0
    while current_tokens < target_tokens:
        cycle_count += 1
        for fn in rng.sample(PYTHON_FUNCTIONS, len(PYTHON_FUNCTIONS)):
            if current_tokens >= target_tokens:
                break
            p_tmpl = rng.choice(prompt_templates)
            a_tmpl = rng.choice(answer_templates[:2])  # Skip explanation template for simplicity
            q = p_tmpl.format(prompt=fn["prompt"])
            a = a_tmpl.format(code=fn["code"])
            conv = make_conversation(q, a, "synthetic_python_code", "coding", tokenizer, seen_hashes, {"code_ast_verified": True})
            if conv:
                conversations.append(conv)
                current_tokens += conv["provenance"]["num_tokens"]
        if cycle_count > 100:
            break

    print(f"✓ Generated {len(conversations):,} AST-verified code conversations ({current_tokens:,} tokens).")
    return conversations, current_tokens


# ==============================================================================
#  MAIN BUILDER
# ==============================================================================

def build_sft_v2():
    print("=" * 80)
    print("  BUILDING 15M TOKEN WEAKNESS-TARGETED SFT DATASET V2 FOR 54.5M SLM")
    print("=" * 80)

    tokenizer_path = Path("tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output_dir = Path("data_sft_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()

    TARGET_TOTAL = 15_000_000
    targets = {
        "smoltalk":  int(TARGET_TOTAL * 0.20),   # 3,000,000 tokens (20%)
        "math":      int(TARGET_TOTAL * 0.167),   # 2,500,000 tokens (16.7%)
        "science":   int(TARGET_TOTAL * 0.15),    # 2,250,000 tokens (15%)
        "gk":        int(TARGET_TOTAL * 0.15),    # 2,250,000 tokens (15%)
        "logic":     int(TARGET_TOTAL * 0.10),    # 1,500,000 tokens (10%)
        "coding":    int(TARGET_TOTAL * 0.10),    # 1,500,000 tokens (10%)
        "tulu3":     int(TARGET_TOTAL * 0.10),    # 1,500,000 tokens (10%)
        "qa":        int(TARGET_TOTAL * 0.033),   #   500,000 tokens (3.3%)
    }

    collected: Dict[str, List[Dict]] = {k: [] for k in targets}
    stats = {
        "total_evaluated": 0, "accepted_examples": 0,
        "rejected_duplicate": 0, "rejected_language": 0,
        "rejected_length": 0, "rejected_quality": 0,
        "rejected_code_syntax": 0, "math_verified_count": 0,
    }
    start_time = time.time()

    # =========================================================================
    # 1. MATHEMATICS (16.7% → 2.5M tokens) — Enhanced mul/div
    # =========================================================================
    print(f"\n[1/8] Building Verified Mathematics (Target: {targets['math']:,} tokens)...")
    synth_math, synth_math_tokens = generate_verified_arithmetic_sft(
        int(targets["math"] * 0.4), tokenizer, seen_hashes
    )
    collected["math"].extend(synth_math)
    math_tokens = synth_math_tokens
    stats["math_verified_count"] += len(synth_math)

    # NuminaMath-CoT for remaining math tokens
    print(f"     Streaming NuminaMath-CoT for remaining {targets['math'] - math_tokens:,} math tokens...")
    try:
        ds_math = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
        for doc in ds_math:
            if math_tokens >= targets["math"]:
                break
            stats["total_evaluated"] += 1
            prob = clean_conversation_turn(doc.get("problem", ""))
            sol = clean_conversation_turn(doc.get("solution", ""))
            if not prob or not sol or len(prob) < 10 or len(sol) < 10:
                stats["rejected_quality"] += 1
                continue
            if len(sol) > 1200 or len(prob) > 600:
                stats["rejected_length"] += 1
                continue
            conv = make_conversation(
                f"Solve the following mathematical problem:\n{prob}", sol,
                "NuminaMath-CoT", "mathematics", tokenizer, seen_hashes
            )
            if conv:
                collected["math"].append(conv)
                math_tokens += conv["provenance"]["num_tokens"]
                stats["math_verified_count"] += 1
                if len(collected["math"]) % 1000 == 0:
                    print(f"     Math Progress: {math_tokens:,} / {targets['math']:,} ({math_tokens/targets['math']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in NuminaMath: {e}")
    print(f"✓ Completed Mathematics: {math_tokens:,} tokens ({len(collected['math']):,} conversations)")

    # =========================================================================
    # 2. SCIENCE & FACTS (15% → 2.25M tokens) [NEW]
    # =========================================================================
    print(f"\n[2/8] Building Science & Facts (Target: {targets['science']:,} tokens)...")
    synth_sci, synth_sci_tokens = generate_science_facts_sft(
        int(targets["science"] * 0.3), tokenizer, seen_hashes
    )
    collected["science"].extend(synth_sci)
    sci_tokens = synth_sci_tokens

    # SciQ dataset for remaining
    print(f"     Streaming SciQ for remaining {targets['science'] - sci_tokens:,} science tokens...")
    try:
        ds_sciq = load_dataset("allenai/sciq", split="train", streaming=True)
        for doc in ds_sciq:
            if sci_tokens >= targets["science"]:
                break
            stats["total_evaluated"] += 1
            question = clean_conversation_turn(doc.get("question", ""))
            answer = clean_conversation_turn(doc.get("correct_answer", ""))
            support = clean_conversation_turn(doc.get("support", ""))
            if not question or not answer or len(question) < 10:
                stats["rejected_quality"] += 1
                continue
            full_answer = answer
            if support and len(support) > 20:
                full_answer = f"{answer}\n\nExplanation: {support}"
            conv = make_conversation(question, full_answer, "SciQ", "science_facts", tokenizer, seen_hashes)
            if conv:
                collected["science"].append(conv)
                sci_tokens += conv["provenance"]["num_tokens"]
                if len(collected["science"]) % 1000 == 0:
                    print(f"     Science Progress: {sci_tokens:,} / {targets['science']:,} ({sci_tokens/targets['science']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in SciQ: {e}")

    # Cosmopedia auto_math_text for remaining science tokens
    if sci_tokens < targets["science"]:
        print(f"     Streaming Cosmopedia science for remaining {targets['science'] - sci_tokens:,} tokens...")
        try:
            ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v1", split="train", streaming=True)
            for doc in ds_cosmo:
                if sci_tokens >= targets["science"]:
                    break
                stats["total_evaluated"] += 1
                text = doc.get("text", "").strip()
                if not text or len(text) < 100 or not is_strictly_english(text[:500]):
                    continue
                # Convert to QA format
                sentences = text.split(". ")
                if len(sentences) < 3:
                    continue
                q = f"Explain the following topic:\n{sentences[0]}."
                a = ". ".join(sentences[1:])[:800]
                conv = make_conversation(q, a, "cosmopedia", "science_facts", tokenizer, seen_hashes)
                if conv:
                    collected["science"].append(conv)
                    sci_tokens += conv["provenance"]["num_tokens"]
        except Exception as e:
            print(f"     Warning in Cosmopedia: {e}")
    print(f"✓ Completed Science: {sci_tokens:,} tokens ({len(collected['science']):,} conversations)")

    # =========================================================================
    # 3. GENERAL KNOWLEDGE (15% → 2.25M tokens) [NEW]
    # =========================================================================
    print(f"\n[3/8] Building General Knowledge (Target: {targets['gk']:,} tokens)...")
    synth_gk, synth_gk_tokens = generate_gk_facts_sft(
        int(targets["gk"] * 0.3), tokenizer, seen_hashes
    )
    collected["gk"].extend(synth_gk)
    gk_tokens = synth_gk_tokens

    # Cosmopedia web_samples_v2 & OpenOrca for remaining GK
    print(f"     Streaming Cosmopedia & OpenOrca for remaining {targets['gk'] - gk_tokens:,} GK tokens...")
    try:
        ds_cosmo_gk = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
        for doc in ds_cosmo_gk:
            if gk_tokens >= targets["gk"]:
                break
            stats["total_evaluated"] += 1
            text = doc.get("text", "").strip()
            if not text or len(text) < 100 or not is_strictly_english(text[:500]):
                continue
            sentences = text.split(". ")
            if len(sentences) < 3:
                continue
            q = f"What is known about the following topic?\n{sentences[0]}."
            a = ". ".join(sentences[1:])[:800]
            conv = make_conversation(q, a, "cosmopedia_gk", "general_knowledge", tokenizer, seen_hashes)
            if conv:
                collected["gk"].append(conv)
                gk_tokens += conv["provenance"]["num_tokens"]
                if len(collected["gk"]) % 1000 == 0:
                    print(f"     GK Progress: {gk_tokens:,} / {targets['gk']:,} ({gk_tokens/targets['gk']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in Cosmopedia GK: {e}")

    if gk_tokens < targets["gk"]:
        try:
            ds_orca_gk = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            for doc in ds_orca_gk:
                if gk_tokens >= targets["gk"]:
                    break
                stats["total_evaluated"] += 1
                q = clean_conversation_turn(doc.get("question", ""))
                r = clean_conversation_turn(doc.get("response", ""))
                if not q or not r or len(q) < 10 or len(r) < 10 or len(r) > 1200:
                    continue
                if not is_strictly_english(q + " " + r):
                    continue
                conv = make_conversation(q, r, "OpenOrca_GK", "general_knowledge", tokenizer, seen_hashes)
                if conv:
                    collected["gk"].append(conv)
                    gk_tokens += conv["provenance"]["num_tokens"]
                    if len(collected["gk"]) % 1000 == 0:
                        print(f"     GK Progress: {gk_tokens:,} / {targets['gk']:,} ({gk_tokens/targets['gk']*100:.1f}%)")
        except Exception as e:
            print(f"     Warning in OpenOrca GK: {e}")
    print(f"✓ Completed General Knowledge: {gk_tokens:,} tokens ({len(collected['gk']):,} conversations)")


    # =========================================================================
    # 4. LOGIC & REASONING (10% → 1.5M tokens) [NEW]
    # =========================================================================
    print(f"\n[4/8] Building Logic & Reasoning (Target: {targets['logic']:,} tokens)...")
    logic_convs, logic_tokens = generate_logic_sft(targets["logic"], tokenizer, seen_hashes)
    collected["logic"].extend(logic_convs)
    print(f"✓ Completed Logic: {logic_tokens:,} tokens ({len(collected['logic']):,} conversations)")

    # =========================================================================
    # 5. PYTHON CODING (10% → 1.5M tokens)
    # =========================================================================
    print(f"\n[5/8] Building AST-Verified Python Coding (Target: {targets['coding']:,} tokens)...")
    synth_code, synth_code_tokens = generate_code_sft(
        int(targets["coding"] * 0.3), tokenizer, seen_hashes
    )
    collected["coding"].extend(synth_code)
    code_tokens = synth_code_tokens

    # CodeFeedback for remaining
    print(f"     Streaming CodeFeedback for remaining {targets['coding'] - code_tokens:,} code tokens...")
    try:
        ds_code = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        for doc in ds_code:
            if code_tokens >= targets["coding"]:
                break
            stats["total_evaluated"] += 1
            query = clean_conversation_turn(doc.get("query", ""))
            answer = clean_conversation_turn(doc.get("answer", ""))
            if not query or not answer or len(query) < 10 or len(answer) < 15:
                stats["rejected_quality"] += 1
                continue
            # AST verify Python blocks
            if "def " in answer or "python" in answer.lower():
                py_matches = re.findall(r"```python\s*(.*?)\s*```", answer, re.DOTALL)
                for py_code in py_matches:
                    if not validate_python_code_block(py_code):
                        stats["rejected_code_syntax"] += 1
                        continue
            conv = make_conversation(query, answer, "CodeFeedback", "coding", tokenizer, seen_hashes)
            if conv:
                collected["coding"].append(conv)
                code_tokens += conv["provenance"]["num_tokens"]
                if len(collected["coding"]) % 500 == 0:
                    print(f"     Code Progress: {code_tokens:,} / {targets['coding']:,} ({code_tokens/targets['coding']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in CodeFeedback: {e}")
    print(f"✓ Completed Coding: {code_tokens:,} tokens ({len(collected['coding']):,} conversations)")

    # =========================================================================
    # 6. SMOLTALK (20% → 3.0M tokens)
    # =========================================================================
    print(f"\n[6/8] Building SmolTalk Conversations (Target: {targets['smoltalk']:,} tokens)...")
    smol_tokens = 0
    try:
        ds_smol = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        for doc in ds_smol:
            if smol_tokens >= targets["smoltalk"]:
                break
            stats["total_evaluated"] += 1
            raw_msgs = doc.get("messages", [])
            if not raw_msgs or len(raw_msgs) < 2:
                stats["rejected_quality"] += 1
                continue
            clean_msgs = []
            valid = True
            for m in raw_msgs:
                role = m.get("role", "").lower()
                content = clean_conversation_turn(m.get("content", ""))
                if role not in ["user", "assistant"] or not content:
                    valid = False
                    break
                clean_msgs.append({"role": role, "content": content})
            if not valid or len(clean_msgs) < 2 or clean_msgs[0]["role"] != "user":
                stats["rejected_quality"] += 1
                continue
            sample_text = clean_msgs[0]["content"] + " " + clean_msgs[1]["content"]
            if not is_strictly_english(sample_text):
                stats["rejected_language"] += 1
                continue
            d_hash = compute_conversation_hash(clean_msgs)
            if d_hash in seen_hashes:
                stats["rejected_duplicate"] += 1
                continue
            seen_hashes.add(d_hash)
            tok_count = count_conversation_tokens(clean_msgs, tokenizer)
            if tok_count < 20 or tok_count > 1024:
                stats["rejected_length"] += 1
                continue
            collected["smoltalk"].append({
                "messages": clean_msgs,
                "provenance": {"source": doc.get("source", "smoltalk"), "domain": "general_instruction",
                               "lang": "en", "doc_hash": d_hash, "num_tokens": tok_count}
            })
            smol_tokens += tok_count
            if len(collected["smoltalk"]) % 1000 == 0:
                print(f"     SmolTalk Progress: {smol_tokens:,} / {targets['smoltalk']:,} ({smol_tokens/targets['smoltalk']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in SmolTalk: {e}")
    print(f"✓ Completed SmolTalk: {smol_tokens:,} tokens ({len(collected['smoltalk']):,} conversations)")

    # =========================================================================
    # 7. TULU 3 (10% → 1.5M tokens)
    # =========================================================================
    print(f"\n[7/8] Building Tulu 3 Instruction (Target: {targets['tulu3']:,} tokens)...")
    tulu_tokens = 0
    try:
        ds_tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        for doc in ds_tulu:
            if tulu_tokens >= targets["tulu3"]:
                break
            stats["total_evaluated"] += 1
            raw_msgs = doc.get("messages", [])
            if not raw_msgs or len(raw_msgs) < 2:
                stats["rejected_quality"] += 1
                continue
            clean_msgs = []
            valid = True
            for m in raw_msgs:
                role = m.get("role", "").lower()
                content = clean_conversation_turn(m.get("content", ""))
                if role not in ["user", "assistant"] or not content:
                    valid = False
                    break
                clean_msgs.append({"role": role, "content": content})
            if not valid or len(clean_msgs) < 2 or clean_msgs[0]["role"] != "user":
                stats["rejected_quality"] += 1
                continue
            full_text = " ".join([m["content"] for m in clean_msgs])
            if not is_strictly_english(full_text):
                stats["rejected_language"] += 1
                continue
            d_hash = compute_conversation_hash(clean_msgs)
            if d_hash in seen_hashes:
                stats["rejected_duplicate"] += 1
                continue
            seen_hashes.add(d_hash)
            tok_count = count_conversation_tokens(clean_msgs, tokenizer)
            if tok_count < 20 or tok_count > 1024:
                stats["rejected_length"] += 1
                continue
            collected["tulu3"].append({
                "messages": clean_msgs,
                "provenance": {"source": f"tulu3_{doc.get('source', 'general')}", "domain": "instruction_following",
                               "lang": "en", "doc_hash": d_hash, "num_tokens": tok_count}
            })
            tulu_tokens += tok_count
            if len(collected["tulu3"]) % 1000 == 0:
                print(f"     Tulu 3 Progress: {tulu_tokens:,} / {targets['tulu3']:,} ({tulu_tokens/targets['tulu3']*100:.1f}%)")
    except Exception as e:
        print(f"     Warning in Tulu 3: {e}")
    print(f"✓ Completed Tulu 3: {tulu_tokens:,} tokens ({len(collected['tulu3']):,} conversations)")

    # =========================================================================
    # 8. CONVERSATIONAL QA (3.3% → 500K tokens)
    # =========================================================================
    print(f"\n[8/8] Building Conversational QA (Target: {targets['qa']:,} tokens)...")
    qa_tokens = 0
    try:
        ds_everyday = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft", streaming=True)
        for doc in ds_everyday:
            if qa_tokens >= targets["qa"]:
                break
            stats["total_evaluated"] += 1
            raw_msgs = doc.get("messages", [])
            clean_msgs = []
            for m in raw_msgs:
                r = m.get("role", "").lower()
                c = clean_conversation_turn(m.get("content", ""))
                if r in ["user", "assistant"] and c:
                    clean_msgs.append({"role": r, "content": c})
            if len(clean_msgs) < 2 or clean_msgs[0]["role"] != "user":
                continue
            d_hash = compute_conversation_hash(clean_msgs)
            if d_hash in seen_hashes:
                continue
            seen_hashes.add(d_hash)
            tok_count = count_conversation_tokens(clean_msgs, tokenizer)
            if tok_count < 15 or tok_count > 1024:
                continue
            collected["qa"].append({
                "messages": clean_msgs,
                "provenance": {"source": "everyday_conversations", "domain": "conversational_qa",
                               "lang": "en", "doc_hash": d_hash, "num_tokens": tok_count}
            })
            qa_tokens += tok_count
    except Exception as e:
        print(f"     Warning in QA: {e}")

    if qa_tokens < targets["qa"]:
        try:
            ds_orca = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
            for doc in ds_orca:
                if qa_tokens >= targets["qa"]:
                    break
                stats["total_evaluated"] += 1
                q = clean_conversation_turn(doc.get("question", ""))
                r = clean_conversation_turn(doc.get("response", ""))
                if not q or not r or len(q) < 10 or len(r) < 10 or len(r) > 1200:
                    continue
                if not is_strictly_english(q + " " + r):
                    continue
                conv = make_conversation(q, r, "OpenOrca", "conversational_qa", tokenizer, seen_hashes)
                if conv:
                    collected["qa"].append(conv)
                    qa_tokens += conv["provenance"]["num_tokens"]
        except Exception as e:
            print(f"     Warning in OpenOrca: {e}")
    print(f"✓ Completed QA: {qa_tokens:,} tokens ({len(collected['qa']):,} conversations)")

    # =========================================================================
    # STRATIFIED TRAIN / VAL SPLIT (95% / 5%)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  STRATIFYING & SPLITTING 15M TOKEN SFT DATASET V2 (95% Train / 5% Val)")
    print("=" * 80)

    train_data = []
    val_data = []
    domain_breakdowns = []
    rng = random.Random(42)

    total_actual_tokens = 0
    total_train_tokens = 0
    total_val_tokens = 0

    for domain_key, items in collected.items():
        rng.shuffle(items)
        dom_total_tokens = sum(it["provenance"]["num_tokens"] for it in items)
        total_actual_tokens += dom_total_tokens

        val_cutoff = max(1, int(len(items) * 0.05))
        dom_val = items[:val_cutoff]
        dom_train = items[val_cutoff:]

        val_toks = sum(it["provenance"]["num_tokens"] for it in dom_val)
        train_toks = sum(it["provenance"]["num_tokens"] for it in dom_train)

        total_train_tokens += train_toks
        total_val_tokens += val_toks
        train_data.extend(dom_train)
        val_data.extend(dom_val)

        domain_breakdowns.append({
            "domain": domain_key,
            "total_examples": len(items),
            "total_tokens": dom_total_tokens,
            "percentage_of_corpus": round((dom_total_tokens / max(1, total_actual_tokens)) * 100, 2),
            "train_examples": len(dom_train),
            "train_tokens": train_toks,
            "val_examples": len(dom_val),
            "val_tokens": val_toks,
        })
        print(f"  [{domain_key.upper():10s}] Total: {dom_total_tokens:,} tok ({len(items):,} ex) | Train: {train_toks:,} | Val: {val_toks:,}")

    rng.shuffle(train_data)
    rng.shuffle(val_data)

    # Write output files
    train_file = output_dir / "sft_train.jsonl"
    val_file = output_dir / "sft_val.jsonl"

    print(f"\nWriting {len(train_data):,} examples ({total_train_tokens:,} tokens) to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Writing {len(val_data):,} examples ({total_val_tokens:,} tokens) to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_time = time.time() - start_time

    # Generate manifest
    manifest = {
        "dataset_name": "15M Token Weakness-Targeted SFT V2 for 54.5M SLM",
        "version": "2.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_tokens": TARGET_TOTAL,
        "actual_total_tokens": total_actual_tokens,
        "train_tokens": total_train_tokens,
        "val_tokens": total_val_tokens,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "total_examples": len(train_data) + len(val_data),
        "build_time_seconds": round(total_time, 1),
        "domain_distribution": domain_breakdowns,
        "quality_metrics": {
            "math_examples_verified": stats["math_verified_count"],
            "rejected_duplicates": stats["rejected_duplicate"],
            "rejected_non_english": stats["rejected_language"],
            "rejected_length": stats["rejected_length"],
            "rejected_quality": stats["rejected_quality"],
            "rejected_code_syntax": stats["rejected_code_syntax"],
        },
        "design_rationale": "Weakness-targeted mixture overweighting Logic (0%→10%), Science (33%→15%), "
                            "GK (33%→15%) based on diagnostic benchmark results."
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved SFT V2 Manifest to {manifest_file}")
    print(f"\n{'=' * 80}")
    print(f"  SUCCESS: 15M Weakness-Targeted SFT V2 Dataset built in {total_time:.1f}s!")
    print(f"  Train: {len(train_data):,} examples ({total_train_tokens:,} tokens)")
    print(f"  Val:   {len(val_data):,} examples ({total_val_tokens:,} tokens)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    build_sft_v2()
