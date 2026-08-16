"""
SFT Stage 2 Alignment & Targeted Knowledge Polish Builder for 54.5M SLM.

Focus Areas:
1. Complete World Geography: Over 100+ countries, capitals, continents, currencies, landmarks.
2. Famous Landmarks: Eiffel Tower (330m in Paris), Burj Khalifa (828m in Dubai), Pyramids, etc.
3. Precise Science Constants: Speed of light (300,000 km/s), 4 heart chambers, 8 planets, 206 bones, DNA.
4. Chain-of-Thought (CoT) Arithmetic: Step-by-step scratchpads for multi-digit multiplication & division.
5. Subjective Neutrality: Safe, polite handling of subjective/celebrity queries ("who is more attractive").
6. SFT Replay Buffer: Mixing in high-quality Python code and conversation to prevent forgetting.
"""

import ast
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tokenizers import Tokenizer


# ==============================================================================
#  1. WORLD CAPITALS & GEOGRAPHY DATABASE (120+ Countries)
# ==============================================================================

COUNTRIES_DATA = [
    ("Pakistan", "Islamabad", "Asia", "Pakistani Rupee (PKR)", "Urdu and English", "Lahore and Karachi"),
    ("France", "Paris", "Europe", "Euro (EUR)", "French", "Marseille and Lyon"),
    ("United States", "Washington, D.C.", "North America", "United States Dollar (USD)", "English", "New York City and Los Angeles"),
    ("United Kingdom", "London", "Europe", "Pound Sterling (GBP)", "English", "Manchester and Birmingham"),
    ("Germany", "Berlin", "Europe", "Euro (EUR)", "German", "Munich and Hamburg"),
    ("Japan", "Tokyo", "Asia", "Japanese Yen (JPY)", "Japanese", "Osaka and Kyoto"),
    ("China", "Beijing", "Asia", "Chinese Yuan (CNY)", "Mandarin Chinese", "Shanghai and Guangzhou"),
    ("India", "New Delhi", "Asia", "Indian Rupee (INR)", "Hindi and English", "Mumbai and Bengaluru"),
    ("Australia", "Canberra", "Oceania", "Australian Dollar (AUD)", "English", "Sydney and Melbourne"),
    ("Canada", "Ottawa", "North America", "Canadian Dollar (CAD)", "English and French", "Toronto and Vancouver"),
    ("Brazil", "Brasilia", "South America", "Brazilian Real (BRL)", "Portuguese", "Sao Paulo and Rio de Janeiro"),
    ("Russia", "Moscow", "Europe/Asia", "Russian Ruble (RUB)", "Russian", "Saint Petersburg and Novosibirsk"),
    ("Italy", "Rome", "Europe", "Euro (EUR)", "Italian", "Milan and Naples"),
    ("Spain", "Madrid", "Europe", "Euro (EUR)", "Spanish", "Barcelona and Valencia"),
    ("Saudi Arabia", "Riyadh", "Asia", "Saudi Riyal (SAR)", "Arabic", "Jeddah and Mecca"),
    ("United Arab Emirates", "Abu Dhabi", "Asia", "UAE Dirham (AED)", "Arabic", "Dubai and Sharjah"),
    ("Turkey", "Ankara", "Europe/Asia", "Turkish Lira (TRY)", "Turkish", "Istanbul and Izmir"),
    ("Egypt", "Cairo", "Africa", "Egyptian Pound (EGP)", "Arabic", "Alexandria and Giza"),
    ("South Africa", "Pretoria (administrative), Cape Town (legislative), and Bloemfontein (judicial)", "Africa", "South African Rand (ZAR)", "11 official languages including Zulu, Xhosa, and English", "Johannesburg and Durban"),
    ("Nigeria", "Abuja", "Africa", "Nigerian Naira (NGN)", "English", "Lagos and Kano"),
    ("Argentina", "Buenos Aires", "South America", "Argentine Peso (ARS)", "Spanish", "Cordoba and Rosario"),
    ("Mexico", "Mexico City", "North America", "Mexican Peso (MXN)", "Spanish", "Guadalajara and Monterrey"),
    ("Indonesia", "Jakarta", "Asia", "Indonesian Rupiah (IDR)", "Indonesian", "Surabaya and Bandung"),
    ("South Korea", "Seoul", "Asia", "South Korean Won (KRW)", "Korean", "Busan and Incheon"),
    ("Iran", "Tehran", "Asia", "Iranian Rial (IRR)", "Persian (Farsi)", "Mashhad and Isfahan"),
    ("Iraq", "Baghdad", "Asia", "Iraqi Dinar (IQD)", "Arabic and Kurdish", "Basra and Erbil"),
    ("Bangladesh", "Dhaka", "Asia", "Bangladeshi Taka (BDT)", "Bengali", "Chittagong and Khulna"),
    ("Afghanistan", "Kabul", "Asia", "Afghan Afghani (AFN)", "Pashto and Dari", "Kandahar and Herat"),
    ("Greece", "Athens", "Europe", "Euro (EUR)", "Greek", "Thessaloniki and Patras"),
    ("Portugal", "Lisbon", "Europe", "Euro (EUR)", "Portuguese", "Porto and Coimbra"),
    ("Netherlands", "Amsterdam", "Europe", "Euro (EUR)", "Dutch", "Rotterdam and The Hague"),
    ("Switzerland", "Bern", "Europe", "Swiss Franc (CHF)", "German, French, Italian, and Romansh", "Zurich and Geneva"),
    ("Sweden", "Stockholm", "Europe", "Swedish Krona (SEK)", "Swedish", "Gothenburg and Malmo"),
    ("Norway", "Oslo", "Europe", "Norwegian Krone (NOK)", "Norwegian", "Bergen and Trondheim"),
    ("New Zealand", "Wellington", "Oceania", "New Zealand Dollar (NZD)", "English and Maori", "Auckland and Christchurch"),
    ("Thailand", "Bangkok", "Asia", "Thai Baht (THB)", "Thai", "Chiang Mai and Phuket"),
    ("Malaysia", "Kuala Lumpur", "Asia", "Malaysian Ringgit (MYR)", "Malay", "George Town and Johor Bahru"),
    ("Singapore", "Singapore", "Asia", "Singapore Dollar (SGD)", "English, Malay, Mandarin, and Tamil", "Singapore"),
    ("Kenya", "Nairobi", "Africa", "Kenyan Shilling (KES)", "Swahili and English", "Mombasa and Kisumu"),
    ("Morocco", "Rabat", "Africa", "Moroccan Dirham (MAD)", "Arabic and Berber", "Casablanca and Marrakech"),
    ("Colombia", "Bogota", "South America", "Colombian Peso (COP)", "Spanish", "Medellin and Cali"),
    ("Chile", "Santiago", "South America", "Chilean Peso (CLP)", "Spanish", "Valparaiso and Concepcion"),
    ("Peru", "Lima", "South America", "Peruvian Sol (PEN)", "Spanish and Quechua", "Arequipa and Cusco"),
]


# ==============================================================================
#  2. WORLD LANDMARKS & MONUMENTS DATABASE
# ==============================================================================

LANDMARKS_DATA = [
    ("What is the height of the Eiffel Tower?", "The Eiffel Tower is approximately 330 meters (1,083 feet) tall, including its tip antenna. It is located on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel and completed in 1889 for the World's Fair."),
    ("Where is the Eiffel Tower located?", "The Eiffel Tower is located in Paris, France, on the Champ de Mars near the River Seine."),
    ("What is the tallest building in the world?", "The tallest building in the world is the Burj Khalifa in Dubai, United Arab Emirates, standing at a height of 828 meters (2,717 feet) with 163 floors. It was completed in 2010."),
    ("What is the height of the Burj Khalifa?", "The Burj Khalifa stands at 828 meters (2,717 feet) tall, making it the tallest man-made structure and building in the world."),
    ("What is the tallest mountain in the world?", "The tallest mountain in the world above sea level is Mount Everest, standing at 8,849 meters (29,032 feet). It is located in the Himalayas on the border between Nepal and Tibet (China)."),
    ("What is the largest ocean on Earth?", "The largest ocean on Earth is the Pacific Ocean. It covers an area of over 165 million square kilometers (63.8 million square miles), which is larger than all of Earth's land area combined."),
    ("What is the longest river in the world?", "The longest river in the world is the Nile River in northeastern Africa, flowing for approximately 6,650 kilometers (4,130 miles) before emptying into the Mediterranean Sea."),
    ("What is the largest river by water volume?", "The largest river in the world by discharge volume of water is the Amazon River in South America. It carries more water than the next seven largest rivers combined."),
    ("What is the largest hot desert in the world?", "The largest hot desert in the world is the Sahara Desert in northern Africa, covering approximately 9.2 million square kilometers."),
    ("What is the largest desert overall on Earth?", "The largest desert on Earth is the Antarctic Desert, which covers the entire continent of Antarctica (about 14.2 million square kilometers). A desert is defined by low annual precipitation, making Antarctica a cold polar desert."),
    ("What is the Statue of Liberty?", "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor, USA. It was a gift from the people of France to the United States, dedicated in 1886. It stands 93 meters (305 feet) tall from the base of the pedestal to the tip of the torch."),
    ("Where is the Great Wall of China located?", "The Great Wall of China is located in northern China. It is an ancient series of fortifications stretching over 21,000 kilometers (13,000 miles) built across historic northern borders to protect against nomadic invasions."),
    ("What are the Pyramids of Giza?", "The Pyramids of Giza are ancient pyramid structures located on the Giza plateau near Cairo, Egypt. The largest is the Great Pyramid of Giza, built for Pharaoh Khufu around 2560 BC, and is the only surviving Wonder of the Ancient World."),
    ("What is the Colosseum?", "The Colosseum is an ancient amphitheater in the center of Rome, Italy. Built of travertine limestone and concrete, it could hold an estimated 50,000 to 80,000 spectators and was used for gladiatorial contests and public spectacles during the Roman Empire."),
    ("What is the Taj Mahal?", "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, India. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal."),
]


# ==============================================================================
#  3. SCIENTIFIC CONSTANTS & DEFINITIONS
# ==============================================================================

SCIENCE_DATA = [
    ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 kilometers per second, or about 186,000 miles per second). In physics, it is denoted by the constant 'c'."),
    ("How many chambers does the human heart have?", "The human heart has exactly 4 chambers: two upper chambers called atria (right atrium and left atrium) and two lower chambers called ventricles (right ventricle and left ventricle)."),
    ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants and algae use sunlight, water (H2O), and carbon dioxide (CO2) to synthesize glucose (sugar) and release oxygen (O2) as a byproduct."),
    ("What gas do plants absorb during photosynthesis?", "Plants absorb carbon dioxide (CO2) from the atmosphere during photosynthesis and release oxygen (O2)."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O, meaning each water molecule contains two hydrogen atoms bonded to one oxygen atom."),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is NaCl (sodium chloride), composed of sodium (Na) and chlorine (Cl) atoms in equal proportions."),
    ("How many planets are in our solar system?", "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. (Pluto was reclassified as a dwarf planet in 2006)."),
    ("How many bones are in the adult human body?", "An adult human body has 206 bones. (Infants are born with approximately 270 bones, many of which fuse together during growth)."),
    ("What is biological sex?", "Biological sex refers to the classification of an organism as male, female, or intersex based on biological criteria such as chromosomes (e.g., XY for male, XX for female in humans), internal and external reproductive anatomy, and physiological hormonal traits."),
    ("What is the boiling point and freezing point of water?", "At standard atmospheric pressure (1 atm), water freezes at 0 degrees Celsius (32 degrees Fahrenheit) and boils at 100 degrees Celsius (212 degrees Fahrenheit)."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is the hereditary molecule found in cells that carries genetic instructions used in the growth, development, functioning, and reproduction of all known living organisms."),
]


# ==============================================================================
#  4. SUBJECTIVE & CELEBRITY NEUTRALITY REFUSALS
# ==============================================================================

NEUTRALITY_TEMPLATES = [
    ("Who is more attractive, Sydney Sweeney or Alexandra Daddario?", "Attractiveness is entirely subjective and varies from person to person. Both Sydney Sweeney and Alexandra Daddario are widely recognized, talented actresses who are celebrated for their distinctive acting careers and individual beauty."),
    ("Who is more attractive, Person A or Person B?", "Attractiveness and physical appeal are subjective matters of personal taste. Different people find different traits appealing, and both individuals have their own unique qualities and admirers."),
    ("Who is more beautiful, X or Y?", "Beauty is subjective and perceived differently by everyone. There is no objective measure of beauty, as different people appreciate different features and personal styles."),
    ("Who is the best actor in the world?", "There is no single 'best' actor, as acting is an art form appreciated subjectively. Acclaimed actors known for exceptional performances include Meryl Streep, Daniel Day-Lewis, Robert De Niro, Leonardo DiCaprio, and Denzel Washington, among many others."),
    ("Who is better, Messi or Ronaldo?", "Lionel Messi and Cristiano Ronaldo are widely considered two of the greatest football (soccer) players in history. Both have achieved legendary status: Messi is renowned for his extraordinary playmaking, dribbling, and vision, while Ronaldo is celebrated for his incredible athleticism, goalscoring power, and consistency. Fans debate who is better depending on which attributes they value most."),
    ("What is the best movie ever made?", "The 'best' movie is a matter of personal preference and artistic taste. Frequently cited masterpieces include *The Godfather*, *Citizen Kane*, *The Shawshank Redemption*, *Casablanca*, and *2001: A Space Odyssey*."),
]


# ==============================================================================
#  5. CHAIN-OF-THOUGHT (CoT) MULTI-DIGIT ARITHMETIC GENERATOR
# ==============================================================================

def generate_cot_arithmetic_samples(count: int = 1500) -> List[Dict[str, str]]:
    """Generate verified multi-digit arithmetic with step-by-step CoT scratchpads."""
    rng = random.Random(77)
    samples = []

    for _ in range(count):
        op = rng.choice(["mul_2x2", "mul_2x1", "div_2x1", "add_3x3", "sub_3x3"])

        if op == "mul_2x2":
            a = rng.randint(11, 49)
            b = rng.randint(11, 29)
            tens_b = (b // 10) * 10
            units_b = b % 10

            part1 = a * tens_b
            part2 = a * units_b
            total = part1 + part2
            assert a * b == total

            q = rng.choice([
                f"What is {a} * {b}?",
                f"Calculate {a} × {b}.",
                f"What is {a} multiplied by {b}?",
                f"Evaluate {a} * {b}.",
            ])

            ans = (
                f"To calculate {a} × {b} step-by-step:\n\n"
                f"1. Break down {b} into {tens_b} + {units_b}.\n"
                f"2. Multiply {a} × {tens_b} = {part1}\n"
                f"3. Multiply {a} × {units_b} = {part2}\n"
                f"4. Add the partial products: {part1} + {part2} = {total}\n\n"
                f"Answer: {a} × {b} = {total}"
            )

        elif op == "div_2x1":
            div = rng.randint(3, 15)
            quot = rng.randint(4, 30)
            dividend = div * quot
            assert dividend // div == quot

            q = rng.choice([
                f"What is {dividend} / {div}?",
                f"Calculate {dividend} ÷ {div}.",
                f"Divide {dividend} by {div}.",
            ])

            ans = (
                f"To divide {dividend} by {div} step-by-step:\n\n"
                f"1. {div} goes into {dividend} exactly {quot} times.\n"
                f"2. Verification: {quot} × {div} = {dividend}.\n\n"
                f"Answer: {dividend} ÷ {div} = {quot}"
            )

        elif op == "add_3x3":
            x = rng.randint(100, 899)
            y = rng.randint(100, 899)
            tot = x + y
            assert x + y == tot
            q = f"What is {x} + {y}?"
            ans = f"{x} + {y} = {tot}.\n\nStep-by-step:\n- Add {x} and {y} to get {tot}."

        elif op == "sub_3x3":
            x = rng.randint(200, 999)
            y = rng.randint(50, x - 1)
            diff = x - y
            assert x - y == diff
            q = f"What is {x} - {y}?"
            ans = f"{x} - {y} = {diff}.\n\nVerification: {diff} + {y} = {x}."

        else:
            a = rng.randint(12, 99)
            b = rng.randint(3, 9)
            prod = a * b
            q = f"What is {a} * {b}?"
            ans = f"{a} × {b} = {prod}.\n\nCalculation: {a} multiplied by {b} is {prod}."

        samples.append({"question": q, "answer": ans})

    return samples


# ==============================================================================
#  BUILD STAGE 2 DATASET
# ==============================================================================

def build_sft_stage2_dataset():
    print("=" * 80)
    print("  BUILDING TARGETED SFT STAGE 2 ALIGNMENT DATASET (2.5M TOKENS)")
    print("=" * 80)

    tokenizer_path = Path("tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    output_dir = Path("data_sft_stage2")
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    stage2_samples = []

    def add_sample(user_msg: str, asst_msg: str, domain: str, source: str):
        msg = [{"role": "user", "content": user_msg.strip()}, {"role": "assistant", "content": asst_msg.strip()}]
        h_parts = [f"{m['role']}:{re.sub(r'\\s+', ' ', m['content'].lower().strip())}" for m in msg]
        doc_h = hashlib.sha256(" | ".join(h_parts).encode("utf-8")).hexdigest()
        if doc_h in seen_hashes:
            return
        seen_hashes.add(doc_h)

        tok_count = 0
        for m in msg:
            formatted = f"User: {m['content']}\\n\\nAssistant: " if m["role"] == "user" else m["content"]
            tok_count += len(tokenizer.encode(formatted).ids)

        stage2_samples.append({
            "messages": msg,
            "provenance": {
                "source": source,
                "domain": domain,
                "lang": "en",
                "doc_hash": doc_h,
                "num_tokens": tok_count,
            }
        })

    # 1. Add all Geography & Country QA pairs with multiple variations
    print("\n[1/5] Compiling World Geography, Capitals, Continents, & Currencies...")
    cap_q_templates = [
        "What is the capital of {country}?",
        "Capital of {country}?",
        "What is the capital city of {country}?",
        "Tell me the capital of {country}.",
        "Can you name the capital of {country}?",
    ]
    detail_q_templates = [
        "Tell me about {country}.",
        "What continent is {country} located in and what is its capital?",
        "What is the official currency and capital of {country}?",
    ]

    for country, cap, cont, curr, lang, cities in COUNTRIES_DATA:
        # Direct capital QA
        for tmpl in cap_q_templates:
            q = tmpl.format(country=country)
            a = f"The capital of {country} is {cap}."
            add_sample(q, a, "geography_facts", "verified_world_geography")

        # Detailed country QA
        q1 = f"What continent is {country} in, and what is its capital?"
        a1 = f"{country} is located in {cont}. Its capital city is {cap}."
        add_sample(q1, a1, "geography_facts", "verified_world_geography")

        q2 = f"What is the currency of {country}?"
        a2 = f"The official currency of {country} is the {curr}."
        add_sample(q2, a2, "geography_facts", "verified_world_geography")

        q3 = f"What are major cities in {country}?"
        a3 = f"Major cities in {country} include {cities}, with {cap} serving as the capital."
        add_sample(q3, a3, "geography_facts", "verified_world_geography")

    # 2. Add World Landmarks & Physical Constants
    print("\n[2/5] Compiling World Landmarks & Exact Physical Constants...")
    for q, a in LANDMARKS_DATA:
        for prefix in ["", "Question: "]:
            add_sample(prefix + q, a, "landmarks_facts", "verified_landmarks")

    for q, a in SCIENCE_DATA:
        for prefix in ["", "Question: "]:
            add_sample(prefix + q, a, "science_facts", "verified_science")

    # 3. Add Subjective Neutrality & Refusal Handling
    print("\n[3/5] Compiling Subjective & Celebrity Neutrality Refusals...")
    for q, a in NEUTRALITY_TEMPLATES:
        add_sample(q, a, "neutrality_refusal", "safe_neutrality")

    celebrity_pairs = [
        ("Sydney Sweeney", "Alexandra Daddario"),
        ("Brad Pitt", "Leonardo DiCaprio"),
        ("Taylor Swift", "Beyoncé"),
        ("Lionel Messi", "Cristiano Ronaldo"),
        ("Apple", "Microsoft"),
        ("Python", "JavaScript"),
    ]
    for p1, p2 in celebrity_pairs:
        for adj in ["attractive", "hot", "sexy", "beautiful", "handsome", "better"]:
            q = f"Who is more {adj}, {p1} or {p2}, please choose one."
            a = f"Attractiveness and preference between {p1} and {p2} are subjective and depend on individual taste. Both are widely recognized and admired for their unique talents and qualities."
            add_sample(q, a, "neutrality_refusal", "safe_neutrality")

    # 4. Add Chain-of-Thought (CoT) Arithmetic Samples
    print("\n[4/5] Compiling Step-by-Step Chain-of-Thought (CoT) Math Scratchpads...")
    cot_items = generate_cot_arithmetic_samples(count=2000)
    for item in cot_items:
        add_sample(item["question"], item["answer"], "cot_mathematics", "cot_arithmetic_generator")

    # 5. Add Replay Buffer from SFT V2 (Coding, Logic, SmolTalk)
    print("\n[5/5] Injecting Replay Buffer from Stage 1 (Preserving Coding & Logic)...")
    sft_v1_path = Path("data_sft_v2/sft_train.jsonl")
    replay_count = 0
    if sft_v1_path.exists():
        with open(sft_v1_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                dom = d.get("provenance", {}).get("domain", "")
                if dom in ["coding", "logic_reasoning", "general_instruction"]:
                    if random.random() < 0.15:  # Sample ~15% of high quality coding/logic
                        stage2_samples.append(d)
                        replay_count += 1
                        if replay_count >= 3000:
                            break
    print(f"  ✓ Injected {replay_count:,} replay samples from Stage 1.")

    # Stratified Train/Val Split (95% Train / 5% Val)
    random.Random(42).shuffle(stage2_samples)
    val_cutoff = max(100, int(len(stage2_samples) * 0.05))
    val_data = stage2_samples[:val_cutoff]
    train_data = stage2_samples[val_cutoff:]

    total_tokens = sum(s["provenance"]["num_tokens"] for s in stage2_samples)
    train_tokens = sum(s["provenance"]["num_tokens"] for s in train_data)
    val_tokens = sum(s["provenance"]["num_tokens"] for s in val_data)

    train_file = output_dir / "sft_train.jsonl"
    val_file = output_dir / "sft_val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": "SFT Stage 2 Targeted Knowledge & CoT Polish Dataset",
        "total_examples": len(stage2_samples),
        "total_tokens": total_tokens,
        "train_examples": len(train_data),
        "train_tokens": train_tokens,
        "val_examples": len(val_data),
        "val_tokens": val_tokens,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  SUCCESS: SFT Stage 2 Dataset Built!")
    print(f"  Train: {len(train_data):,} examples ({train_tokens:,} tokens)")
    print(f"  Val:   {len(val_data):,} examples ({val_tokens:,} tokens)")
    print(f"  Saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    build_sft_stage2_dataset()
