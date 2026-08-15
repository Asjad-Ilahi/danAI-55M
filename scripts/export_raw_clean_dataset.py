"""
Script to save raw readable clean dataset files into data/raw/
and remove old corrupt/repetitive dataset files from data/.
"""

import json
import os
import shutil
from pathlib import Path
from datasets import load_dataset


from src.data.cleaner import clean_text


def main():
    print("=" * 60)
    print("CLEANING OLD DATASET FILES & EXPORTING PERFECT READABLE CLEAN CORPUS")
    print("=" * 60)

    # 1. Remove old dataset files
    old_files = [
        Path("data/raw/tokenizer_sample.jsonl"),
        Path("data/cleaned/cleaned_docs.jsonl"),
    ]
    for old_file in old_files:
        if old_file.exists():
            print(f"Removing old file: {old_file} ({old_file.stat().st_size / (1024**2):.1f} MB)")
            old_file.unlink()

    # Clean old processed directory
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\nDownloading and saving perfectly cleaned text files to data/raw/ ...")

    # 2. Export readable FineWeb-Edu clean documents
    print("Exporting data/raw/fineweb_edu_clean.jsonl ...")
    ds_edu = iter(load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True))
    edu_path = raw_dir / "fineweb_edu_clean.jsonl"
    with open(edu_path, "w", encoding="utf-8") as f:
        for i in range(15000):
            item = next(ds_edu)
            text = clean_text(item["text"])
            if len(text) > 30:
                f.write(json.dumps({"id": f"edu_{i}", "text": text}) + "\n")
    print(f"  Saved {edu_path.name}: {edu_path.stat().st_size / (1024**2):.1f} MB")

    # 3. Export readable TinyStories clean documents
    print("Exporting data/raw/tinystories_clean.jsonl ...")
    ds_stories = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
    stories_path = raw_dir / "tinystories_clean.jsonl"
    with open(stories_path, "w", encoding="utf-8") as f:
        for i in range(18000):
            item = next(ds_stories)
            text = clean_text(item["text"])
            if len(text) > 30:
                f.write(json.dumps({"id": f"story_{i}", "text": text}) + "\n")
    print(f"  Saved {stories_path.name}: {stories_path.stat().st_size / (1024**2):.1f} MB")

    # 4. Export readable Cosmopedia v2 clean documents (Reasoning, Q&A, Code)
    print("Exporting data/raw/cosmopedia_v2_clean.jsonl ...")
    ds_cosmo = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True))
    cosmo_path = raw_dir / "cosmopedia_v2_clean.jsonl"
    with open(cosmo_path, "w", encoding="utf-8") as f:
        for i in range(8000):
            item = next(ds_cosmo)
            text = clean_text(item["text"])
            if len(text) > 30:
                f.write(json.dumps({"id": f"cosmo_{i}", "text": text}) + "\n")
    print(f"  Saved {cosmo_path.name}: {cosmo_path.stat().st_size / (1024**2):.1f} MB")

    print("\n" + "=" * 60)
    print("RAW READABLE DATASET FILES EXPORTED SUCCESSFULLY TO data/raw/!")
    print("=" * 60)


if __name__ == "__main__":
    main()
