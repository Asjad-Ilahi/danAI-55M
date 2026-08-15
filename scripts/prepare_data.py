"""
Data cleaning and preprocessing pipeline script per §7.

Reads raw docs from data/raw/ → cleans → normalizes → quality filters → exact dedup →
saves to data/cleaned/ and writes preprocessing report to data/reports/cleaning_report.json
"""

import argparse
import json
import os
from pathlib import Path

from src.data.cleaner import TextCleaner
from src.data.deduplicator import ExactDeduplicator
from src.data.datasets.text_dataset import TextDatasetAdapter


def prepare_data(raw_dir: Path, cleaned_dir: Path, reports_dir: Path) -> dict:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    adapter = TextDatasetAdapter(raw_dir)
    cleaner = TextCleaner()
    dedup = ExactDeduplicator()

    total_docs = 0
    cleaned_docs = 0
    stats = {
        "total_input_docs": 0,
        "valid_docs": 0,
        "rejected_empty": 0,
        "rejected_too_short": 0,
        "rejected_too_long": 0,
        "rejected_low_word_count": 0,
        "rejected_repetitive": 0,
        "rejected_duplicates": 0,
        "input_total_chars": 0,
        "cleaned_total_chars": 0,
    }

    output_file = cleaned_dir / "cleaned_docs.jsonl"
    with open(output_file, "w", encoding="utf-8") as out_f:
        for doc in adapter.stream_documents():
            total_docs += 1
            stats["total_input_docs"] += 1
            text = doc["text"]
            stats["input_total_chars"] += len(text)

            is_valid, reason = cleaner.filter_doc(text)
            if not is_valid:
                stats[f"rejected_{reason}"] = stats.get(f"rejected_{reason}", 0) + 1
                continue

            cleaned_text = cleaner.clean_text(text)
            if dedup.is_duplicate(cleaned_text):
                stats["rejected_duplicates"] += 1
                continue

            stats["valid_docs"] += 1
            stats["cleaned_total_chars"] += len(cleaned_text)

            record = {
                "id": doc["id"],
                "text": cleaned_text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_file = reports_dir / "cleaning_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING REPORT")
    print("=" * 60)
    print(f"  Input Documents:       {stats['total_input_docs']:,}")
    print(f"  Valid Documents:       {stats['valid_docs']:,}")
    print(f"  Rejected Duplicates:   {stats['rejected_duplicates']:,}")
    print(f"  Rejected Short/Empty:  {stats['rejected_too_short'] + stats['rejected_empty']:,}")
    print(f"  Rejected Repetitive:   {stats['rejected_repetitive']:,}")
    print(f"  Input Chars:           {stats['input_total_chars']:,}")
    print(f"  Cleaned Chars:         {stats['cleaned_total_chars']:,}")
    print(f"  Report Saved:          {report_file}")
    print("=" * 60 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare and clean raw dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--cleaned-dir", type=str, default="data/cleaned")
    parser.add_argument("--reports-dir", type=str, default="data/reports")
    args = parser.parse_args()

    prepare_data(Path(args.raw_dir), Path(args.cleaned_dir), Path(args.reports_dir))


if __name__ == "__main__":
    main()
