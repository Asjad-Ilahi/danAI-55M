"""
Generate SHA-256 & MD5 Cryptographic Hashes for all dataset artifacts in the repository.

Outputs:
1. DATASET_CHECKSUMS_SHA256.json (Machine-readable full manifest with hashes, byte sizes, and timestamps)
2. DATASET_CHECKSUMS.md (Human-readable markdown table of all pretraining and SFT datasets)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DATASET_DIRS = [
    "data",
    "data_100m",
    "data_3b",
    "data_sft",
    "data_sft_v2",
    "data_sft_stage2",
    "data_sft_final",
    "data_sft_knowledge",
    "knowledge_base",
]

ALLOWED_EXTENSIONS = {
    ".bin", ".jsonl", ".json", ".db", ".sqlite3", ".gz", ".zst", ".arrow", ".parquet", ".csv", ".txt"
}


def compute_file_hashes(filepath: Path) -> Dict[str, Any]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size_bytes = filepath.stat().st_size

    # Stream in 1MB chunks
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)

    size_mb = size_bytes / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb / 1024:.2f} GB"

    return {
        "file_path": str(filepath.as_posix()),
        "size_bytes": size_bytes,
        "size_human": size_str,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
        "last_modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(filepath.stat().st_mtime)),
    }


def main():
    root = Path(".")
    records: List[Dict[str, Any]] = []
    total_bytes = 0

    print("=" * 80)
    print("  COMPUTING CRYPTOGRAPHIC SHA-256 CHECKSUMS FOR ALL DATASETS")
    print("=" * 80)

    for d_name in DATASET_DIRS:
        d_path = root / d_name
        if not d_path.exists():
            continue

        print(f"\nScanning directory: {d_name}/ ...")
        for dirpath, _, filenames in os.walk(d_path):
            for fname in sorted(filenames):
                f_path = Path(dirpath) / fname
                if f_path.suffix.lower() in ALLOWED_EXTENSIONS or fname in ("manifest.json", "local_knowledge.db"):
                    rel_path = f_path.relative_to(root)
                    print(f"  • Hashing: {rel_path.as_posix()} ({f_path.stat().st_size / (1024 * 1024):.2f} MB)...")
                    meta = compute_file_hashes(rel_path)
                    records.append(meta)
                    total_bytes += meta["size_bytes"]

    # Write JSON manifest
    json_path = root / "DATASET_CHECKSUMS_SHA256.json"
    manifest_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_files": len(records),
        "total_size_bytes": total_bytes,
        "total_size_human": f"{total_bytes / (1024 * 1024 * 1024):.2f} GB",
        "datasets": records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Write Markdown summary table
    md_path = root / "DATASET_CHECKSUMS.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Provenance & SHA-256 Checksum Manifest\n\n")
        f.write(f"Generated on: `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`  \n")
        f.write(f"Total Tracked Files: **{len(records)}**  \n")
        f.write(f"Total Dataset Footprint: **{total_bytes / (1024 * 1024 * 1024):.2f} GB**\n\n")
        f.write("This document provides immutable cryptographic verification that these exact dataset splits, pretraining tokens, and supervised fine-tuning (SFT) mixtures were generated and used during training.\n\n")
        f.write("| Dataset File Path | Size | SHA-256 Checksum |\n")
        f.write("|:---|:---:|:---|\n")

        for r in records:
            f.write(f"| `{r['file_path']}` | {r['size_human']} | `{r['sha256']}` |\n")

    print("\n" + "=" * 80)
    print(f"  SUCCESS: Computed hashes for {len(records)} dataset files!")
    print(f"  Total Data Processed: {total_bytes / (1024 * 1024 * 1024):.2f} GB")
    print(f"  JSON Manifest: {json_path.as_posix()}")
    print(f"  Markdown Manifest: {md_path.as_posix()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
