"""
Custom dataset adapter for organization text or JSONL files.
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any
from src.data.datasets.base import BaseDatasetAdapter


class CustomDatasetAdapter(BaseDatasetAdapter):
    """Adapter for custom structured dataset files (JSONL, CSV)."""

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        for jsonl_path in self.data_dir.glob("**/*.jsonl"):
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        try:
                            obj = json.loads(line)
                            text = obj.get("text") or obj.get("content") or ""
                            if text.strip():
                                yield {
                                    "text": text,
                                    "id": f"{jsonl_path.name}_{idx}",
                                    "metadata": obj.get("metadata", {}),
                                }
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
