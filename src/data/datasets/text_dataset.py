"""
Plain text dataset adapter.
"""

from pathlib import Path
from typing import Iterator, Dict, Any
from src.data.datasets.base import BaseDatasetAdapter


class TextDatasetAdapter(BaseDatasetAdapter):
    """Adapter for plain text / markdown / code files."""

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        extensions = ["*.txt", "*.md", "*.py", "*.jsonl"]
        for ext in extensions:
            for file_path in self.data_dir.glob(f"**/{ext}"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if text.strip():
                            yield {
                                "text": text,
                                "id": str(file_path.relative_to(self.data_dir)),
                                "metadata": {"source_file": str(file_path)},
                            }
                except Exception:
                    continue
