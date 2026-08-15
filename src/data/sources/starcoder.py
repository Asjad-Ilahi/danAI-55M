"""
StarCoderData / Code streaming dataset adapter per §11.

Streams code from public non-gated datasets (e.g. codeparrot/codeparrot-clean-train)
without requiring Hugging Face authentication tokens.
"""

from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.data.sources.base import BaseSourceAdapter


class StarCoderSourceAdapter(BaseSourceAdapter):
    """Streams code documents from public HF code repositories."""

    def __init__(self, dataset_name: str = "codeparrot/codeparrot-clean-train", subset: str = None, split: str = "train"):
        super().__init__(dataset_name, subset, split)

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        # Try primary non-gated public dataset
        try:
            if self.subset:
                ds = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
            else:
                ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        except Exception:
            # Fallback to public non-gated Python code dataset
            ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=True)

        for idx, item in enumerate(ds):
            text = item.get("content", "") or item.get("text", "") or item.get("code", "")
            if text and len(text.strip()) > 30:
                yield {
                    "id": f"starcoder_{idx}",
                    "text": text,
                    "source": "starcoder",
                    "metadata": {"lang": item.get("lang", "python")},
                }
