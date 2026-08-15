"""
Cosmopedia streaming dataset adapter.
"""

from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.data.sources.base import BaseSourceAdapter


class CosmopediaSourceAdapter(BaseSourceAdapter):
    """Streams documents from HuggingFaceTB/cosmopedia."""

    def __init__(self, dataset_name: str = "HuggingFaceTB/cosmopedia", subset: str = "auto_math_text", split: str = "train"):
        super().__init__(dataset_name, subset, split)

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        ds = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
        for idx, item in enumerate(ds):
            text = item.get("text", "")
            if text:
                yield {
                    "id": f"cosmopedia_{idx}",
                    "text": text,
                    "source": "cosmopedia",
                    "metadata": {"prompt": item.get("prompt", "")},
                }
