"""
FineWeb streaming dataset adapter.
"""

from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.data.sources.base import BaseSourceAdapter


class FineWebSourceAdapter(BaseSourceAdapter):
    """Streams documents from HuggingFaceFW/fineweb."""

    def __init__(self, dataset_name: str = "HuggingFaceFW/fineweb", subset: str = "sample-10BT", split: str = "train"):
        super().__init__(dataset_name, subset, split)

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        ds = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
        for idx, item in enumerate(ds):
            text = item.get("text", "")
            if text:
                yield {
                    "id": f"fineweb_{item.get('id', idx)}",
                    "text": text,
                    "source": "fineweb",
                    "metadata": {"language": item.get("language", "en")},
                }
