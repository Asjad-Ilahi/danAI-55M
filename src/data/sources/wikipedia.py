"""
Wikipedia streaming dataset adapter.
"""

from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.data.sources.base import BaseSourceAdapter


class WikipediaSourceAdapter(BaseSourceAdapter):
    """Streams English articles from wikimedia/wikipedia."""

    def __init__(self, dataset_name: str = "wikimedia/wikipedia", subset: str = "20231101.en", split: str = "train"):
        super().__init__(dataset_name, subset, split)

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        ds = load_dataset(self.dataset_name, name=self.subset, split=self.split, streaming=True)
        for idx, item in enumerate(ds):
            text = item.get("text", "")
            if text:
                yield {
                    "id": f"wiki_{item.get('id', idx)}",
                    "text": text,
                    "source": "wikipedia",
                    "metadata": {"title": item.get("title", "")},
                }
