"""
OpenWebMath streaming dataset adapter.
"""

from typing import Iterator, Dict, Any
from datasets import load_dataset
from src.data.sources.base import BaseSourceAdapter


class OpenWebMathSourceAdapter(BaseSourceAdapter):
    """Streams documents from open-web-math/open-web-math."""

    def __init__(self, dataset_name: str = "open-web-math/open-web-math", subset: str = None, split: str = "train"):
        super().__init__(dataset_name, subset, split)

    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        for idx, item in enumerate(ds):
            text = item.get("text", "")
            if text:
                yield {
                    "id": f"openwebmath_{idx}",
                    "text": text,
                    "source": "openwebmath",
                    "metadata": {"url": item.get("url", "")},
                }
