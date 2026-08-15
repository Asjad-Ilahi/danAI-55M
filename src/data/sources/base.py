"""
Base streaming source adapter interface.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any


class BaseSourceAdapter(ABC):
    """Abstract base class for HF streaming dataset sources."""

    def __init__(self, dataset_name: str, subset: str = None, split: str = "train"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split

    @abstractmethod
    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        """
        Yields raw document dicts:
        {
            "id": str,
            "text": str,
            "source": str,
            "metadata": dict
        }
        """
        pass
