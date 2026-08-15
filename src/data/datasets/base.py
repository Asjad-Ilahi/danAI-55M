"""
Base dataset adapter interface per §8.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Dict, Any


class BaseDatasetAdapter(ABC):
    """Abstract interface for dataset adapters."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def stream_documents(self) -> Iterator[Dict[str, Any]]:
        """
        Yield document dicts: {'text': str, 'id': str, 'metadata': dict}
        """
        pass
