"""
Cross-dataset exact SHA-256 hash deduplicator per §15.
"""

import hashlib
import re
from typing import Set


class CrossDatasetExactDedup:
    """Tracks SHA-256 text hashes across all datasets."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_duplicate(self, text: str) -> bool:
        norm = self._normalize(text)
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
