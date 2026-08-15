"""
Exact document deduplicator per §7.

Uses hash-based (SHA-256) exact deduplication to remove duplicate documents.
"""

import hashlib
import re
from typing import Set, Tuple


class ExactDeduplicator:
    """Exact document deduplicator based on normalized hash matching."""

    def __init__(self):
        self.seen_hashes: Set[str] = set()

    def _normalize_for_hash(self, text: str) -> str:
        """Normalize text by lowercasing and stripping non-alphanumeric characters for hashing."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        norm_text = self._normalize_for_hash(text)
        return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text has been seen before; if not, add hash to seen set."""
        doc_hash = self.compute_hash(text)
        if doc_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(doc_hash)
        return False

    def reset(self) -> None:
        """Clear hash memory."""
        self.seen_hashes.clear()
