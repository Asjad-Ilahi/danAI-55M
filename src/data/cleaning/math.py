"""
Math & technical writing cleaner per §12.

Preserves LaTeX formulas, mathematical equations, and derivations.
"""

import re
import unicodedata
from typing import Tuple


class MathCleaner:
    """Specialized cleaner for mathematical and technical content."""

    def __init__(self, min_doc_chars: int = 50, max_doc_chars: int = 1_500_000):
        self.min_doc_chars = min_doc_chars
        self.max_doc_chars = max_doc_chars

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    def filter(self, text: str) -> Tuple[bool, str]:
        cleaned = self.clean(text)
        if not cleaned:
            return False, "empty"

        char_len = len(cleaned)
        if char_len < self.min_doc_chars:
            return False, "too_short"
        if char_len > self.max_doc_chars:
            return False, "too_long"

        return True, "valid"
