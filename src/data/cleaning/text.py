"""
General text cleaner and quality filter per §6, §7, §16.
"""

import re
import unicodedata
from typing import Tuple, Dict, Any


class TextCleaner:
    """Normalizes and quality-filters general web text."""

    def __init__(
        self,
        min_doc_chars: int = 50,
        max_doc_chars: int = 1_000_000,
        min_word_count: int = 5,
        min_alpha_ratio: float = 0.50,
        max_symbol_ratio: float = 0.25,
    ):
        self.min_doc_chars = min_doc_chars
        self.max_doc_chars = max_doc_chars
        self.min_word_count = min_word_count
        self.min_alpha_ratio = min_alpha_ratio
        self.max_symbol_ratio = max_symbol_ratio

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # 1. Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # 2. Line ending normalization
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip null bytes & control chars (except tab and newline)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 4. Remove HTML tags while preserving text inside
        text = re.sub(r"<(?:p|br|div|span|header|footer|nav|a|td|tr|table)\s*/?>", "\n", text, flags=re.IGNORECASE)

        # 5. Trim trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 6. Collapse excessive blank lines (max 2 consecutive blank lines)
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

        words = cleaned.split()
        if len(words) < self.min_word_count:
            return False, "low_word_count"

        # Character distribution checks
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        alpha_ratio = alpha_chars / max(1, char_len)
        if alpha_ratio < self.min_alpha_ratio:
            return False, "low_alpha_ratio"

        symbol_chars = sum(1 for c in cleaned if not c.isalnum() and not c.isspace())
        symbol_ratio = symbol_chars / max(1, char_len)
        if symbol_ratio > self.max_symbol_ratio:
            return False, "excessive_symbols"

        # Excessive repetition check
        if re.search(r"(.)\1{40,}", cleaned):
            return False, "character_repetition"

        return True, "valid"
