"""
Code preservation cleaner per §11.

Preserves code syntax, indentation, comments, docstrings, and structure.
Does NOT run generic whitespace normalization.
"""

import re
import unicodedata
from typing import Tuple


class CodeCleaner:
    """Specialized cleaner for programming code."""

    def __init__(self, min_doc_chars: int = 30, max_doc_chars: int = 2_000_000):
        self.min_doc_chars = min_doc_chars
        self.max_doc_chars = max_doc_chars

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Unicode NFC normalization & line endings
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null bytes
        text = text.replace("\x00", "")

        # Strip trailing trailing-line whitespace while preserving leading indentation!
        lines = [line.rstrip(" \t\r") for line in text.split("\n")]
        return "\n".join(lines).strip()

    def filter(self, text: str) -> Tuple[bool, str]:
        cleaned = self.clean(text)
        if not cleaned:
            return False, "empty"

        char_len = len(cleaned)
        if char_len < self.min_doc_chars:
            return False, "too_short"
        if char_len > self.max_doc_chars:
            return False, "too_long"

        # Check for minified JavaScript or single extremely long lines (> 5,000 chars)
        lines = cleaned.split("\n")
        max_line_len = max(len(line) for line in lines)
        if max_line_len > 5000:
            return False, "minified_or_generated_line"

        return True, "valid"
