"""
Comprehensive text cleaning and normalization pipeline for SLM training datasets.

Removes boilerplate headers/footers, divider lines (----), mid-sentence line breaks,
smart quotes, hyphenated line wraps (en-\nvironment -> environmental), and excess whitespace.
"""

import html
import re
import unicodedata


def strip_html_tags(text: str) -> str:
    """Strip and clean HTML tags (<p>, <pre>, <code>, <br>) and unescape HTML entities."""
    if not text or ("<" not in text and "&" not in text):
        return text
    text = html.unescape(text)
    text = re.sub(r'<pre\s*><code\s*>', '\n```\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</code>\s*</pre>', '\n```\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:p|div|h[1-6]|li|blockquote|br)[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(?:code|pre|span|a|strong|em|b|i|ul|ol|table|tr|td|th)[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>\n]+>', '', text)
    return text


def clean_code(text: str) -> str:
    """
    Clean code snippets (Python, HTML, JS, CSS) preserving exact syntax & indentation.
    """
    if not text:
        return ""
    
    # 1. Normalize Unicode (NFKC) & unescape HTML
    text = unicodedata.normalize("NFKC", text)
    if "&lt;" in text or "&gt;" in text or "&amp;" in text:
        text = html.unescape(text)
    
    # 2. Convert tabs to 4 spaces
    text = text.replace("\t", "    ")
    
    # 3. Standardize quotes & clean non-breaking space
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    
    # 4. Strip trailing whitespace per line without touching leading indentation
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # 5. Remove horizontal divider line bars (e.g. # ----------------)
    text = re.sub(r'\n[ \t]*[#/-=]{5,}[ \t]*(?=\n|$)', '\n', text)
    
    # 6. Collapse 3+ consecutive newlines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def clean_prose(text: str) -> str:
    """
    Clean prose text (General Knowledge, Stories, Math word problems, Q&A) normalizing formatting.
    """
    if not text:
        return ""
    
    # 0. Strip HTML tags & unescape entities
    text = strip_html_tags(text)

    # 1. Normalize Unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Strip GSM8k / Math internal reasoning scratchpad tags like <<8*2=16>>
    text = re.sub(r'<<[^>\n]+>>', '', text)
    
    # 3. Smart quotes, apostrophes, dashes, bullets, and stray backslashes
    text = (
        text.replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2014", " - ")
            .replace("\u2013", " - ")
            .replace("\xa0", " ")
            .replace("\u2022", "* ")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\\n", "\n")  # convert literal string '\n' to actual newline
    )
    
    # 4. Remove excessive markdown bold/italic tags (**word** -> word) for cleaner training text
    text = re.sub(r'\*{2,}(.*?)\*{2,}', r'\1', text)
    
    # 5. Strip horizontal divider lines and repetitive bars
    text = re.sub(r'\n\s*[-=_*#]{3,}\s*(?=\n|$)', '\n', text)
    text = re.sub(r'^[-=_*#]{3,}\s*\n', '', text)
    
    # 6. Remove PDF/Web boilerplate headers & footers
    boilerplate_patterns = [
        r'Click here to download.*?\n',
        r'Click here to print.*?\n',
        r'No\.\s*\d+;\s*Updated.*?\n',
        r'All rights reserved.*?\n',
        r'Terms of Use.*?\n',
        r'Privacy Policy.*?\n',
        r'Cookie Policy.*?\n',
        r'Table of Contents.*?\n',
    ]
    for bp in boilerplate_patterns:
        text = re.sub(bp, '', text, flags=re.IGNORECASE)
    
    # 7. Fix hyphenated word breaks at line ends ('en-\nvironment' -> 'environment')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 8. Normalize paragraph & mid-sentence line breaks
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        lines = p.split('\n')
        # Preserve lists, bullet points, headers
        if any(line.strip().startswith(('-', '*', '1.', '2.', '3.', '4.', '5.', '#')) for line in lines):
            clean_p = '\n'.join(l.strip() for l in lines if l.strip())
        else:
            # Collapse mid-sentence newlines into single spaces
            clean_p = ' '.join(l.strip() for l in lines if l.strip())
        if clean_p:
            cleaned_paragraphs.append(clean_p)
    
    text = '\n\n'.join(cleaned_paragraphs)
    
    # 9. Collapse redundant horizontal whitespace & 3+ consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def is_valid_quality(text: str, is_code: bool = False) -> bool:
    """
    Check if a document meets strict quality standards.
    Filters out malformed, extremely short (<50 chars / <15 words), control-character corrupted,
    or highly repetitive documents.
    """
    if not text or len(text) < 50:
        return False
    
    # 1. Non-printable control character check
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in ('\n', '\t', '\r'))
    if control_chars / len(text) > 0.05:
        return False
        
    words = text.split()
    if len(words) < 15:
        return False
        
    # 2. High n-gram repetition check (4-grams)
    if len(words) >= 30:
        from collections import Counter
        four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        counts = Counter(four_grams)
        most_common_count = counts.most_common(1)[0][1] if counts else 1
        if (most_common_count * 4) / len(words) > 0.35:
            return False
            
    return True


def clean_text(text: str, is_code: bool = False) -> str:
    """
    Apply strict, high-quality cleaning transformations.
    
    Args:
        text: Raw input text.
        is_code: If True, preserves exact line breaks, indentation, and structure.
    """
    if is_code:
        return clean_code(text)
    else:
        return clean_prose(text)



