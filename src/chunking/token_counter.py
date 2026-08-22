import re


def estimate_tokens(text: str) -> int:
    """
    Lightweight, language-aware token estimation using Unicode words and symbols.
    Works consistently across Indic scripts and English.
    """
    if not text or not text.strip():
        return 0

    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)
