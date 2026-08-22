import re
from typing import List
from .token_counter import estimate_tokens


def split_sentences(text: str) -> List[str]:
    """
    Language-aware sentence splitting supporting English (.!?) and Indic (।॥) terminators.
    """
    sentences = re.split(
        r'(?<=[.!?।॥])\s+',
        text.strip()
    )
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_chunk(
    text: str,
    max_tokens: int = 400,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Build chunks preserving sentence boundaries with sliding sentence overlap.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if current_chunk and (current_tokens + sentence_tokens > max_tokens):
            chunks.append(" ".join(current_chunk))
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap.copy()
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
