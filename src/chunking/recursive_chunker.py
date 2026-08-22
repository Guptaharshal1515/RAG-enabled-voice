from typing import List
from .token_counter import estimate_tokens


SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "। ",
    " "
]


def recursive_chunk(
    text: str,
    max_tokens: int = 500
) -> List[str]:
    """
    Recursive hierarchical splitting prioritizing paragraphs, sentences, clauses, then words.
    """
    text = text.strip()
    if not text:
        return []

    if estimate_tokens(text) <= max_tokens:
        return [text]

    return _split_recursively(
        text=text,
        separators=SEPARATORS,
        max_tokens=max_tokens
    )


def _split_recursively(
    text: str,
    separators: List[str],
    max_tokens: int
) -> List[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        words = text.split()
        return [
            " ".join(words[i:i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]

    separator = separators[0]
    parts = text.split(separator)

    chunks = []
    current_parts = []

    for part in parts:
        candidate = (separator.join(current_parts + [part])).strip()
        if estimate_tokens(candidate) <= max_tokens:
            current_parts.append(part)
        else:
            if current_parts:
                current_text = separator.join(current_parts).strip()
                chunks.extend(
                    _split_recursively(
                        current_text,
                        separators[1:],
                        max_tokens
                    )
                )
            current_parts = [part]

    if current_parts:
        remaining = separator.join(current_parts).strip()
        chunks.extend(
            _split_recursively(
                remaining,
                separators[1:],
                max_tokens
            )
        )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]
