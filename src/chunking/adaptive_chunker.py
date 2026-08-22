from typing import Dict, Any, List
from .token_counter import estimate_tokens
from .sentence_chunker import sentence_chunk
from .recursive_chunker import recursive_chunk


class AdaptiveChunker:
    """
    Adaptive Multi-Strategy Chunking Router.
    Routes incoming text to the optimal chunking strategy based on length and structure:
    - SHORT (<= 200 tokens): Keep whole document intact to avoid context fragmentation.
    - MEDIUM (201 - 800 tokens): Sentence-aware chunking with overlap.
    - LONG (801 - 2000 tokens): Recursive hierarchical chunking.
    - VERY_LONG (> 2000 tokens): Semantic fallback / hook for Phase 3 embeddings.
    """

    def __init__(
        self,
        short_threshold: int = 200,
        medium_threshold: int = 800,
        long_threshold: int = 2000
    ):
        self.short_threshold = short_threshold
        self.medium_threshold = medium_threshold
        self.long_threshold = long_threshold

    def chunk(self, text: str) -> Dict[str, Any]:
        text = text.strip() if text else ""
        token_count = estimate_tokens(text)

        if token_count == 0:
            return {
                "strategy": "empty",
                "chunks": [],
                "token_count": 0
            }

        # Strategy 1: Short documents kept intact
        if token_count <= self.short_threshold:
            return {
                "strategy": "whole_document",
                "chunks": [text],
                "token_count": token_count
            }

        # Strategy 2: Sentence-aware chunking
        elif token_count <= self.medium_threshold:
            return {
                "strategy": "sentence_aware",
                "chunks": sentence_chunk(
                    text,
                    max_tokens=400,
                    overlap_sentences=1
                ),
                "token_count": token_count
            }

        # Strategy 3: Recursive hierarchical chunking
        elif token_count <= self.long_threshold:
            return {
                "strategy": "recursive",
                "chunks": recursive_chunk(
                    text,
                    max_tokens=500
                ),
                "token_count": token_count
            }

        # Strategy 4: Semantic chunking hook (fallback to recursive in Phase 2)
        else:
            return {
                "strategy": "semantic_fallback_recursive",
                "chunks": recursive_chunk(
                    text,
                    max_tokens=500
                ),
                "token_count": token_count
            }
