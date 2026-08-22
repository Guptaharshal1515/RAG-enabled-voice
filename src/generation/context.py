from typing import List, Dict, Any, Tuple
from src.chunking.token_counter import estimate_tokens

MAX_CONTEXT_TOKENS = 2500


def build_context(
    results: List[Dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format retrieved results into a structured, bounded context string.
    Ensures total context does not exceed the token budget.
    """
    context_blocks = []
    included_results = []
    total_tokens = 0

    for i, result in enumerate(results, start=1):
        chunk_id = result.get("chunk_id", f"chunk_{i}")
        doc = result.get("document", result)
        text = doc.get("text", "")
        lang = doc.get("language", "unknown")

        block = (
            f"[SOURCE {i}]\n"
            f"chunk_id: {chunk_id}\n"
            f"language: {lang}\n"
            f"text: {text}\n"
        )
        block_tokens = estimate_tokens(block)

        if total_tokens + block_tokens > max_tokens and context_blocks:
            # Respect context budget limit
            break

        context_blocks.append(block)
        included_results.append(result)
        total_tokens += block_tokens

    formatted_context = "\n\n".join(context_blocks)
    return formatted_context, included_results
