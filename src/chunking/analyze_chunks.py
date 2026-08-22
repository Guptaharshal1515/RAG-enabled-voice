import sys
from collections import Counter
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def analyze_chunks(chunks: List[Dict[str, Any]]):
    """
    Computes and displays chunking analytics including strategy breakdown,
    token statistics, and language distributions.
    """
    if not chunks:
        print("No chunks to analyze.")
        return

    strategy_counts = Counter(chunk["chunk_strategy"] for chunk in chunks)
    token_counts = [chunk["token_count"] for chunk in chunks]
    lang_counts = Counter(chunk.get("language", "unknown") for chunk in chunks)

    print("\n" + "=" * 55)
    print("           CHUNKING ANALYTICS & PROFILE")
    print("=" * 55)

    print(f"Total Chunks: {len(chunks)}")

    print("\nStrategy Distribution:")
    for strategy, count in strategy_counts.items():
        pct = (count / len(chunks)) * 100
        bar = "#" * int(pct / 4)
        print(f"  {strategy:<28} : {count:>5} ({pct:>5.1f}%) {bar}")

    print("\nLanguage Distribution:")
    for lang, count in lang_counts.items():
        pct = (count / len(chunks)) * 100
        print(f"  {lang:<15} : {count:>5} ({pct:>5.1f}%)")

    if token_counts:
        sorted_tokens = sorted(token_counts)
        p50 = sorted_tokens[len(sorted_tokens) // 2]
        p95 = sorted_tokens[int(len(sorted_tokens) * 0.95)]

        print("\nToken Statistics per Chunk:")
        print(f"  Minimum tokens : {min(token_counts)}")
        print(f"  P50 (Median)   : {p50}")
        print(f"  Average tokens : {sum(token_counts) / len(token_counts):.2f}")
        print(f"  P95 tokens     : {p95}")
        print(f"  Maximum tokens : {max(token_counts)}")
    print("=" * 55 + "\n")
