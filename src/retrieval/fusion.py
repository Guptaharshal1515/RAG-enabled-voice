from collections import defaultdict
from typing import List, Dict, Any


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 30
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to combine rank-ordered candidate lists
    from heterogeneous retrieval systems (Dense + BM25).

    Formula:
        RRF_Score(d) = sum_{system} ( 1 / (k + rank_{system}(d)) )
    """
    scores = defaultdict(float)
    documents = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc = result.get("document", result)
            chunk_id = doc.get("chunk_id", str(result.get("index", rank)))

            scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in documents:
                documents[chunk_id] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    output = []
    for chunk_id, score in ranked[:top_k]:
        output.append({
            "chunk_id": chunk_id,
            "rrf_score": score,
            "document": documents[chunk_id]
        })

    return output
