from typing import List, Dict, Any, Optional
import time
import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker that scores (query, passage) pairs jointly
    to capture fine-grained semantic relevance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a query.
        """
        if not candidates:
            return []

        pairs = [
            (query, str(c.get("document", c).get("text", "")))
            for c in candidates
        ]

        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Normalize with sigmoid if scores are logits
        if isinstance(raw_scores, np.ndarray):
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
        else:
            scores = [1.0 / (1.0 + np.exp(-s)) for s in raw_scores]

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            doc = cand.get("document", cand).copy()
            scored_candidates.append({
                "chunk_id": cand.get("chunk_id", doc.get("chunk_id")),
                "reranker_score": round(float(score), 4),
                "rrf_score": cand.get("rrf_score", 0.0),
                "document": doc
            })

        # Sort descending by reranker_score
        reranked = sorted(
            scored_candidates,
            key=lambda x: x["reranker_score"],
            reverse=True
        )[:top_k]

        return reranked
