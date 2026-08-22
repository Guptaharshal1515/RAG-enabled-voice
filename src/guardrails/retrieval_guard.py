from typing import List, Dict, Any
from src.guardrails.policy import GuardrailDecision


class RetrievalGuard:
    """
    Validates retrieved candidate chunks to filter out off-topic queries
    and insufficient knowledge domain coverage.
    """

    def __init__(
        self,
        min_results: int = 1,
        min_score: float = 0.005 # Calibrated minimal RRF / cosine threshold
    ):
        self.min_results = min_results
        self.min_score = min_score

    def check(self, results: List[Dict[str, Any]]) -> GuardrailDecision:
        if not results:
            return GuardrailDecision(
                passed=False,
                reason="no_relevant_context",
                stage="retrieval_validation"
            )

        if len(results) < self.min_results:
            return GuardrailDecision(
                passed=False,
                reason="no_relevant_context",
                stage="retrieval_validation",
                metadata={"results_count": len(results)}
            )

        # Evaluate top retrieval score
        top_result = results[0]
        score = float(
            top_result.get("reranker_score")
            or top_result.get("rrf_score")
            or top_result.get("score", 0.0)
        )
        dense_score = float(top_result.get("dense_score", score))
        bm25_score = float(top_result.get("bm25_score", 0.0))

        if score < self.min_score:
            return GuardrailDecision(
                passed=False,
                reason="low_retrieval_score",
                stage="retrieval_validation",
                metadata={"top_score": score, "threshold": self.min_score}
            )

        return GuardrailDecision(
            passed=True,
            stage="retrieval_validation",
            metadata={"top_score": score, "results_count": len(results)}
        )
