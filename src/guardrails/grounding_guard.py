import re
from typing import List, Dict, Any, Set, Tuple
from src.guardrails.policy import GuardrailDecision


class GroundingGuard:
    """
    Multi-Tier Grounding & Citation Verifier.
    Ensures generated answers are strictly factual, contain valid citations,
    and are supported by the retrieved evidence.
    """

    def __init__(
        self,
        min_grounding_score: float = 0.50,
        strict_citations: bool = True
    ):
        self.min_grounding_score = min_grounding_score
        self.strict_citations = strict_citations

    def validate_citations(
        self,
        cited_sources: List[str],
        retrieved_results: List[Dict[str, Any]]
    ) -> Tuple[bool, Set[str], Set[str]]:
        """
        Verifies that cited sources exist in the retrieved context.
        Returns: (all_valid, valid_set, hallucinated_set)
        """
        available_ids = {
            r.get("chunk_id")
            for r in retrieved_results
            if r.get("chunk_id")
        }

        # Also allow matching by document_id
        for r in retrieved_results:
            doc = r.get("document", r)
            if "document_id" in doc:
                available_ids.add(doc["document_id"])

        cited_set = set(cited_sources)
        hallucinated = cited_set - available_ids
        valid = cited_set.intersection(available_ids)

        is_valid = len(hallucinated) == 0
        return is_valid, valid, hallucinated

    def verify_grounding(
        self,
        answer: str,
        retrieved_results: List[Dict[str, Any]],
        cited_sources: List[str]
    ) -> GuardrailDecision:
        """
        Verify that claims in the answer are grounded in the retrieved passages.
        """
        if not answer or not answer.strip():
            return GuardrailDecision(
                passed=False,
                reason="empty_answer",
                stage="grounding_verification"
            )

        # 1. Check if the answer is a valid refusal
        refusal_phrases = [
            "do not have enough information",
            "could not find relevant information",
            "not enough information in the provided sources",
            "i cannot answer"
        ]
        if any(p in answer.lower() for p in refusal_phrases):
            return GuardrailDecision(
                passed=True,
                stage="grounding_verification",
                metadata={"is_refusal": True, "grounding_score": 1.0}
            )

        # 2. Validate Citations
        citation_valid, valid_cites, fake_cites = self.validate_citations(
            cited_sources,
            retrieved_results
        )

        if self.strict_citations and fake_cites:
            return GuardrailDecision(
                passed=False,
                reason="ungrounded_answer",
                stage="grounding_verification",
                metadata={"fake_citations": list(fake_cites)}
            )

        # 3. Extract text from retrieved results
        corpus_texts = []
        for r in retrieved_results:
            doc = r.get("document", r)
            corpus_texts.append(str(doc.get("text", "")))
        full_corpus = " ".join(corpus_texts).lower()
        corpus_tokens = set(re.findall(r"\w+", full_corpus, flags=re.UNICODE))

        # 4. Split answer into claims / sentences
        clean_answer = re.sub(r"\[[a-zA-Z0-9_\-]+\]", "", answer) # remove citations
        claims = [
            c.strip()
            for c in re.split(r'(?<=[.!?।॥])\s+', clean_answer)
            if c.strip()
        ]

        if not claims:
            return GuardrailDecision(
                passed=True,
                stage="grounding_verification",
                metadata={"grounding_score": 1.0}
            )

        supported_claims = 0
        claim_details = []

        for claim in claims:
            claim_words = set(re.findall(r"\w+", claim.lower(), flags=re.UNICODE))
            if not claim_words:
                continue

            overlap = claim_words.intersection(corpus_tokens)
            overlap_ratio = len(overlap) / len(claim_words)
            is_supported = overlap_ratio >= 0.40

            if is_supported:
                supported_claims += 1

            claim_details.append({
                "claim": claim,
                "overlap_ratio": round(overlap_ratio, 2),
                "supported": is_supported
            })

        grounding_score = supported_claims / max(len(claims), 1)
        passed = grounding_score >= self.min_grounding_score

        return GuardrailDecision(
            passed=passed,
            reason=None if passed else "ungrounded_answer",
            stage="grounding_verification",
            metadata={
                "grounding_score": round(grounding_score, 2),
                "supported_claims": supported_claims,
                "total_claims": len(claims),
                "claims": claim_details,
                "citations_valid": citation_valid
            }
        )
