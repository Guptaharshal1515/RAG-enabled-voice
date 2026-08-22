from typing import Dict, Any, Optional
from dataclasses import dataclass, field


REFUSALS: Dict[str, str] = {
    "empty_query": "Please provide a question.",
    "query_too_long": "Your question exceeds the maximum allowable length.",
    "unsafe_query": "I cannot help with that request as it violates safety guidelines.",
    "no_relevant_context": "I could not find relevant information in the knowledge base.",
    "low_retrieval_score": "I could not find sufficiently relevant information in the knowledge base.",
    "ungrounded_answer": "I couldn't verify the answer against the retrieved evidence.",
    "generation_error": "An error occurred while synthesizing the response.",
    "prompt_injection_detected": "The request contains invalid instructions and cannot be processed."
}


def get_refusal_message(reason: str) -> str:
    """
    Get centralized standard user-facing refusal text for a guardrail trigger.
    """
    return REFUSALS.get(reason, "Knowledge base context retrieved.")


@dataclass
class GuardrailDecision:
    passed: bool
    reason: Optional[str] = None
    stage: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
