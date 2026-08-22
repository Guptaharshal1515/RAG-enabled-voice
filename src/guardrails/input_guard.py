import re
from typing import Tuple, Optional
from src.guardrails.policy import GuardrailDecision


class InputGuard:
    """
    Validates user query string prior to retrieval:
    - Empty / whitespace check
    - Maximum query length
    - Unsafe input patterns
    """

    def __init__(self, max_query_length: int = 1000):
        self.max_query_length = max_query_length
        self.unsafe_patterns = [
            r"<script.*?>.*?</script>",
            r"(DROP\s+TABLE|DELETE\s+FROM|UNION\s+SELECT)",
            r"exec\s*\(\s*['\"]",
        ]

    def check(self, query: str) -> GuardrailDecision:
        if query is None or not str(query).strip():
            return GuardrailDecision(
                passed=False,
                reason="empty_query",
                stage="input_validation"
            )

        clean_query = str(query).strip()

        if len(clean_query) > self.max_query_length:
            return GuardrailDecision(
                passed=False,
                reason="query_too_long",
                stage="input_validation",
                metadata={"length": len(clean_query), "max": self.max_query_length}
            )

        for pattern in self.unsafe_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                return GuardrailDecision(
                    passed=False,
                    reason="unsafe_query",
                    stage="input_validation"
                )

        return GuardrailDecision(
            passed=True,
            stage="input_validation",
            metadata={"query_length": len(clean_query)}
        )
