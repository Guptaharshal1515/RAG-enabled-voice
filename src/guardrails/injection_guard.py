import re
from typing import List, Optional
from src.guardrails.policy import GuardrailDecision


class InjectionGuard:
    """
    Detects and neutralizes prompt-injection and jailbreak attacks in user queries
    and retrieved documents.
    """

    PATTERNS: List[str] = [
        r"ignore\s+(all\s+|the\s+)?(previous\s+|prior\s+)?instructions",
        r"reveal\s+(the\s+|your\s+)?system\s+prompt",
        r"you\s+are\s+now\s+(an\s+unfiltered|in\s+developer\s+mode|dan)",
        r"system\s+override",
        r"disregard\s+(all\s+)?safety\s+guidelines",
        r"print\s+(your\s+)?internal\s+rules",
        r"act\s+as\s+an\s+unrestricted",
    ]

    def has_injection(self, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.PATTERNS)

    def check_query(self, query: str) -> GuardrailDecision:
        """
        Validates whether user query is a direct jailbreak attempt.
        """
        if self.has_injection(query):
            return GuardrailDecision(
                passed=False,
                reason="prompt_injection_detected",
                stage="injection_guard"
            )
        return GuardrailDecision(passed=True, stage="injection_guard")

    def sanitize_evidence(self, text: str) -> str:
        """
        Strips injection commands from evidence to prevent indirect injection.
        """
        if not text:
            return ""
        cleaned = text
        for p in self.PATTERNS:
            cleaned = re.sub(p, "[REDACTED_INSTRUCTION]", cleaned, flags=re.IGNORECASE)
        return cleaned
