import json
import re
from typing import List, Dict, Any, Tuple
from src.generation.llm import LLM, generate_with_retry
from src.generation.context import build_context
from src.generation.prompt import build_prompt
from src.generation.schemas import StructuredAnswer


class Generator:
    """
    Grounded Response Generator.
    Coordinates context construction, grounded prompt assembly, LLM generation with retries,
    and structured JSON validation.
    """

    def __init__(
        self,
        llm: LLM,
        max_retries: int = 2,
        timeout_sec: float = 5.0
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec

    def generate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_context_tokens: int = 2500
    ) -> Tuple[StructuredAnswer, List[Dict[str, Any]]]:
        """
        Generate grounded answer from retrieved results.
        Returns (StructuredAnswer, included_context_results).
        """
        context_str, included_results = build_context(
            retrieved_results,
            max_tokens=max_context_tokens
        )

        prompt = build_prompt(query=query, context=context_str)

        raw_output = generate_with_retry(
            llm=self.llm,
            prompt=prompt,
            max_retries=self.max_retries,
            timeout_sec=self.timeout_sec
        )

        structured_answer = self._parse_output(raw_output, included_results)
        return structured_answer, included_results

    def _parse_output(
        self,
        raw_output: str,
        included_results: List[Dict[str, Any]]
    ) -> StructuredAnswer:
        valid_chunk_ids = {r.get("chunk_id") for r in included_results if r.get("chunk_id")}

        # Try to parse as JSON first
        try:
            # Handle possible markdown code blocks ```json ... ```
            cleaned = re.sub(r"^```json\s*", "", raw_output.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            data = json.loads(cleaned)

            answer = str(data.get("answer", "")).strip()
            source_ids = data.get("source_ids", [])
            # Filter to only existing source IDs
            validated_sources = [sid for sid in source_ids if sid in valid_chunk_ids]

            return StructuredAnswer(
                answer=answer,
                source_ids=validated_sources,
                confidence=1.0 if validated_sources else 0.5
            )
        except Exception:
            # Fallback parsing from raw text
            answer = raw_output.strip()
            # Extract bracketed citations [chunk_id]
            found_citations = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", answer)
            validated_sources = [cid for cid in found_citations if cid in valid_chunk_ids]

            return StructuredAnswer(
                answer=answer,
                source_ids=validated_sources,
                confidence=0.8 if validated_sources else 0.4
            )
