import time
import uuid
import logging
from typing import Dict, Any, Optional, List

from src.generation.schemas import Source, RAGResponse, StructuredAnswer
from src.generation.generator import Generator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.guardrails.input_guard import InputGuard
from src.guardrails.retrieval_guard import RetrievalGuard
from src.guardrails.injection_guard import InjectionGuard
from src.guardrails.grounding_guard import GroundingGuard
from src.guardrails.policy import get_refusal_message, GuardrailDecision
from src.observability.cache import LRUCache
from src.observability.timer import Timer

logger = logging.getLogger("RAGPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Req:%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class RAGPipeline:
    """
    Guarded Multilingual Voice-Ready RAG Execution Harness.
    Integrates multi-tier guardrails, concurrent retrieval, in-memory caching,
    and sub-millisecond precision latency instrumentation.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        max_context_tokens: int = 2500,
        max_query_chars: int = 1000,
        min_retrieval_score: float = 0.005,
        min_grounding_score: float = 0.50,
        max_regeneration_attempts: int = 1,
        enable_cache: bool = True
    ):
        self.retriever = retriever
        self.generator = generator
        self.max_context_tokens = max_context_tokens
        self.max_query_chars = max_query_chars
        self.max_regeneration_attempts = max_regeneration_attempts
        self.enable_cache = enable_cache

        # In-memory query response cache
        self.cache = LRUCache(max_size=2000, ttl_sec=3600.0)

        # Initialize Guardrails
        self.input_guard = InputGuard(max_query_length=max_query_chars)
        self.injection_guard = InjectionGuard()
        self.retrieval_guard = RetrievalGuard(min_score=min_retrieval_score)
        self.grounding_guard = GroundingGuard(min_grounding_score=min_grounding_score)

        self.request_logs: List[Dict[str, Any]] = []

    def run(
        self,
        query: str,
        top_k: int = 5,
        enable_reranking: bool = False
    ) -> RAGResponse:
        """
        Execute full guarded RAG pipeline for an incoming query.
        """
        request_id = str(uuid.uuid4())[:8]
        t_total_start = time.perf_counter()
        guardrail_trail: Dict[str, Any] = {}

        # ----------------------------------------------------
        # 1. Input Guard
        # ----------------------------------------------------
        input_decision = self.input_guard.check(query)
        guardrail_trail["input_guard"] = input_decision.passed

        if not input_decision.passed:
            refusal_text = get_refusal_message(input_decision.reason or "empty_query")
            return self._build_response(
                request_id=request_id,
                query=query or "",
                answer=refusal_text,
                sources=[],
                refusal=True,
                error=input_decision.reason,
                t_total_start=t_total_start,
                guardrail_trail=guardrail_trail
            )

        clean_query = str(query).strip()

        # Cache check
        if self.enable_cache:
            cached_resp: Optional[RAGResponse] = self.cache.get(clean_query)
            if cached_resp is not None:
                total_ms = (time.perf_counter() - t_total_start) * 1000
                cached_dict = cached_resp.to_dict()
                return RAGResponse(
                    request_id=request_id,
                    query=clean_query,
                    answer=cached_resp.answer,
                    sources=cached_resp.sources,
                    retrieval_method=cached_resp.retrieval_method + "_cached",
                    grounded=cached_resp.grounded,
                    refusal=cached_resp.refusal,
                    error=cached_resp.error,
                    latencies_ms={
                        "retrieval": 0.0,
                        "generation": 0.0,
                        "cache_lookup": round(total_ms, 2),
                        "total": round(total_ms, 2)
                    }
                )

        # 1b. Direct Injection Query Check
        injection_decision = self.injection_guard.check_query(clean_query)
        guardrail_trail["injection_guard_query"] = injection_decision.passed
        if not injection_decision.passed:
            refusal_text = get_refusal_message("prompt_injection_detected")
            return self._build_response(
                request_id=request_id,
                query=clean_query,
                answer=refusal_text,
                sources=[],
                refusal=True,
                error="prompt_injection_detected",
                t_total_start=t_total_start,
                guardrail_trail=guardrail_trail
            )

        # ----------------------------------------------------
        # 2. Retrieval Stage
        # ----------------------------------------------------
        t_retrieval_start = time.perf_counter()
        retrieval_response = self.retriever.search(
            query=clean_query,
            top_k=top_k,
            enable_reranking=enable_reranking
        )
        retrieved_results = retrieval_response.get("results", [])
        retrieval_method = retrieval_response.get("retrieval_method", "hybrid_rrf")
        t_retrieval_end = time.perf_counter()
        retrieval_ms = (t_retrieval_end - t_retrieval_start) * 1000

        # Sanitize any indirect injection patterns inside evidence
        for r in retrieved_results:
            doc = r.get("document", r)
            if "text" in doc:
                doc["text"] = self.injection_guard.sanitize_evidence(doc["text"])

        # ----------------------------------------------------
        # 3. Retrieval Guard (Off-topic & Confidence Filter)
        # ----------------------------------------------------
        retrieval_decision = self.retrieval_guard.check(retrieved_results)
        guardrail_trail["retrieval_guard"] = retrieval_decision.passed
        guardrail_trail["retrieval_score"] = retrieval_decision.metadata.get("top_score", 0.0)

        if not retrieval_decision.passed:
            refusal_text = get_refusal_message(retrieval_decision.reason or "no_relevant_context")
            return self._build_response(
                request_id=request_id,
                query=clean_query,
                answer=refusal_text,
                sources=[],
                retrieval_method=retrieval_method,
                refusal=True,
                error=retrieval_decision.reason,
                t_total_start=t_total_start,
                retrieval_ms=retrieval_ms,
                guardrail_trail=guardrail_trail
            )

        # ----------------------------------------------------
        # 4. LLM Generation + Grounding Verification Loop
        # ----------------------------------------------------
        t_gen_start = time.perf_counter()
        attempt = 0
        final_structured_ans: Optional[StructuredAnswer] = None
        final_included_context = retrieved_results
        grounding_passed = False
        final_grounding_decision: Optional[GuardrailDecision] = None

        while attempt <= self.max_regeneration_attempts:
            attempt += 1
            try:
                structured_ans, included_context = self.generator.generate(
                    query=clean_query,
                    retrieved_results=retrieved_results,
                    max_context_tokens=self.max_context_tokens
                )
                final_structured_ans = structured_ans
                final_included_context = included_context

                # Grounding & Citation Check
                grounding_decision = self.grounding_guard.verify_grounding(
                    answer=structured_ans.answer,
                    retrieved_results=included_context,
                    cited_sources=structured_ans.source_ids
                )
                final_grounding_decision = grounding_decision

                if grounding_decision.passed:
                    grounding_passed = True
                    break

            except Exception as gen_err:
                logger.error(f"Generation error on attempt {attempt}: {gen_err}")
                break

        t_gen_end = time.perf_counter()
        generation_ms = (t_gen_end - t_gen_start) * 1000

        guardrail_trail["grounding_guard"] = grounding_passed
        guardrail_trail["regeneration_attempts"] = attempt

        # Handle Grounding Failure
        if not grounding_passed or not final_structured_ans:
            refusal_text = get_refusal_message("ungrounded_answer")
            return self._build_response(
                request_id=request_id,
                query=clean_query,
                answer=refusal_text,
                sources=[],
                retrieval_method=retrieval_method,
                refusal=True,
                error="ungrounded_answer",
                t_total_start=t_total_start,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                guardrail_trail=guardrail_trail
            )

        raw_answer = final_structured_ans.answer
        cited_ids = final_structured_ans.source_ids
        is_refusal = bool(final_grounding_decision and final_grounding_decision.metadata.get("is_refusal", False))

        # Build Source objects
        sources_list: List[Source] = []
        for r in final_included_context:
            cid = r.get("chunk_id", "")
            doc = r.get("document", r)
            doc_id = doc.get("document_id", cid)
            score = float(r.get("reranker_score") or r.get("rrf_score") or r.get("score", 0.0))
            sources_list.append(Source(
                chunk_id=cid,
                document_id=doc_id,
                score=round(score, 4),
                text=doc.get("text"),
                language=doc.get("language")
            ))

        return self._build_response(
            request_id=request_id,
            query=clean_query,
            answer=raw_answer,
            sources=sources_list,
            retrieval_method=retrieval_method,
            grounded=bool(cited_ids and not is_refusal),
            refusal=is_refusal,
            error=None,
            t_total_start=t_total_start,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            guardrail_trail=guardrail_trail
        )

    def _build_response(
        self,
        request_id: str,
        query: str,
        answer: str,
        sources: List[Source],
        refusal: bool = False,
        grounded: bool = False,
        retrieval_method: str = "hybrid_rrf",
        error: Optional[str] = None,
        t_total_start: float = 0.0,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        guardrail_trail: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        total_ms = (time.perf_counter() - t_total_start) * 1000

        resp = RAGResponse(
            request_id=request_id,
            query=query,
            answer=answer,
            sources=sources,
            retrieval_method=retrieval_method,
            grounded=grounded,
            refusal=refusal,
            error=error,
            latencies_ms={
                "retrieval": round(retrieval_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(total_ms, 2)
            }
        )

        if self.enable_cache and not error and not refusal and query:
            self.cache.put(query, resp)

        log_entry = {
            "request_id": request_id,
            "query": query,
            "refusal": refusal,
            "grounded": grounded,
            "error": error,
            "guardrail_trail": guardrail_trail or {},
            "num_sources": len(sources),
            "latencies_ms": resp.latencies_ms
        }
        self.request_logs.append(log_entry)
        return resp
