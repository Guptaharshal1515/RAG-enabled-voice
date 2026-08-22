import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from src.retrieval.search import Retriever
from src.retrieval.bm25_index import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """
    Unified Multilingual Hybrid Retrieval Engine.
    Combines Dense Vector Retrieval (FAISS) + Lexical Keyword Matching (BM25)
    with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_index: BM25Index,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: int = 20,
        bm25_candidates: int = 20,
        fusion_candidates: int = 30,
        enable_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full Hybrid retrieve -> fuse -> rerank pipeline.
        """
        t_total_start = time.perf_counter()

        # 1 & 2. Concurrent Dense & BM25 Retrieval
        t_ret_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(
                self.dense_retriever.search, query, dense_candidates
            )
            future_bm25 = executor.submit(
                self.bm25_index.search, query, bm25_candidates
            )
            dense_response = future_dense.result()
            bm25_results_raw = future_bm25.result()
        t_ret_end = time.perf_counter()
        retrieval_concurrent_ms = (t_ret_end - t_ret_start) * 1000

        dense_results = []
        dense_score_map = {}
        for r in dense_response["results"]:
            cid = r["chunk_id"]
            dense_score_map[cid] = float(r["score"])
            dense_results.append({"document": r, "score": r["score"], "chunk_id": cid})

        bm25_results = []
        bm25_score_map = {}
        for r in bm25_results_raw:
            doc = r["document"]
            cid = doc.get("chunk_id", "")
            bm25_score_map[cid] = float(r["score"])
            bm25_results.append({"document": doc, "score": r["score"], "chunk_id": cid})

        # 3. Reciprocal Rank Fusion (RRF)
        t_fusion_start = time.perf_counter()
        fused_candidates = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=60,
            top_k=fusion_candidates
        )
        t_fusion_end = time.perf_counter()

        # 4. Cross-Encoder Reranking
        t_rerank_start = time.perf_counter()
        if enable_reranking and self.reranker:
            final_chunks = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_k=top_k
            )
            retrieval_method = "hybrid_rrf_reranked"
        else:
            final_chunks = [
                {
                    "chunk_id": c["chunk_id"],
                    "rrf_score": c["rrf_score"],
                    "reranker_score": None,
                    "document": c["document"]
                }
                for c in fused_candidates[:top_k]
            ]
            retrieval_method = "hybrid_rrf"
        t_rerank_end = time.perf_counter()

        t_total_end = time.perf_counter()

        # Format output
        formatted_results = []
        for c in final_chunks:
            cid = c.get("chunk_id", "")
            doc = c["document"].copy()
            doc["reranker_score"] = c.get("reranker_score")
            doc["rrf_score"] = c.get("rrf_score")
            doc["dense_score"] = dense_score_map.get(cid, 0.0)
            doc["bm25_score"] = bm25_score_map.get(cid, 0.0)
            formatted_results.append(doc)

        return {
            "query": query,
            "retrieval_method": retrieval_method,
            "results": formatted_results,
            "latencies_ms": {
                "retrieval_concurrent": round(retrieval_concurrent_ms, 3),
                "dense_retrieval": round(retrieval_concurrent_ms, 3),
                "bm25_retrieval": round(retrieval_concurrent_ms, 3),
                "rrf_fusion": round((t_fusion_end - t_fusion_start) * 1000, 3),
                "reranking": round((t_rerank_end - t_rerank_start) * 1000, 3) if enable_reranking else 0.0,
                "total": round((t_total_end - t_total_start) * 1000, 3)
            }
        }
