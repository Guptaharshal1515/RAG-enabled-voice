import time
from typing import List, Dict, Any
from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.metadata_store import MetadataStore


class Retriever:
    """
    Unified dense retrieval engine combining EmbeddingModel, FAISSIndex, and MetadataStore.
    Tracks and reports sub-millisecond retrieval latencies.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: FAISSIndex,
        metadata_store: MetadataStore
    ):
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.metadata_store = metadata_store

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Execute dense cosine retrieval for a given query.
        Returns:
            {
                "query": query,
                "results": [...],
                "latencies_ms": {
                    "embedding": float,
                    "faiss_search": float,
                    "metadata_lookup": float,
                    "total": float
                }
            }
        """
        t0 = time.perf_counter()

        # 1. Encode query
        t_embed_start = time.perf_counter()
        query_vector = self.embedding_model.encode_query(query)
        t_embed_end = time.perf_counter()

        # 2. FAISS Nearest Neighbor Search
        t_search_start = time.perf_counter()
        scores, indices = self.vector_index.search(query_vector, top_k=top_k)
        t_search_end = time.perf_counter()

        # 3. Metadata retrieval & ranking
        t_meta_start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            meta = self.metadata_store.get(int(idx))
            if meta:
                item = dict(meta)
                item["score"] = float(score)
                results.append(item)
        t_meta_end = time.perf_counter()

        t_total = time.perf_counter() - t0

        return {
            "query": query,
            "results": results,
            "latencies_ms": {
                "embedding": round((t_embed_end - t_embed_start) * 1000, 3),
                "faiss_search": round((t_search_end - t_search_start) * 1000, 3),
                "metadata_lookup": round((t_meta_end - t_meta_start) * 1000, 3),
                "total": round(t_total * 1000, 3)
            }
        }
