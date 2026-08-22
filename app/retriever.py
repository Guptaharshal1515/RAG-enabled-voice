import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever
from src.retrieval.hybrid_retriever import HybridRetriever

# Lazy singleton holders
_EMBEDDER: Optional[EmbeddingModel] = None
_FAISS_INDEX: Optional[FAISSIndex] = None
_BM25_INDEX: Optional[BM25Index] = None
_METADATA_STORE: Optional[MetadataStore] = None
_RETRIEVER: Optional[HybridRetriever] = None


@dataclass
class SearchResult:
    query: str
    results: List[Dict[str, Any]]
    embed_ms: float
    search_ms: float
    total_ms: float


def _get_engine() -> HybridRetriever:
    global _EMBEDDER, _FAISS_INDEX, _BM25_INDEX, _METADATA_STORE, _RETRIEVER
    if _RETRIEVER is None:
        index_dir = "data/index"
        faiss_path = os.path.join(index_dir, "vectors.faiss")
        bm25_path = os.path.join(index_dir, "bm25.pkl")
        meta_path = os.path.join(index_dir, "metadata.parquet")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _EMBEDDER = EmbeddingModel(model_name)
        _FAISS_INDEX = FAISSIndex.load(faiss_path)
        _BM25_INDEX = BM25Index.load(bm25_path)
        _METADATA_STORE = MetadataStore.load(meta_path)

        dense_ret = Retriever(_EMBEDDER, _FAISS_INDEX, _METADATA_STORE)
        _RETRIEVER = HybridRetriever(dense_ret, _BM25_INDEX)
    return _RETRIEVER


def warmup():
    """
    Load models and indices into RAM and run warm-up inference.
    """
    engine = _get_engine()
    for _ in range(5):
        search("warmup query test", top_k=5)


def search(query: str, top_k: int = 5) -> SearchResult:
    """
    Execute high-speed retrieval and return SearchResult with exact sub-stage timings.
    """
    t_start = time.perf_counter()
    engine = _get_engine()

    # 1. Encode embedding
    t_emb_start = time.perf_counter()
    query_vec = engine.dense_retriever.embedding_model.encode_query(query)
    t_emb_end = time.perf_counter()
    embed_ms = (t_emb_end - t_emb_start) * 1000

    # 2. FAISS + BM25 Fast Retrieval
    t_search_start = time.perf_counter()
    response = engine.search(query, top_k=top_k, enable_reranking=False)
    t_search_end = time.perf_counter()
    search_ms = (t_search_end - t_search_start) * 1000

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_start) * 1000

    return SearchResult(
        query=query,
        results=response.get("results", []),
        embed_ms=round(embed_ms, 2),
        search_ms=round(search_ms, 2),
        total_ms=round(total_ms, 2)
    )
