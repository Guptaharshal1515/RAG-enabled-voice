import unittest
import numpy as np
from src.retrieval.bm25_index import BM25Index, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever
from src.retrieval.hybrid_retriever import HybridRetriever


class MockEmbedder:
    def __init__(self):
        self.dimension = 4

    def encode_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class MockReranker:
    def rerank(self, query, candidates, top_k=5):
        # Reverse candidate order for test
        ranked = []
        for idx, c in enumerate(candidates):
            doc = c.get("document", c)
            ranked.append({
                "chunk_id": doc["chunk_id"],
                "reranker_score": 0.9 - (idx * 0.1),
                "rrf_score": c.get("rrf_score", 0.0),
                "document": doc
            })
        return ranked[:top_k]


class TestHybridRetrieval(unittest.TestCase):

    def setUp(self):
        self.docs = [
            {"chunk_id": "c1", "document_id": "d1", "text": "CRISPR Cas9 is a gene editing technology."},
            {"chunk_id": "c2", "document_id": "d2", "text": "Earthquakes are caused by tectonic plates."},
            {"chunk_id": "c3", "document_id": "d3", "text": "Photosynthesis converts solar light into sugar."}
        ]

    def test_tokenize(self):
        tokens = tokenize("CRISPR-Cas9 gene editing!")
        self.assertIn("crispr", tokens)
        self.assertIn("cas9", tokens)
        self.assertIn("gene", tokens)

    def test_bm25_index_and_search(self):
        index = BM25Index(self.docs)
        results = index.search("CRISPR gene", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["document"]["chunk_id"], "c1")

    def test_reciprocal_rank_fusion(self):
        list1 = [{"document": self.docs[0]}, {"document": self.docs[1]}]
        list2 = [{"document": self.docs[0]}, {"document": self.docs[2]}]
        fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        # c1 was rank 1 in both lists, so it must be top
        self.assertEqual(fused[0]["chunk_id"], "c1")
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_hybrid_retriever_pipeline(self):
        faiss_idx = FAISSIndex(dimension=4)
        faiss_idx.add(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32))

        meta = MetadataStore()
        for d in self.docs:
            meta.add(d)

        bm25 = BM25Index(self.docs)
        embedder = MockEmbedder()
        dense_retriever = Retriever(embedder, faiss_idx, meta)
        mock_reranker = MockReranker()

        hybrid = HybridRetriever(dense_retriever, bm25, mock_reranker)
        res = hybrid.search("CRISPR gene", top_k=2, enable_reranking=True)

        self.assertEqual(len(res["results"]), 2)
        self.assertIn("latencies_ms", res)
        self.assertIn("dense_retrieval", res["latencies_ms"])
        self.assertIn("bm25_retrieval", res["latencies_ms"])
        self.assertIn("rrf_fusion", res["latencies_ms"])
        self.assertIn("reranking", res["latencies_ms"])
        self.assertIn("total", res["latencies_ms"])


if __name__ == "__main__":
    unittest.main()
