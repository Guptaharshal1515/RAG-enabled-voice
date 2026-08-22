import unittest
import numpy as np
import os
import shutil
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever


class MockEmbeddingModel:
    def __init__(self, dimension=4):
        self.dimension = dimension

    def encode_query(self, query: str) -> np.ndarray:
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return vec / np.linalg.norm(vec)


class TestRetrievalPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/temp_index"
        os.makedirs(self.test_dir, exist_ok=True)
        self.dimension = 4
        self.faiss_index = FAISSIndex(self.dimension)
        self.metadata_store = MetadataStore()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_faiss_add_and_search(self):
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.707, 0.707, 0.0, 0.0]
        ], dtype=np.float32)
        self.faiss_index.add(vectors)
        self.assertEqual(self.faiss_index.total_vectors, 3)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        scores, indices = self.faiss_index.search(query, top_k=2)

        self.assertEqual(indices[0], 0) # Exact match
        self.assertAlmostEqual(scores[0], 1.0, places=4)
        self.assertEqual(indices[1], 2) # Closest match

    def test_faiss_save_and_load(self):
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.faiss_index.add(vectors)
        save_path = os.path.join(self.test_dir, "test.faiss")
        self.faiss_index.save(save_path)

        loaded_index = FAISSIndex.load(save_path)
        self.assertEqual(loaded_index.total_vectors, 1)
        self.assertEqual(loaded_index.dimension, 4)

    def test_metadata_store_save_load_and_retriever(self):
        v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.faiss_index.add(v1)

        doc = {
            "chunk_id": "doc_001_c0",
            "document_id": "doc_001",
            "text": "Sample text for testing",
            "language": "en"
        }
        self.metadata_store.add(doc)

        save_meta_path = os.path.join(self.test_dir, "meta.parquet")
        self.metadata_store.save(save_meta_path)
        loaded_meta = MetadataStore.load(save_meta_path)

        mock_embedder = MockEmbeddingModel(dimension=4)
        retriever = Retriever(mock_embedder, self.faiss_index, loaded_meta)

        res = retriever.search("sample query", top_k=1)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["chunk_id"], "doc_001_c0")
        self.assertIn("latencies_ms", res)
        self.assertIn("total", res["latencies_ms"])


if __name__ == "__main__":
    unittest.main()
