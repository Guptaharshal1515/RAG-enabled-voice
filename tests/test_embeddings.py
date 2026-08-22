import unittest
import numpy as np
from src.embeddings.model import EmbeddingModel


class TestEmbeddingModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedder = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")

    def test_dynamic_dimension(self):
        self.assertEqual(self.embedder.dimension, 384)

    def test_encode_single_and_batch(self):
        texts = ["Short sample sentence.", "Another test phrase."]
        embeddings = self.embedder.encode(texts)
        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_normalization(self):
        query = "Test normalization of query vector"
        q_vec = self.embedder.encode_query(query)
        self.assertEqual(q_vec.shape, (384,))
        # Norm of L2-normalized vector must be approx 1.0
        norm = np.linalg.norm(q_vec)
        self.assertAlmostEqual(norm, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
