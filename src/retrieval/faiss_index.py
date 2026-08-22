from typing import Tuple
import os
import faiss
import numpy as np


class FAISSIndex:
    """
    High-performance FAISS Flat Inner-Product index for cosine similarity search
    over L2-normalized embeddings.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        """
        Add float32 normalized embeddings to the index.
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}"
            )
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top_k nearest neighbors.
        Returns:
            scores: 1D array of inner product / cosine scores
            indices: 1D array of vector IDs
        """
        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index file not found at '{path}'")
        raw_index = faiss.read_index(path)
        obj = cls(raw_index.d)
        obj.index = raw_index
        return obj
