from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper for multilingual SentenceTransformer embedding models.
    Ensures normalized embeddings for fast inner-product (cosine) retrieval in FAISS.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode a list of texts into normalized float32 embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalized 1D float32 vector.
        """
        return self.encode([query], show_progress_bar=False)[0]
