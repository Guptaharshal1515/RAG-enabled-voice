import re
import os
import pickle
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """
    Multilingual word-level tokenizer using Unicode regex.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Index:
    """
    BM25 Okapi lexical index built on the same shared chunk corpus.
    """

    def __init__(self, documents: Optional[List[Dict[str, Any]]] = None):
        self.documents = documents or []
        if self.documents:
            tokenized_corpus = [
                tokenize(str(doc.get("text", "")))
                for doc in self.documents
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for i in ranked_indices:
            if scores[i] <= 0:
                continue
            doc = self.documents[i].copy()
            results.append({
                "index": i,
                "score": float(scores[i]),
                "document": doc
            })
        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 index not found at '{path}'")
        with open(path, "rb") as f:
            return pickle.load(f)
