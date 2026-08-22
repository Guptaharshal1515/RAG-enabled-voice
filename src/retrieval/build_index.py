import os
import sys
import json
import time
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.metadata_store import MetadataStore


def build_vector_index(
    chunks_path: str = "data/processed/chunks.parquet",
    index_dir: str = "data/index",
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 32
):
    """
    Build and persist FAISS index and metadata store from processed chunks.
    """
    print(f"Loading chunks from '{chunks_path}'...")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file '{chunks_path}' not found. Run process_documents.py first.")

    df = pd.read_parquet(chunks_path)
    records = df.to_dict(orient="records")
    total_chunks = len(records)
    print(f"Loaded {total_chunks} chunks.")

    print(f"Initializing embedding model: {model_name}...")
    t0 = time.perf_counter()
    embedder = EmbeddingModel(model_name=model_name)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s. Embedding dimension: {embedder.dimension}")

    texts = [str(r.get("text", "")) for r in records]

    print(f"Encoding {total_chunks} chunks (batch_size={batch_size})...")
    t_enc = time.perf_counter()
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    enc_time = time.perf_counter() - t_enc
    print(f"Embeddings generated in {enc_time:.2f}s ({total_chunks / max(enc_time, 0.001):.1f} chunks/sec).")

    print("Building FAISS FlatIP index...")
    faiss_index = FAISSIndex(dimension=embedder.dimension)
    faiss_index.add(embeddings)

    print("Building BM25 Okapi lexical index...")
    bm25_index = BM25Index(documents=records)

    metadata_store = MetadataStore()
    for r in records:
        metadata_store.add(r)

    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, "vectors.faiss")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    meta_path = os.path.join(index_dir, "metadata.parquet")
    config_path = os.path.join(index_dir, "index_config.json")

    print(f"Saving FAISS index to '{faiss_path}'...")
    faiss_index.save(faiss_path)

    print(f"Saving BM25 index to '{bm25_path}'...")
    bm25_index.save(bm25_path)

    print(f"Saving metadata to '{meta_path}'...")
    metadata_store.save(meta_path)

    config = {
        "model_name": model_name,
        "dimension": embedder.dimension,
        "total_vectors": faiss_index.total_vectors,
        "total_bm25_docs": len(records),
        "index_type": "Hybrid (FAISS FlatIP + BM25Okapi)",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nIndex build complete!")
    print(f"  Vectors indexed : {faiss_index.total_vectors}")
    print(f"  Config saved to : {config_path}")


if __name__ == "__main__":
    build_vector_index(
        chunks_path="data/processed/chunks.parquet",
        index_dir="data/index",
        model_name="sentence-transformers/all-MiniLM-L6-v2", # Fast default, can also use BAAI/bge-m3
        batch_size=32
    )
