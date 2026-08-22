import os
import sys
import json
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.providers.fast_provider import FastGroundedProvider
from src.generation.generator import Generator
from src.harness.rag_pipeline import RAGPipeline


def initialize_rag_system(index_dir: str = "data/index") -> RAGPipeline:
    print("=" * 65)
    print("           INITIALIZING VOICE-READY RAG HARNESS")
    print("=" * 65)

    faiss_path = os.path.join(index_dir, "vectors.faiss")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    meta_path = os.path.join(index_dir, "metadata.parquet")
    config_path = os.path.join(index_dir, "index_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"1. Loading Embedding Model: {model_name}")
    embedder = EmbeddingModel(model_name)

    print(f"2. Loading FAISS Dense Index & BM25 Lexical Index ({config.get('total_vectors', 0)} chunks)...")
    faiss_index = FAISSIndex.load(faiss_path)
    bm25_index = BM25Index.load(bm25_path)
    metadata_store = MetadataStore.load(meta_path)

    dense_retriever = Retriever(embedder, faiss_index, metadata_store)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_index)

    print("3. Initializing LLM Generation Engine...")
    llm = FastGroundedProvider()
    generator = Generator(llm)

    print("4. Assembling RAG Pipeline Harness...")
    pipeline = RAGPipeline(hybrid_retriever, generator, max_context_tokens=2500)
    print("RAG Pipeline ready!\n")
    return pipeline


def run_demo():
    pipeline = initialize_rag_system()

    queries = [
        "What is a corporation in law?",
        "What causes earthquakes?",
        "Who won yesterday's football match?"
    ]

    for q in queries:
        print("-" * 65)
        print(f"QUERY: {q}")
        response = pipeline.run(q, top_k=3, enable_reranking=False)
        print(f"ANSWER:\n  {response.answer}")
        print(f"SOURCES CITED: {[s.chunk_id for s in response.sources]}")
        print(f"GROUNDED: {response.grounded} | REFUSAL: {response.refusal}")
        print(f"LATENCY BREAKDOWN: {response.latencies_ms}")
        print("-" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
