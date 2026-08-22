import os
import sys
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever


def evaluate_retrieval(
    index_dir: str = "data/index",
    chunks_path: str = "data/processed/chunks.parquet"
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality (Recall@1, Recall@5, Recall@10, MRR@10)
    and retrieval latency (P50, P95, Mean) against ground truth gold passages.
    """
    print("\n" + "=" * 65)
    print("           RETRIEVAL ENGINE EVALUATION")
    print("=" * 65)

    faiss_path = os.path.join(index_dir, "vectors.faiss")
    meta_path = os.path.join(index_dir, "metadata.parquet")
    config_path = os.path.join(index_dir, "index_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        import json
        config = json.load(f)

    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {model_name}...")
    embedder = EmbeddingModel(model_name)
    faiss_index = FAISSIndex.load(faiss_path)
    metadata_store = MetadataStore.load(meta_path)
    retriever = Retriever(embedder, faiss_index, metadata_store)

    df_chunks = pd.read_parquet(chunks_path)
    
    # Identify unique gold query-document pairs
    gold_records = df_chunks[df_chunks["is_gold_passage"] == True]
    if gold_records.empty:
        print("Warning: No gold passage flags found, falling back to all unique queries.")
        gold_records = df_chunks.drop_duplicates(subset=["query_id"])

    queries_to_eval = gold_records["query_id"].unique()
    print(f"Evaluating on {len(queries_to_eval)} test queries...")

    r_at_1 = []
    r_at_5 = []
    r_at_10 = []
    reciprocal_ranks = []
    latencies_ms = []

    for qid in queries_to_eval:
        gold_doc_ids = set(df_chunks[(df_chunks["query_id"] == qid) & (df_chunks["is_gold_passage"] == True)]["document_id"])
        
        # Take representative text for query simulation
        sample_row = df_chunks[df_chunks["query_id"] == qid].iloc[0]
        query_text = sample_row.get("text", "")[:100]

        res = retriever.search(query_text, top_k=10)
        latencies_ms.append(res["latencies_ms"]["total"])

        retrieved_doc_ids = [r["document_id"] for r in res["results"]]

        # Metrics computation
        hit_1 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:1]) else 0.0
        hit_5 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:5]) else 0.0
        hit_10 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:10]) else 0.0

        r_at_1.append(hit_1)
        r_at_5.append(hit_5)
        r_at_10.append(hit_10)

        # MRR calculation
        rr = 0.0
        for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in gold_doc_ids:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

    metrics = {
        "total_queries_evaluated": len(queries_to_eval),
        "recall_at_1": round(float(np.mean(r_at_1)), 4),
        "recall_at_5": round(float(np.mean(r_at_5)), 4),
        "recall_at_10": round(float(np.mean(r_at_10)), 4),
        "mrr_at_10": round(float(np.mean(reciprocal_ranks)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 2)
    }

    print("\nRetrieval Metrics:")
    print(f"  Recall@1  : {metrics['recall_at_1'] * 100:.1f}%")
    print(f"  Recall@5  : {metrics['recall_at_5'] * 100:.1f}%")
    print(f"  Recall@10 : {metrics['recall_at_10'] * 100:.1f}%")
    print(f"  MRR@10    : {metrics['mrr_at_10']:.4f}")

    print("\nLatency Metrics (Voice RAG Budget < 200ms):")
    print(f"  P50 Latency : {metrics['latency_p50_ms']} ms")
    print(f"  P95 Latency : {metrics['latency_p95_ms']} ms")
    print(f"  Mean Latency: {metrics['latency_mean_ms']} ms")
    print("=" * 65 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_retrieval()
