import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.model import EmbeddingModel
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.bm25_index import BM25Index
from src.retrieval.metadata_store import MetadataStore
from src.retrieval.search import Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever


def run_benchmark(
    index_dir: str = "data/index",
    chunks_path: str = "data/processed/chunks.parquet"
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("           COMPREHENSIVE RETRIEVAL BENCHMARK")
    print("      (FAISS Only vs BM25 Only vs Hybrid RRF vs Hybrid + Reranker)")
    print("=" * 75)

    faiss_path = os.path.join(index_dir, "vectors.faiss")
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    meta_path = os.path.join(index_dir, "metadata.parquet")
    config_path = os.path.join(index_dir, "index_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"Loading embedding model: {model_name}...")
    embedder = EmbeddingModel(model_name)
    faiss_index = FAISSIndex.load(faiss_path)
    bm25_index = BM25Index.load(bm25_path)
    metadata_store = MetadataStore.load(meta_path)

    print("Initializing CrossEncoder Reranker...")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    dense_retriever = Retriever(embedder, faiss_index, metadata_store)
    hybrid_retriever = HybridRetriever(dense_retriever, bm25_index, reranker)

    df_chunks = pd.read_parquet(chunks_path)
    gold_records = df_chunks[df_chunks["is_gold_passage"] == True]
    if gold_records.empty:
        gold_records = df_chunks.drop_duplicates(subset=["query_id"])

    queries_to_eval = gold_records["query_id"].unique()
    print(f"Evaluating across {len(queries_to_eval)} test queries...\n")

    strategies = ["FAISS_Only", "BM25_Only", "Hybrid_RRF", "Hybrid_Reranked"]
    results_summary = {}

    for strat in strategies:
        r_at_1 = []
        r_at_5 = []
        r_at_10 = []
        reciprocal_ranks = []
        latencies = []

        for qid in queries_to_eval:
            gold_doc_ids = set(df_chunks[(df_chunks["query_id"] == qid) & (df_chunks["is_gold_passage"] == True)]["document_id"])
            sample_row = df_chunks[df_chunks["query_id"] == qid].iloc[0]
            query_text = sample_row.get("text", "")[:120]

            t0 = time.perf_counter()

            if strat == "FAISS_Only":
                res = dense_retriever.search(query_text, top_k=10)
                retrieved_doc_ids = [r["document_id"] for r in res["results"]]
                elapsed_ms = res["latencies_ms"]["total"]

            elif strat == "BM25_Only":
                bm_res = bm25_index.search(query_text, top_k=10)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                retrieved_doc_ids = [r["document"]["document_id"] for r in bm_res]

            elif strat == "Hybrid_RRF":
                h_res = hybrid_retriever.search(query_text, top_k=10, enable_reranking=False)
                elapsed_ms = h_res["latencies_ms"]["total"]
                retrieved_doc_ids = [r["document_id"] for r in h_res["results"]]

            elif strat == "Hybrid_Reranked":
                h_res = hybrid_retriever.search(query_text, top_k=10, enable_reranking=True)
                elapsed_ms = h_res["latencies_ms"]["total"]
                retrieved_doc_ids = [r["document_id"] for r in h_res["results"]]

            latencies.append(elapsed_ms)

            hit_1 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:1]) else 0.0
            hit_5 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:5]) else 0.0
            hit_10 = 1.0 if any(d in gold_doc_ids for d in retrieved_doc_ids[:10]) else 0.0

            r_at_1.append(hit_1)
            r_at_5.append(hit_5)
            r_at_10.append(hit_10)

            rr = 0.0
            for rank_idx, doc_id in enumerate(retrieved_doc_ids, start=1):
                if doc_id in gold_doc_ids:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

        results_summary[strat] = {
            "Recall@1": round(float(np.mean(r_at_1)), 4),
            "Recall@5": round(float(np.mean(r_at_5)), 4),
            "Recall@10": round(float(np.mean(r_at_10)), 4),
            "MRR@10": round(float(np.mean(reciprocal_ranks)), 4),
            "P50_ms": round(float(np.percentile(latencies, 50)), 2),
            "P95_ms": round(float(np.percentile(latencies, 95)), 2),
            "Mean_ms": round(float(np.mean(latencies)), 2)
        }

    # Print Table
    print(f"{'Strategy':<18} | {'Recall@1':<9} | {'Recall@5':<9} | {'Recall@10':<9} | {'MRR@10':<8} | {'Latency P50':<11} | {'Latency P95':<11}")
    print("-" * 85)
    for strat, m in results_summary.items():
        print(f"{strat:<18} | {m['Recall@1']*100:>7.1f}% | {m['Recall@5']*100:>7.1f}% | {m['Recall@10']*100:>8.1f}% | {m['MRR@10']:>8.4f} | {m['P50_ms']:>8.2f} ms | {m['P95_ms']:>8.2f} ms")
    print("=" * 85 + "\n")

    os.makedirs("evaluation/results", exist_ok=True)
    out_file = "evaluation/results/retrieval_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Benchmark results exported to '{out_file}'.")

    return results_summary


if __name__ == "__main__":
    run_benchmark()
