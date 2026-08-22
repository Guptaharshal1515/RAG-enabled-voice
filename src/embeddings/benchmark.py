import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.embeddings.model import EmbeddingModel


def benchmark_embedding_models(
    models: List[str] = None,
    test_texts: List[str] = None,
    test_query: str = "What is a corporation in law?"
) -> List[Dict[str, Any]]:
    """
    Benchmark candidate embedding models on dimension, encoding latency, and throughput.
    """
    if models is None:
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]

    if test_texts is None:
        test_texts = [
            "A corporation is a company authorized to act as a single entity.",
            "Earthquakes occur due to tectonic plate friction.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "భారతదేశం వివిధ సంస్కృతులకు నిలయం.",
            "भारत एक विशाल और सुंदर देश है।"
        ] * 10 # 50 samples

    results = []

    print("\n" + "=" * 65)
    print("           EMBEDDING MODEL BENCHMARK")
    print("=" * 65)

    for model_name in models:
        print(f"\nEvaluating model: {model_name}")
        t0 = time.perf_counter()
        embedder = EmbeddingModel(model_name)
        load_time = time.perf_counter() - t0

        # Benchmark batch encoding
        t_batch_start = time.perf_counter()
        embeddings = embedder.encode(test_texts, batch_size=16)
        t_batch_end = time.perf_counter()
        batch_duration_ms = (t_batch_end - t_batch_start) * 1000
        throughput = len(test_texts) / (t_batch_end - t_batch_start)

        # Benchmark single query encoding (relevant for voice latency budget)
        query_latencies = []
        for _ in range(10):
            t_q_start = time.perf_counter()
            q_vec = embedder.encode_query(test_query)
            t_q_end = time.perf_counter()
            query_latencies.append((t_q_end - t_q_start) * 1000)

        p50_query_ms = np.median(query_latencies)
        p95_query_ms = np.percentile(query_latencies, 95)

        res = {
            "model_name": model_name,
            "dimension": embedder.dimension,
            "load_time_sec": round(load_time, 2),
            "batch_50_docs_ms": round(batch_duration_ms, 2),
            "throughput_docs_sec": round(throughput, 1),
            "query_p50_ms": round(p50_query_ms, 2),
            "query_p95_ms": round(p95_query_ms, 2)
        }
        results.append(res)

        print(f"  Dimension        : {res['dimension']}")
        print(f"  Query Latency P50: {res['query_p50_ms']} ms")
        print(f"  Query Latency P95: {res['query_p95_ms']} ms")
        print(f"  Throughput       : {res['throughput_docs_sec']} docs/sec")

    print("\n" + "=" * 65)
    return results


if __name__ == "__main__":
    benchmark_embedding_models()
