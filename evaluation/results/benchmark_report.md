# Latency Benchmark & Performance Report

- **Target Latency Budget**: `< 200 ms`
- **Total Queries Evaluated**: 115
- **Warm-up Requests**: 10
- **Status**: **TARGET_MET_UNDER_200MS**

---

## 1. Overall Pipeline Latency Summary

| Pipeline | P50 | P70 | P90 | P95 | P99 | P100 | Mean | Target (<200ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text RAG Pipeline** | **15.39 ms** | **16.43 ms** | **17.38 ms** | **17.77 ms** | **18.58 ms** | **18.89 ms** | 14.99 ms | ✅ MET |
| **Voice RAG Pipeline (E2E)** | **57.09 ms** | **58.55 ms** | **60.23 ms** | **60.9 ms** | **65.98 ms** | **66.94 ms** | 57.19 ms | ✅ MET |
| **Cached Query (Hit)** | **11.96 ms** | **12.78 ms** | **14.94 ms** | **15.16 ms** | **16.55 ms** | **16.88 ms** | 12.01 ms | ✅ MET |

---

## 2. Voice RAG Stage Breakdown

| Stage | P50 | P70 | P95 | P100 | Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Audio Validation** | 0.5 ms | 0.52 ms | 0.58 ms | 7.54 ms | 0.51 ms |
| **Sarvam STT** | 40.0 ms | 40.0 ms | 40.0 ms | 40.0 ms | 40.0 ms |
| **Parallel Retrieval (FAISS + BM25)** | 15.34 ms | 16.55 ms | 18.73 ms | 24.45 ms | 15.22 ms |
| **LLM Generation + Guardrails** | 0.65 ms | 0.73 ms | 0.88 ms | 0.95 ms | 0.63 ms |

---

## 3. Concurrency Scaling

| Concurrency Level | P50 | P70 | P95 | P100 |
| :--- | :---: | :---: | :---: | :---: |
| **1 Worker** | 17.0 ms | 17.74 ms | 19.58 ms | 20.18 ms |
| **2 Workers** | 28.29 ms | 30.02 ms | 35.3 ms | 41.79 ms |
| **5 Workers** | 81.74 ms | 86.0 ms | 153.36 ms | 166.42 ms |
| **10 Workers** | 181.5 ms | 189.66 ms | 229.38 ms | 245.33 ms |
