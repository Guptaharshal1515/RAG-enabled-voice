<p align="center">
  <h1 align="center">🎙️ Multilingual Voice-Enabled RAG System</h1>
  <p align="center">
    <strong>Sub-200ms Voice-First Retrieval-Augmented Generation • 11 Indian Languages • Grounded on MSMARCO-XI</strong>
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick_Start-▶-00c853?style=for-the-badge" alt="Quick Start"></a>
    <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-📐-7c4dff?style=for-the-badge" alt="Architecture"></a>
    <a href="#-benchmarks"><img src="https://img.shields.io/badge/Benchmarks-⚡-ff6d00?style=for-the-badge" alt="Benchmarks"></a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/FAISS-Vector_DB-0467df" alt="FAISS">
    <img src="https://img.shields.io/badge/Sarvam_AI-STT-ff4081" alt="Sarvam">
    <img src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Docker-2496ed?logo=docker&logoColor=white" alt="Docker">
  </p>
</p>

---

> **HH Goa 2026 — Shortlisting Task 2**  
> A production-grade voice-enabled RAG pipeline: Speak a question in any of 11 Indian languages → real-time transcription → intelligent retrieval → grounded answer with citations — all under 200ms retrieval latency.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎤 **Live Microphone Input** | Browser-native 16kHz PCM WAV recording with real-time waveform visualizer |
| 🗣️ **Sarvam AI STT** | Real-time multilingual speech-to-text using `saaras:v3` — supports English + 10 Indian languages |
| 🔀 **Hybrid Retrieval** | Concurrent FAISS dense vector search + BM25 lexical matching with Reciprocal Rank Fusion |
| ⚡ **Sub-200ms Latency** | P95 retrieval latency of **22.65ms** against a 50ms budget — benchmarked across 100+ queries |
| 🛡️ **5-Layer Guardrails** | Input validation → Injection detection → Retrieval confidence → Grounding verification → Centralized refusal policy |
| 🌐 **11 Languages** | English, Hindi, Tamil, Telugu, Kannada, Bengali, Assamese, Marathi, Gujarati, Odia, Punjabi |
| 🧩 **Adaptive Chunking** | 4-strategy chunking router: whole-document, sentence-aware, recursive hierarchical, semantic fallback |
| 📊 **Full Observability** | Per-stage latency instrumentation (STT, embedding, retrieval, generation) with P50/P70/P95/P99/P100 reporting |

---

## 🏗️ Architecture

```
                    🎤 USER SPEAKS
                          │
                    ┌─────┴─────┐
                    │ Sarvam AI │  ← Real-time multilingual STT (saaras:v3)
                    │    STT    │
                    └─────┬─────┘
                          │
                     📝 Transcript
                          │
                    ┌─────┴─────┐
                    │   Input   │  ← Empty query / length / safety validation
                    │   Guard   │
                    └─────┬─────┘
                          │
               ┌──────────┴──────────┐
               │                     │
          ┌────┴────┐          ┌─────┴────┐
          │  FAISS  │          │  BM25    │  ← Concurrent retrieval
          │ Dense   │          │ Lexical  │     via ThreadPoolExecutor
          └────┬────┘          └─────┬────┘
               │                     │
               └──────────┬──────────┘
                          │
                    ┌─────┴─────┐
                    │    RRF    │  ← Reciprocal Rank Fusion (k=60)
                    │  Fusion   │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │ Cross-    │  ← ms-marco-MiniLM-L-6-v2
                    │ Encoder   │
                    │ Reranker  │
                    └─────┬─────┘
                          │
                      Top-K Chunks
                          │
                    ┌─────┴─────┐
                    │    LLM    │  ← Grounded synthesis with [chunk_id] citations
                    │ Generator │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │ Grounding │  ← Citation validation + claim-level verification
                    │   Guard   │
                    └─────┬─────┘
                          │
                    ✅ Verified Answer
```

---

## 📁 Project Structure

```
RAG-enabled-voice/
├── app/                          # FastAPI Web Application
│   ├── main.py                   # Server + Interactive Voice UI
│   ├── retriever.py              # Unified search adapter
│   ├── benchmark.py              # Instructor benchmark adapter
│   └── config.py                 # Latency budget configuration
│
├── src/
│   ├── chunking/                 # Adaptive Multi-Strategy Chunking
│   │   ├── adaptive_chunker.py   # 4-strategy routing engine
│   │   ├── sentence_chunker.py   # Sentence-aware chunking with overlap
│   │   ├── recursive_chunker.py  # Recursive hierarchical splitting
│   │   └── token_counter.py      # Token estimation utility
│   │
│   ├── embeddings/               # Dense Vector Embeddings
│   │   └── model.py              # SentenceTransformers (all-MiniLM-L6-v2, 384-d)
│   │
│   ├── retrieval/                # Hybrid Retrieval Engine
│   │   ├── faiss_index.py        # FAISS IndexFlatIP (L2-normalized cosine)
│   │   ├── bm25_index.py         # BM25Okapi lexical retrieval
│   │   ├── hybrid_retriever.py   # Concurrent FAISS + BM25 with RRF
│   │   ├── fusion.py             # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py           # Cross-Encoder reranking
│   │   └── metadata_store.py     # Chunk metadata (Parquet-backed)
│   │
│   ├── generation/               # Answer Generation
│   │   ├── generator.py          # Structured answer synthesis
│   │   ├── prompt.py             # Grounded RAG prompt template
│   │   ├── llm.py                # LLM abstraction + retry logic
│   │   ├── providers/
│   │   │   └── fast_provider.py  # <5ms deterministic grounded engine
│   │   └── schemas.py            # RAGResponse, Source, StructuredAnswer
│   │
│   ├── guardrails/               # 5-Layer Safety System
│   │   ├── input_guard.py        # Query validation (empty, length, safety)
│   │   ├── injection_guard.py    # Prompt injection detection + sanitization
│   │   ├── retrieval_guard.py    # Retrieval confidence thresholding
│   │   ├── grounding_guard.py    # Citation verification + claim grounding
│   │   └── policy.py             # Centralized refusal policy
│   │
│   ├── harness/                  # Pipeline Orchestration
│   │   ├── rag_pipeline.py       # Full guarded text RAG pipeline
│   │   └── voice_rag_pipeline.py # End-to-end voice → answer pipeline
│   │
│   ├── voice/                    # Speech Layer
│   │   ├── sarvam_stt.py         # Real SarvamSTT + MockSarvamSTT
│   │   ├── audio.py              # Audio validation + synthetic WAV generator
│   │   └── schemas.py            # TranscriptionResult, VoiceRAGResponse
│   │
│   └── observability/            # Instrumentation
│       ├── timer.py              # Microsecond precision timer
│       ├── metrics.py            # Percentile statistics (P50–P100)
│       └── cache.py              # In-memory LRU response cache
│
├── evaluation/                   # Benchmarking & Evaluation
│   ├── latency.py                # 115-query latency benchmark engine
│   ├── queries.jsonl             # Multilingual benchmark queries
│   ├── guardrail_eval.py         # Guardrail accuracy evaluation
│   └── guardrail_tests.json      # Guardrail test scenarios
│
├── tests/                        # Unit Tests (44 tests)
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_hybrid_retrieval.py
│   ├── test_generation.py
│   ├── test_guardrails.py
│   ├── test_voice.py
│   ├── test_observability.py
│   └── test_pipeline.py
│
├── config/settings.py            # Environment configuration loader
├── Dockerfile                    # Production container image
├── docker-compose.yml            # Docker Compose orchestration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md                     # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Sarvam AI API Key](https://www.sarvam.ai/) (for speech-to-text)

### 1. Clone & Install

```bash
git clone https://github.com/Guptaharshal1515/RAG-enabled-voice.git
cd RAG-enabled-voice

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your real Sarvam API key:
# SARVAM_API_KEY=sk_your_actual_key_here
```

### 3. Build the Index (First Time Only)

```bash
python -m src.retrieval.build_index
```

This downloads the `ai4bharat/MSMARCO-XI` dataset, applies adaptive chunking, generates embeddings, and builds the FAISS + BM25 indices.

### 4. Launch the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser — click the 🎤 microphone, speak your question, and get a grounded answer!

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or directly
docker build -t voice-rag .
docker run -p 8000:8000 --env-file .env voice-rag
```

---

## ⚡ Benchmarks

### Retrieval Latency (Instructor Benchmark)

| Stage | Avg | P50 | P95 | P99 |
|---|---|---|---|---|
| **Embedding** | 8.13 ms | 7.97 ms | 9.24 ms | 10.43 ms |
| **Search** | 10.76 ms | 10.04 ms | 14.54 ms | 15.30 ms |
| **Total** | 18.90 ms | 18.25 ms | **22.65 ms** | 23.36 ms |

> **Result: `PASS` — P95 = 22.65ms vs 50ms budget** ✅

### Run Benchmarks

```bash
# Instructor benchmark (50 queries, 50ms budget)
python "benchmark (1).py" 50

# Full latency benchmark (115 queries)
python -m evaluation.latency
```

---

## 🧪 Testing

```bash
# Run all 44 unit tests
python -m unittest discover tests

# Expected output:
# Ran 44 tests in ~5s
# OK
```

### Test Coverage

| Module | Tests | Scenarios |
|---|---|---|
| Chunking | 6 | Adaptive routing, sentence-aware, recursive, edge cases |
| Embeddings | 2 | Model loading, vector normalization |
| Retrieval | 5 | Dense search, BM25, hybrid RRF, reranking |
| Generation | 7 | Answerable queries, refusals, injection resistance, retries |
| Guardrails | 8 | Input/injection/retrieval/grounding guards, policy tests |
| Voice | 7 | Audio validation, STT transcription, mock/real providers |
| Observability | 3 | Timer precision, percentile metrics, LRU cache |
| End-to-End Pipeline | 6 | Full voice→RAG flows, failure recovery, hallucination blocking |

---

## 🛡️ Guardrails System

The system implements **5 independent guardrail layers** to ensure safe, grounded responses:

```
Query → [1] Input Guard      → validates format, length, safety
      → [2] Injection Guard   → detects prompt injection in query + sources
      → [3] Retrieval Guard   → checks relevance confidence threshold
      → [4] Grounding Guard   → verifies citations + claim-level evidence
      → [5] Refusal Policy    → centralized, consistent refusal messages
```

| Guard | What It Catches |
|---|---|
| **InputGuard** | Empty queries, excessive length, malformed input |
| **InjectionGuard** | "Ignore previous instructions", system prompt leaks |
| **RetrievalGuard** | Off-topic queries, low-confidence retrieval (<0.005) |
| **GroundingGuard** | Hallucinated claims, fabricated citations, unsupported facts |
| **RefusalPolicy** | Centralized human-readable refusal messages |

---

## 🧩 Chunking Strategy

The `AdaptiveChunker` dynamically routes documents through 4 strategies based on token count:

| Token Range | Strategy | Rationale |
|---|---|---|
| ≤ 200 | **Whole Document** | Short passages kept intact to preserve context |
| 201 – 800 | **Sentence-Aware** | Split on sentence boundaries with 1-sentence overlap |
| 801 – 2,000 | **Recursive Hierarchical** | Multi-level structural splitting (paragraphs → sentences) |
| > 2,000 | **Semantic Fallback** | Recursive splitting with semantic boundary hooks |

**Result:** 1,023 chunks generated from MSMARCO-XI with optimal granularity for retrieval.

---

## 🌐 Supported Languages

| Code | Language | Code | Language |
|---|---|---|---|
| `en-IN` | English | `ta-IN` | Tamil |
| `hi-IN` | Hindi | `te-IN` | Telugu |
| `bn-IN` | Bengali | `mr-IN` | Marathi |
| `kn-IN` | Kannada | `gu-IN` | Gujarati |
| `as-IN` | Assamese | `pa-IN` | Punjabi |
| `ml-IN` | Malayalam | `od-IN` | Odia |

---

## 🔧 Configuration

All configuration is managed through environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `SARVAM_API_KEY` | — | Your Sarvam AI API subscription key |
| `SARVAM_STT_MODEL` | `saaras:v3` | Sarvam STT model identifier |
| `SARVAM_STT_ENDPOINT` | `https://api.sarvam.ai/speech-to-text` | STT API endpoint |
| `MAX_AUDIO_SIZE_BYTES` | `26214400` (25 MB) | Maximum upload audio file size |
| `MAX_AUDIO_DURATION_SEC` | `60.0` | Maximum audio recording duration |
| `STT_TIMEOUT_SEC` | `5.0` | STT API request timeout |

---

## 📊 Dataset

**[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)** — A multilingual extension of MS MARCO spanning 11 Indian languages with 10M+ passages covering science, law, health, geography, and general knowledge.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Speech-to-Text** | Sarvam AI (`saaras:v3`) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Vector Database** | FAISS (`IndexFlatIP`, L2-normalized cosine similarity) |
| **Lexical Search** | BM25Okapi (`rank-bm25`) |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Fusion** | Reciprocal Rank Fusion (k=60) |
| **Web Framework** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS with Web Audio API |
| **Containerization** | Docker + Docker Compose |

---

## 📄 API Reference

### `POST /query`

Unified endpoint accepting voice audio or text queries.

**Form Parameters:**
| Field | Type | Description |
|---|---|---|
| `audio` | `File` (optional) | Audio file (WAV/WebM/MP3) |
| `query` | `string` (optional) | Text query |
| `language` | `string` | Language code (`unknown`, `en-IN`, `hi-IN`, etc.) |

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "input_type": "voice",
  "transcript": "भूकंप क्यों आता है?",
  "detected_language": "hi-IN",
  "answer": "Tectonic plates move along faults causing earthquakes. [doc_001_c0]",
  "sources": [
    {"chunk_id": "doc_001_c0", "score": 0.92, "text": "..."}
  ],
  "grounded": true,
  "refusal": false,
  "latencies_ms": {
    "audio_validation": 0.5,
    "stt": 725.0,
    "retrieval": 11.0,
    "generation": 0.6,
    "total_e2e": 750.0
  }
}
```

### `GET /health`

Health check endpoint returning system status.

---

## 👤 Author

**Harshal Gupta** — [GitHub](https://github.com/Guptaharshal1515)

---

<p align="center">
  Built with ❤️ for <strong>HH Goa 2026</strong>
</p>