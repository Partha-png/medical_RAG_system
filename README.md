# 🏥 Medical RAG System

> A production-ready Retrieval-Augmented Generation system for medical document question-answering, built to answer only from your uploaded documents — no hallucinations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-00a393.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What It Does

Upload a medical PDF or text file. Ask questions. Get answers drawn exclusively from that document — with citations, quality metrics, and full evaluation tooling built in.

The system uses Retrieval-Augmented Generation: instead of relying on a language model's internal knowledge (which can hallucinate), every answer is grounded in chunks retrieved from your document via vector similarity or keyword search.

---

## Architecture

```
PDF / TXT Upload
      │
      ▼
  Document Chunker (token-based, 150 tokens, 100 overlap)
      │
      ├─── BioBERT Encoder ──► FAISS Index
      ├─── MedCPT Encoder  ──► FAISS Index
      ├─── BM25 Index (keyword)
      └─── Hybrid (BioBERT + BM25, RRF fusion)
      
  Query
      │
      ▼
  Retriever (top-k chunks from matching index)
      │
      ▼
  Groq LLM (context-only prompt, temp=0.1)
      │
      ▼
  Answer + Citations + Quality Metrics
```

**Backend:** FastAPI with session management, conversation history, and an evaluation API  
**Frontend:** Streamlit with chat UI, evaluation table, and manual scoring tab

---

## Features

**5 Retrieval Methods**
- `biobert` — BioBERT semantic search (medical domain fine-tuned)
- `medcpt` — MedCPT Article Encoder (clinical corpus)
- `bm25` — BM25 keyword search, no GPU required
- `hybrid` — BioBERT + BM25 fused via Reciprocal Rank Fusion (RRF)

**Automatic Quality Metrics (per query, no ground truth needed)**
- Faithfulness & Hallucination Rate
- Context Relevance, Top Chunk Relevance, Context Coverage, Context Diversity
- Answer Relevance, Context Utilization
- Latency (retrieval ms, generation ms, total ms)
- CPU %, RAM usage, GPU memory (if available)

**Manual / Reference-Based Evaluation (with ground truth)**
- BLEU, ROUGE-1/2/L, METEOR, Exact Match, Token F1
- BERTScore F1, Answer Correctness, Completeness
- Precision@k, Recall@k, Hit Rate, MRR, nDCG@k

**Production-Ready Backend**
- Session isolation: each user gets their own FAISS index directory
- Atomic file writes (no corrupt JSON on concurrent access)
- Retriever caching (models loaded once per session, not per query)
- Automatic temp-file cleanup after document processing

---

## Project Structure

```
medical_rag/
├── app.py                          # Streamlit frontend
├── backend/
│   ├── main.py                     # FastAPI app, middleware, exception handlers
│   ├── api/routes/
│   │   ├── session_routes.py       # Session CRUD
│   │   ├── document_routes.py      # Upload + processing
│   │   ├── query_routes.py         # RAG query + retrieval-only
│   │   └── evaluation_routes.py   # Auto + manual evaluation endpoints
│   ├── services/
│   │   ├── rag_service.py          # Retriever cache, LLM call, timings
│   │   ├── document_service.py     # Chunking, encoding, FAISS creation
│   │   ├── session_service.py      # Session lifecycle
│   │   ├── conversation_service.py # Conversation history
│   │   └── evaluation_service.py  # Metric dispatch
│   ├── repositories/
│   │   ├── session_repository.py   # JSON-backed session store
│   │   └── conversation_repository.py
│   ├── models/                     # Pydantic request/response models
│   └── core/
│       ├── config.py               # All env vars and paths
│       └── exceptions.py           # Custom exception types
├── information_retrieval/
│   ├── document_encoding/
│   │   ├── encoder.py              # Entry point: reads file, chunks, encodes, indexes
│   │   ├── bioBERT_encoder.py      # BioBERT mean-pool + L2 norm
│   │   ├── medcpt_encoder.py       # MedCPT mean-pool + L2 norm
│   │   ├── chunker.py              # Token-aware sentence-boundary chunker
│   │   └── faiss_manager.py        # FlatL2 / IVFFlat index builder
│   ├── retrievers/
│   │   ├── biobertretriever.py     # FAISS similarity search (BioBERT)
│   │   ├── medcptretriever.py      # FAISS similarity search (MedCPT)
│   │   ├── bm25_retriever.py       # BM25Okapi keyword retrieval
│   │   └── hybrid_retriever.py     # RRF fusion of dense + sparse
│   └── evaluation/
│       ├── auto_metrics.py         # Reference-free metrics (per query)
│       ├── manual_metrics.py       # BLEU/ROUGE/BERTScore/nDCG/...
│       ├── rag_metrics.py          # Legacy standalone RAG metrics
│       ├── retrieval_metrics.py    # Precision/Recall/F1 helpers
│       └── resource_metrics.py     # CPU/RAM/GPU sampler (context manager)
├── rag_system/
│   └── rag_pipeline.py             # Standalone CLI pipeline (development/testing)
├── data/                           # Sessions + conversation JSON files (auto-created)
├── information_retrieval/faiss_container/sessions/   # Per-session FAISS indices
├── temp_uploads/                   # Cleaned up after each upload
├── requirements.txt
├── start_backend.bat
└── start_frontend.bat
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Groq API key ([get one free](https://console.groq.com))
- GPU optional — all encoders auto-detect CUDA and fall back to CPU

### Installation

```bash
git clone https://github.com/yourusername/medical-rag.git
cd medical-rag

python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt

# Create .env
echo GROQ_API_KEY=your_key_here > .env
```

### Running

**Windows (batch files):**
```
start_backend.bat   # Terminal 1
start_frontend.bat  # Terminal 2
```

**Manual:**
```bash
# Terminal 1 — FastAPI backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app.py
```

**On Replit / Codespaces:** Both services start automatically via the configured workflows.

| Service | URL |
|---|---|
| Streamlit frontend | http://localhost:5000 (Replit) or http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger API docs | http://localhost:8000/docs |

---

## Usage Guide

### 1. Upload & Process
Go to the **Upload & Process** tab. Choose an encoder from the sidebar, upload a PDF or TXT file, and click **Process Document**. The document is chunked, encoded, and stored in a session-specific FAISS index.

### 2. Chat
Switch to the **Chat** tab. Ask questions in natural language. Each answer shows:
- The generated response (grounded in your document only)
- Retrieved source chunks (expandable)
- Automatic quality metrics (faithfulness, hallucination rate, latency, etc.)

### 3. Compare Encoders
Change the encoder in the sidebar and re-ask the same question. Every run is logged to the **Evaluation Table** tab, where you can filter by question, compare encoders side-by-side, and download a CSV.

### 4. Manual Evaluation
In the **Manual Evaluation** tab, pick any logged run, paste a ground-truth answer and optionally paste relevant passages. The system computes BLEU, ROUGE, BERTScore, nDCG@k, and more, and merges results into the Evaluation Table.

---

## Configuration

All configuration is via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model served by Groq |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_MAX_TOKENS` | `500` | Max answer length |
| `DEFAULT_ENCODER` | `biobert` | Default encoder type |
| `BATCH_SIZE` | `8` | Embedding batch size |
| `CHUNK_MAX_TOKENS` | `150` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `TOP_K_CHUNKS` | `3` | Default chunks to retrieve |
| `API_PORT` | `8000` | Backend port |
| `FRONTEND_PORT` | `5000` | Frontend port |

---

## Encoder Details

| Encoder | Type | Model | GPU needed |
|---|---|---|---|
| `biobert` | Dense | `dmis-lab/biobert-v1.1` | Recommended |
| `medcpt` | Dense | `ncbi/MedCPT-Article-Encoder` | Recommended |
| `bm25` | Sparse | BM25Okapi (rank-bm25) | No |
| `hybrid` | Dense + Sparse | BioBERT + BM25, RRF α=0.5 | Recommended |

All dense encoders use mean pooling over the attention mask and L2 normalisation, producing unit-norm embeddings compatible with FAISS IndexFlatL2 / IndexIVFFlat.

For large documents (>1000 chunks), the system automatically switches to an IVF index for faster retrieval.

---

## API Reference

The full interactive API is available at `/docs` when the backend is running. Key endpoints:

```
POST   /api/sessions                         Create a new session
GET    /api/sessions                         List all sessions
DELETE /api/sessions/{session_id}            Delete session + data

POST   /api/documents/upload                 Upload + process document
DELETE /api/documents/{session_id}           Delete session index data

POST   /api/query                            RAG query (retrieve + generate + metrics)
POST   /api/retrieve                         Retrieve chunks only (no LLM)

GET    /api/sessions/{session_id}/conversation
DELETE /api/sessions/{session_id}/conversation

POST   /api/evaluation/manual               Reference-based evaluation
POST   /api/evaluation/retrieval            Precision/Recall/F1
POST   /api/evaluation/rag                  BLEU/ROUGE/overlap
```

---

## Troubleshooting

**`Cannot connect to backend`** — Make sure the FastAPI server is running on port 8000 before starting Streamlit.

**`FAISS index not found`** — Upload and process a document first. The index is session-specific; resetting the session clears it.

**`GROQ_API_KEY not configured`** — Add the key to a `.env` file in the project root or export it as an environment variable.

**CUDA out of memory** — Reduce `BATCH_SIZE` in `.env` (try 2 or 4). On CPU-only machines everything still works, just slower.

**BERTScore / METEOR unavailable** — These are optional dependencies. The system falls back gracefully: BERTScore falls back to Token F1, METEOR is skipped. Install `bert-score` and run `python -m nltk.downloader wordnet omw-1.4` to enable them.

**Elasticsearch retriever** — Not included in this version. Use `biobert`, `medcpt`, `bm25`, or `hybrid`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
