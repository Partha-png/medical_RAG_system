# 🏥 Medical RAG System

> Production-ready Retrieval-Augmented Generation system for medical document analysis with zero hallucinations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)


## 🎯 What is This?

A medical question-answering system that **never hallucinates** because it only uses information from your uploaded documents. Upload medical PDFs, ask questions, get accurate answers with source citations.

**Why this matters:** Traditional LLMs hallucinate medical information. This system uses Retrieval-Augmented Generation (RAG) to ensure every answer comes directly from your source documents.

## ✨ Features

- 🎯 **5 Retrieval Methods**
  - BioBERT (semantic search, medical-specific)
  - MedCPT (clinical trial corpus)
  - BM25 (keyword-based, no GPU needed)
  - Hybrid (combines BioBERT + BM25)
  - Elasticsearch (scalable production)

- 🔒 **Zero Hallucination Architecture**
  - Retrieval-only answers
  - Source citation tracking
  - Document-grounded responses

- 🚀 **Production Ready**
  - FastAPI backend with auto-docs
  - Session management
  - Conversation history
  - Multi-user support
  - Comprehensive error handling

- 💻 **Developer Friendly**
  - Clean architecture
  - Extensive documentation
  - Easy deployment
  - Evaluation metrics included

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│   PDF/TXT   │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Chunker   │─────▶│   Encoder    │
│ (150 tokens)│      │ BioBERT/MedCPT│
└─────────────┘      └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ FAISS Index  │
                     │   Storage    │
                     └──────┬───────┘
                            │
       ┌────────────────────┴────────────────────┐
       │                                         │
       ▼                                         ▼
┌─────────────┐                          ┌─────────────┐
│   Query     │                          │  Retrieval  │
│  Encoding   │─────────────────────────▶│  (Top-K)    │
└─────────────┘                          └──────┬──────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  LLM (Groq)  │
                                         │  Generation  │
                                         └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │    Answer    │
                                         │ w/ Citations │
                                         └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Virtual environment (recommended)
- Groq API Key ([Get one free](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-rag.git
cd medical-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Running the Application

**Start Backend (Terminal 1):**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Start Frontend (Terminal 2):**
```bash
streamlit run app.py
```

## Access the Application

- **Frontend (Streamlit):** http://localhost:8501
- **Backend API (FastAPI):** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs

## Retriever Options

The system supports 5 retrieval methods:

1. **BioBERT** - Medical domain semantic search
2. **MedCPT** - Clinical document specialized search
3. **BM25** - Fast keyword-based search (no GPU required)
4. **Hybrid** - Combines BioBERT + BM25 for best results
5. **Elasticsearch** - Requires Elasticsearch server running on localhost:9200

## Note on Elasticsearch

To use Elasticsearch retriever:
1. Download and install Elasticsearch: https://www.elastic.co/downloads/elasticsearch
2. Start Elasticsearch server on port 9200
3. Select "elasticsearch" encoder in the UI

For testing without Elasticsearch server, use **BioBERT**, **MedCPT**, **BM25**, or **Hybrid** retrievers.

## Project Structure

```
medical_rag/
├── backend/              # FastAPI backend
│   ├── api/             # API routes
│   ├── core/            # Config & exceptions
│   ├── models/          # Pydantic models
│   ├── repositories/    # Data persistence
│   └── services/        # Business logic
├── information_retrieval/
│   ├── document_encoding/  # Document processing
│   └── retrievers/         # 5 retrieval implementations
├── app.py               # Streamlit frontend
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## Deployment Tips

- Use `gunicorn` for production FastAPI deployment
- Set `--reload` flag to False in production
- Use a reverse proxy (nginx) for frontend
- Store FAISS indices on persistent storage
- Monitor memory usage (ML models are heavy)
- Use GPU if available for faster encoding

## Troubleshooting

**Import Error:** Ensure all dependencies are installed
**CUDA Error:** System auto-detects CPU/GPU, will use CPU if no GPU available
**Connection Error:** Ensure both backend and frontend are running
**Elasticsearch Error:** Either install ES server or use a different retriever