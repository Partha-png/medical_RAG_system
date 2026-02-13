# Medical RAG System - Deployment Guide

## Prerequisites

- Python 3.10+ 
- pip
- Virtual environment (recommended)

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application

### Start Backend Server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend (in a separate terminal)
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