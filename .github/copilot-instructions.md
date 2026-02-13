# Medical RAG System - AI Agent Instructions

## Project Overview
Medical RAG (Retrieval-Augmented Generation) System for answering medical questions using document embedding + LLM generation. Uses Streamlit UI with dual-encoder support (BioBERT and MedCPT) and FAISS vector indexing.

## Architecture & Data Flow

### Core Pipeline
```
Document Upload (PDF/TXT) 
  → Chunking (token-based, 150 tokens, 100 overlap) 
  → Embedding (BioBERT or MedCPT) 
  → FAISS Index Storage 
  → Query Retrieval (k nearest chunks)
  → LLM Generation (Groq API)
```

### Key Components
- **`information_retrieval/document_encoding/`**: Document processing pipeline
  - `encoder.py`: Main orchestrator - reads files, chunks text, batches embeddings
  - `bioBERT_encoder.py` / `medcpt_encoder.py`: Dual encoder implementations (L2 norm, mean pooling)
  - `chunker.py`: Token-aware chunking with sentence-level overlap
  - `faiss_manager.py`: Vector index management
- **`rag_system/`**: Retrieval and generation
  - `rag_pipeline.py`: `medicalrag` class - orchestrates retrieval + Groq LLM
  - `biobertretriever.py` / `medcptretriever.py`: FAISS similarity search
  - `*queryencoder.py`: Query-specific encoders
- **`app.py`**: Streamlit UI (file upload, processing, chat interface)

## Critical Patterns & Conventions

### Encoding Pattern
Both BioBERT and MedCPT follow identical structure:
1. Tokenize with `max_length=512` (truncation + padding)
2. Mean pooling over attention mask (not CLS token)
3. L2 normalization: `F.normalize(embeddings, p=2, dim=1)`
4. Return numpy array
**When adding encoders**: Ensure L2 normalization for consistent FAISS similarity.

### Chunking Strategy
- Split on `.` (sentence boundaries), not fixed-size windows
- Max 150 tokens per chunk, 100-token overlap via last 2 sentences
- Preserves medical context across sentence splits
- Used in: `token_chunk()` in [chunker.py](information_retrieval/document_encoding/chunker.py#L1)

### File Organization
- FAISS indices stored: `information_retrieval/faiss_container/{biobert,medcpt}index.faiss`
- Metadata (chunk-to-text mappings): `*metadata.pkl` (pickle format)
- Temp uploads: `temp_uploads/` (cleanup not implemented - consider adding)
- Embeddings cached: `.npy` files for debugging

### LLM Integration
- **Provider**: Groq API (via `groq` Python client)
- **Model**: `openai/gpt-oss-120b` (OSS distilled model, not actual OpenAI)
- **Temperature**: 0.1 (highly deterministic for medical context)
- **Max tokens**: 500
- **Critical system prompt**: *"Use ONLY the retrieved medical documents to answer"* (prevents hallucination)
- API key: `GROQ_API_KEY` from `.env`

### Session State Management
Streamlit session state keys (in [app.py](app.py#L14)):
- `encoded`: Document processed flag
- `rag_instance`: Active `medicalrag` object
- `chat_history`: Chat messages (list of dicts with `role`/`content`)
- `file_processed`: Filename for display

## Development Workflow

### Setup
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env with GROQ_API_KEY
echo GROQ_API_KEY=your_key > .env
```

### Running
```powershell
# Streamlit app (hot-reloads on code changes)
streamlit run app.py

# Unit testing encoders
python information_retrieval/document_encoding/bioBERT_encoder.py
python information_retrieval/document_encoding/medcpt_encoder.py
python information_retrieval/document_encoding/chunker.py
```

### Common Tasks
1. **Add new encoder**: Mirror BioBERT structure in `information_retrieval/document_encoding/new_encoder.py`, add case in `encoder.py` and `rag_pipeline.py`
2. **Adjust chunking**: Edit `max_tokens` and `overlap` in `token_chunk()` - monitor FAISS index size
3. **Change LLM**: Update `self.model` in `rag_pipeline.py` and `temperature`/`max_tokens` if needed
4. **Debug FAISS**: App includes path inspection in error handler; check `FAISS_DIR/` contents

## External Dependencies
- **Models**: Hugging Face (BioBERT, MedCPT) - auto-downloaded on first use
- **GPU**: CUDA hardcoded in encoders (`self.device = "cuda"`) - modify for CPU
- **Vector DB**: FAISS (CPU version from `faiss-cpu==1.8.0`)
- **LLM API**: Groq (rate limits unknown, no retry logic implemented)

## Integration Points & Gotchas
1. **FAISS persistence**: Only works if metadata pickle exists alongside index - check both before retrieval
2. **Batch encoding**: Large documents need `batch_size` tuning (default 8) to avoid OOM
3. **Temperature mismatch**: Groq may use different temperature scaling than OpenAI
4. **File cleanup**: `temp_uploads/` accumulates files - add cleanup after processing
5. **Model mismatch**: Query encoder must match document encoder (enforced in `rag_pipeline.py`)

## Code Style
- Snake_case for functions/variables, PascalCase for classes
- Type hints present but not comprehensive
- No unit tests (consider adding for encoder outputs)
- Minimal error handling - wrap API calls with try/except
