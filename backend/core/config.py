"""
Simple configuration for Medical RAG Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"
FAISS_DIR = PROJECT_ROOT / "information_retrieval" / "faiss_container"
FAISS_SESSIONS_DIR = FAISS_DIR / "sessions"

# Create directories if they don't exist
for directory in [DATA_DIR, SESSIONS_DIR, CONVERSATIONS_DIR, TEMP_UPLOADS_DIR, FAISS_DIR, FAISS_SESSIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
API_TITLE = "Medical RAG API"
API_VERSION = "1.0.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
FRONTEND_PORT = _env_int("FRONTEND_PORT", 5000)

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 500)

# Encoder Configuration
DEFAULT_ENCODER = os.getenv("DEFAULT_ENCODER", "biobert")
BATCH_SIZE = _env_int("BATCH_SIZE", 8)

# Chunking Configuration
CHUNK_MAX_TOKENS = _env_int("CHUNK_MAX_TOKENS", 150)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)

# Retrieval Configuration
TOP_K_CHUNKS = _env_int("TOP_K_CHUNKS", 3)
