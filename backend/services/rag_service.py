"""
RAG service for query processing
"""
import threading
from typing import Dict, List, Tuple
from groq import Groq
from backend.core import config
from backend.core.exceptions import RetrievalError, LLMError
from information_retrieval.retrievers.biobertretriever import BioBERTRetriever
from information_retrieval.retrievers.medcptretriever import MedCPTRetriever


class RAGService:
    """Business logic for RAG query processing"""

    _retriever_cache: Dict[Tuple[str, str], object] = {}
    _cache_lock = threading.Lock()

    def __init__(self):
        self.groq_api_key = config.GROQ_API_KEY
        self.model = config.LLM_MODEL
        self._client = None

    @property
    def client(self):
        """Lazy-load Groq client"""
        if self._client is None:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not configured")
            try:
                self._client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                raise ValueError(f"Failed to initialize Groq client: {str(e)}") from e
        return self._client

    def _build_retriever(self, session_id: str, encoder_type: str):
        faiss_dir = config.FAISS_SESSIONS_DIR / session_id

        if not faiss_dir.exists():
            raise RetrievalError(f"No index found for session {session_id}")

        encoder_type_lower = encoder_type.lower()

        if encoder_type_lower == "biobert":
            return BioBERTRetriever(str(faiss_dir))
        if encoder_type_lower == "medcpt":
            return MedCPTRetriever(str(faiss_dir))
        if encoder_type_lower == "bm25":
            from information_retrieval.retrievers.bm25_retriever import BM25Retriever
            return BM25Retriever(str(faiss_dir))
        if encoder_type_lower == "hybrid":
            from information_retrieval.retrievers.hybrid_retriever import HybridRetriever
            return HybridRetriever(
                faiss_dir=str(faiss_dir),
                dense_model="biobert",
                alpha=0.5,
            )
        raise RetrievalError(f"Unsupported encoder type: {encoder_type}")

    def get_retriever(self, session_id: str, encoder_type: str):
        """Get (and cache) a retriever for a session.

        Caching is essential: each retriever loads hundreds of MB of model
        weights and the FAISS index. Without this, the server OOMs after a
        handful of queries.
        """
        key = (session_id, encoder_type.lower())
        cached = self._retriever_cache.get(key)
        if cached is not None:
            return cached

        with self._cache_lock:
            cached = self._retriever_cache.get(key)
            if cached is not None:
                return cached
            try:
                retriever = self._build_retriever(session_id, encoder_type)
            except RetrievalError:
                raise
            except Exception as e:
                raise RetrievalError(f"Failed to initialize retriever: {str(e)}") from e
            self._retriever_cache[key] = retriever
            return retriever

    @classmethod
    def invalidate_session_cache(cls, session_id: str) -> None:
        """Drop any cached retrievers for a given session id."""
        with cls._cache_lock:
            stale = [k for k in cls._retriever_cache if k[0] == session_id]
            for k in stale:
                cls._retriever_cache.pop(k, None)

    def retrieve_chunks(self, session_id: str, encoder_type: str, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant chunks for a query"""
        try:
            retriever = self.get_retriever(session_id, encoder_type)
            return retriever.retrieve(query, k=k)
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}") from e

    def generate_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answer using LLM with retrieved chunks"""
        try:
            context = self._format_chunks(chunks)

            prompt = f"""ANSWER USING ONLY THE RETRIEVED MEDICAL DOCUMENTS BELOW. PROVIDE A CLEAN, CONCISE ANSWER.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a knowledgeable medical assistant. Use ONLY the retrieved medical documents to answer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise LLMError(f"Answer generation failed: {str(e)}") from e

    def query(self, session_id: str, encoder_type: str, question: str, k: int = 3) -> Tuple[str, List[str]]:
        """Full RAG query: retrieve chunks and generate answer."""
        chunks = self.retrieve_chunks(session_id, encoder_type, question, k)
        answer = self.generate_answer(question, chunks)
        return answer, chunks

    def _format_chunks(self, chunks: List[str]) -> str:
        formatted = []
        for idx, chunk in enumerate(chunks, 1):
            formatted.append(f"[DOC {idx}]\n{chunk}\n")
        return "\n".join(formatted)
