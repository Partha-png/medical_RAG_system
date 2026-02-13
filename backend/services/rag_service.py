"""
RAG service for query processing
"""
import os
from typing import List, Optional
from pathlib import Path
from groq import Groq
from backend.core import config
from backend.core.exceptions import RetrievalError, LLMError
from information_retrieval.retrievers.biobertretriever import BioBERTRetriever
from information_retrieval.retrievers.medcptretriever import MedCPTRetriever


class RAGService:
    """Business logic for RAG query processing"""
    
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
                # Simple initialization with just api_key (works with Groq v1.0.0+)
                self._client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                raise ValueError(f"Failed to initialize Groq client: {str(e)}")
        return self._client
    
    def get_retriever(self, session_id: str, encoder_type: str):
        """Get the appropriate retriever for a session"""
        faiss_dir = config.FAISS_SESSIONS_DIR / session_id
        
        if not faiss_dir.exists():
            raise RetrievalError(f"No index found for session {session_id}")
        
        try:
            encoder_type_lower = encoder_type.lower()
            
            if encoder_type_lower == "biobert":
                return BioBERTRetriever(str(faiss_dir))
            elif encoder_type_lower == "medcpt":
                return MedCPTRetriever(str(faiss_dir))
            elif encoder_type_lower == "bm25":
                from information_retrieval.retrievers.bm25_retriever import BM25Retriever
                return BM25Retriever(str(faiss_dir))
            elif encoder_type_lower == "elasticsearch":
                from information_retrieval.retrievers.elasticsearch_retriever import ElasticsearchRetriever
                return ElasticsearchRetriever(
                    host=config.ELASTICSEARCH_HOST,
                    port=config.ELASTICSEARCH_PORT,
                    index_name=f"session_{session_id}"
                )
            elif encoder_type_lower == "hybrid":
                from information_retrieval.retrievers.hybrid_retriever import HybridRetriever
                # Hybrid uses both dense and sparse, default to biobert for dense
                return HybridRetriever(
                    faiss_dir=str(faiss_dir),
                    dense_model="biobert",
                    alpha=0.5  # Equal weight to dense and sparse
                )
            else:
                raise RetrievalError(f"Unsupported encoder type: {encoder_type}")
        except Exception as e:
            raise RetrievalError(f"Failed to initialize retriever: {str(e)}")
    
    def retrieve_chunks(self, session_id: str, encoder_type: str, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant chunks for a query"""
        try:
            retriever = self.get_retriever(session_id, encoder_type)
            chunks = retriever.retrieve(query, k=k)
            return chunks
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {str(e)}")
    
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
            raise LLMError(f"Answer generation failed: {str(e)}")
    
    def query(self, session_id: str, encoder_type: str, question: str, k: int = 3) -> tuple[str, List[str]]:
        """
        Full RAG query: retrieve chunks and generate answer
        
        Returns:
            tuple of (answer, chunks)
        """
        chunks = self.retrieve_chunks(session_id, encoder_type, question, k)
        answer = self.generate_answer(question, chunks)
        return answer, chunks
    
    def _format_chunks(self, chunks: List[str]) -> str:
        """Format retrieved chunks for the prompt"""
        formatted = []
        for idx, chunk in enumerate(chunks, 1):
            formatted.append(f"[DOC {idx}]\n{chunk}\n")
        return "\n".join(formatted)
