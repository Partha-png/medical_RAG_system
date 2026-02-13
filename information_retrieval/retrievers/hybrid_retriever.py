"""
Hybrid Retriever - Combines dense and sparse retrieval
"""
import numpy as np
from typing import List, Optional
from .biobertretriever import BioBERTRetriever
from .medcptretriever import MedCPTRetriever
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    Hybrid retriever combining dense (semantic) and sparse (keyword) retrieval
    """
    
    def __init__(
        self, 
        faiss_dir: str,
        dense_model: str = "biobert",
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever
        
        Args:
            faiss_dir: Directory containing FAISS and BM25 indices
            dense_model: Dense retriever to use ('biobert' or 'medcpt')
            alpha: Weight for dense retrieval (1-alpha for sparse)
                   alpha=1.0 means pure dense, alpha=0.0 means pure sparse
        """
        self.faiss_dir = faiss_dir
        self.alpha = alpha
        
        # Initialize dense retriever
        if dense_model.lower() == "biobert":
            self.dense_retriever = BioBERTRetriever(faiss_dir)
        elif dense_model.lower() == "medcpt":
            self.dense_retriever = MedCPTRetriever(faiss_dir)
        else:
            raise ValueError(f"Unsupported dense model: {dense_model}")
        
        # Initialize sparse retriever (BM25)
        try:
            self.sparse_retriever = BM25Retriever(faiss_dir)
        except FileNotFoundError:
            print("Warning: BM25 index not found. Hybrid retrieval will use dense-only.")
            self.sparse_retriever = None
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k documents using hybrid approach
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        # If no sparse retriever, fall back to dense only
        if self.sparse_retriever is None:
            return self.dense_retriever.retrieve(query, k=k)
        
        # Retrieve more candidates from each retriever
        k_candidates = k * 3  # Get more candidates for reranking
        
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query, k=k_candidates)
        sparse_results = self.sparse_retriever.retrieve(query, k=k_candidates)
        
        # Combine and deduplicate
        all_docs = []
        doc_scores = {}
        
        # Score dense results
        for i, doc in enumerate(dense_results):
            score = (k_candidates - i) / k_candidates  # Higher rank = higher score
            dense_score = score * self.alpha
            doc_scores[doc] = doc_scores.get(doc, 0) + dense_score
            if doc not in all_docs:
                all_docs.append(doc)
        
        # Score sparse results
        for i, doc in enumerate(sparse_results):
            score = (k_candidates - i) / k_candidates
            sparse_score = score * (1 - self.alpha)
            doc_scores[doc] = doc_scores.get(doc, 0) + sparse_score
            if doc not in all_docs:
                all_docs.append(doc)
        
        # Sort by combined score and return top-k
        ranked_docs = sorted(all_docs, key=lambda d: doc_scores[d], reverse=True)
        return ranked_docs[:k]
    
    def set_alpha(self, alpha: float):
        """
        Update the alpha parameter
        
        Args:
            alpha: New weight for dense retrieval (0.0 to 1.0)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha


if __name__ == "__main__":
    # Example usage
    faiss_dir = r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\faiss_container"
    
    try:
        # Create hybrid retriever
        retriever = HybridRetriever(
            faiss_dir=faiss_dir,
            dense_model="biobert",
            alpha=0.5  # Equal weight to dense and sparse
        )
        
        # Test retrieval
        query = "What are the symptoms of diabetes?"
        results = retriever.retrieve(query, k=3)
        
        print("Hybrid retrieval results:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
