"""
Hybrid Retriever - Combines dense and sparse retrieval using Reciprocal
Rank Fusion (RRF) so the merge respects both ranked lists without needing
the underlying score scales to be comparable.
"""
from typing import List
from .biobertretriever import BioBERTRetriever
from .medcptretriever import MedCPTRetriever
from .bm25_retriever import BM25Retriever


# RRF constant from Cormack et al. 2009. 60 is the canonical default.
_RRF_K = 60


class HybridRetriever:
    """Hybrid retriever combining dense (semantic) and sparse (keyword) retrieval."""

    def __init__(
        self,
        faiss_dir: str,
        dense_model: str = "biobert",
        alpha: float = 0.5,
    ):
        """
        Args:
            faiss_dir: Directory containing FAISS and BM25 indices.
            dense_model: 'biobert' or 'medcpt'.
            alpha: Weight for dense list (1 - alpha goes to sparse). 0..1.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        self.faiss_dir = faiss_dir
        self.alpha = alpha

        if dense_model.lower() == "biobert":
            self.dense_retriever = BioBERTRetriever(faiss_dir)
        elif dense_model.lower() == "medcpt":
            self.dense_retriever = MedCPTRetriever(faiss_dir)
        else:
            raise ValueError(f"Unsupported dense model: {dense_model}")

        try:
            self.sparse_retriever = BM25Retriever(faiss_dir)
        except FileNotFoundError:
            print("Warning: BM25 index not found. Hybrid retrieval will use dense-only.")
            self.sparse_retriever = None

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if self.sparse_retriever is None:
            return self.dense_retriever.retrieve(query, k=k)

        k_candidates = max(k * 3, k)

        dense_results = self.dense_retriever.retrieve(query, k=k_candidates)
        sparse_results = self.sparse_retriever.retrieve(query, k=k_candidates)

        # Reciprocal Rank Fusion, weighted by alpha.
        scores: dict = {}
        order: list = []

        def _add(results, weight):
            for rank, doc in enumerate(results):
                contribution = weight / (_RRF_K + rank + 1)
                if doc in scores:
                    scores[doc] += contribution
                else:
                    scores[doc] = contribution
                    order.append(doc)

        _add(dense_results, self.alpha)
        _add(sparse_results, 1.0 - self.alpha)

        ranked = sorted(order, key=lambda d: scores[d], reverse=True)
        return ranked[:k]

    def set_alpha(self, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha
