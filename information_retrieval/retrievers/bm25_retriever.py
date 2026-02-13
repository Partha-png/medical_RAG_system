"""
BM25 Retriever - Traditional keyword-based search
"""
import os
import pickle
from typing import List
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 retriever for keyword-based document retrieval"""
    
    def __init__(self, faiss_dir: str):
        """
        Initialize BM25 retriever
        
        Args:
            faiss_dir: Directory containing BM25 index and metadata
        """
        self.faiss_dir = faiss_dir
        
        # Load BM25 index
        bm25_index_path = os.path.join(faiss_dir, "bm25_index.pkl")
        if not os.path.exists(bm25_index_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_index_path}")
        
        with open(bm25_index_path, "rb") as f:
            index_data = pickle.load(f)
            self.bm25 = index_data["bm25"]
            self.documents = index_data["documents"]
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        # Tokenize query (simple whitespace tokenization)
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Return corresponding documents
        return [self.documents[i] for i in top_k_indices]


def create_bm25_index(documents: List[str], output_dir: str):
    """
    Create and save BM25 index
    
    Args:
        documents: List of document texts
        output_dir: Directory to save index
    """
    # Tokenize documents (simple whitespace tokenization)
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Save index and documents
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "bm25_index.pkl")
    
    with open(index_path, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "documents": documents
        }, f)
    
    print(f"BM25 index created with {len(documents)} documents")
    print(f"Saved to: {index_path}")


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        "Diabetes is a metabolic disease characterized by high blood sugar.",
        "Common symptoms include increased thirst and frequent urination.",
        "Treatment involves insulin therapy and lifestyle modifications."
    ]
    
    # Create index
    create_bm25_index(sample_docs, "test_bm25")
    
    # Test retrieval
    retriever = BM25Retriever("test_bm25")
    results = retriever.retrieve("diabetes symptoms", k=2)
    print("\nRetrieved documents:")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")
