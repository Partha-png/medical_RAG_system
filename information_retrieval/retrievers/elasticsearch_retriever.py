"""
Elasticsearch Retriever - Scalable search engine integration
"""
import os
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch


class ElasticsearchRetriever:
    """Elasticsearch retriever for scalable document search"""
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "medical_documents",
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False
    ):
        """
        Initialize Elasticsearch retriever
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Name of the index to search
            username: Optional username for authentication
            password: Optional password for authentication
            use_ssl: Whether to use SSL connection
        """
        self.index_name = index_name
        
        # Build connection URL
        protocol = "https" if use_ssl else "http"
        
        # Connect to Elasticsearch
        if username and password:
            self.es = Elasticsearch(
                [f"{protocol}://{host}:{port}"],
                basic_auth=(username, password),
                verify_certs=False if use_ssl else True
            )
        else:
            self.es = Elasticsearch([f"{protocol}://{host}:{port}"])
        
        # Check connection
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {host}:{port}")
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document texts
        """
        # Perform search
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "title"],
                        "type": "best_fields"
                    }
                },
                "size": k
            }
        )
        
        # Extract document texts
        documents = []
        for hit in response["hits"]["hits"]:
            documents.append(hit["_source"]["content"])
        
        return documents
    
    def create_index(self, documents: List[Dict[str, str]]):
        """
        Create index and add documents
        
        Args:
            documents: List of documents, each with 'content' and optional 'title'
        """
        # Create index if it doesn't exist
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "title": {"type": "text"}
                        }
                    }
                }
            )
            print(f"Created index: {self.index_name}")
        
        # Index documents
        for i, doc in enumerate(documents):
            self.es.index(
                index=self.index_name,
                id=i,
                body=doc
            )
        
        # Refresh index to make documents searchable
        self.es.indices.refresh(index=self.index_name)
        print(f"Indexed {len(documents)} documents")
    
    def delete_index(self):
        """Delete the index"""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"Deleted index: {self.index_name}")


if __name__ == "__main__":
    # Example usage (requires Elasticsearch running)
    try:
        retriever = ElasticsearchRetriever(
            host="localhost",
            port=9200,
            index_name="test_medical_docs"
        )
        
        # Create sample documents
        sample_docs = [
            {
                "title": "Diabetes Overview",
                "content": "Diabetes is a metabolic disease characterized by high blood sugar."
            },
            {
                "title": "Diabetes Symptoms",
                "content": "Common symptoms include increased thirst and frequent urination."
            },
            {
                "title": "Diabetes Treatment",
                "content": "Treatment involves insulin therapy and lifestyle modifications."
            }
        ]
        
        # Create index and add documents
        retriever.create_index(sample_docs)
        
        # Test retrieval
        results = retriever.retrieve("diabetes symptoms", k=2)
        print("\nRetrieved documents:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc}")
        
        # Clean up
        retriever.delete_index()
        
    except ConnectionError as e:
        print(f"Error: {e}")
        print("Make sure Elasticsearch is running on localhost:9200")
