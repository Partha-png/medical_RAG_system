"""
Evaluation service for RAG metrics
"""
from typing import List, Dict
from information_retrieval.evaluation.retrieval_metrics import calculate_retrieval_metrics
from information_retrieval.evaluation.rag_metrics import calculate_rag_metrics


class EvaluationService:
    """Business logic for evaluation metrics"""
    
    def evaluate_retrieval(
        self, 
        retrieved_docs: List[str], 
        relevant_docs: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of ground truth relevant document IDs
            
        Returns:
            dict with precision, recall, f1, etc.
        """
        try:
            metrics = calculate_retrieval_metrics(retrieved_docs, relevant_docs)
            return metrics
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_rag(
        self, 
        generated_answer: str, 
        reference_answer: str,
        retrieved_chunks: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG answer quality
        
        Args:
            generated_answer: Generated answer from RAG
            reference_answer: Ground truth reference answer
            retrieved_chunks: Optional retrieved chunks for context evaluation
            
        Returns:
            dict with BLEU, ROUGE, etc.
        """
        try:
            metrics = calculate_rag_metrics(
                generated_answer, 
                reference_answer,
                retrieved_chunks
            )
            return metrics
        except Exception as e:
            return {"error": str(e)}
    
    def batch_evaluate(
        self, 
        queries: List[str],
        retrieved_docs_list: List[List[str]],
        generated_answers: List[str],
        relevant_docs_list: List[List[str]] = None,
        reference_answers: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Batch evaluation for multiple queries
        
        Returns:
            dict with per-query and aggregated metrics
        """
        results = {
            "per_query": [],
            "aggregated": {}
        }
        
        for i, query in enumerate(queries):
            query_result = {
                "query": query,
                "retrieval_metrics": {},
                "rag_metrics": {}
            }
            
            # Evaluate retrieval if ground truth available
            if relevant_docs_list and i < len(relevant_docs_list):
                query_result["retrieval_metrics"] = self.evaluate_retrieval(
                    retrieved_docs_list[i], 
                    relevant_docs_list[i]
                )
            
            # Evaluate RAG if reference available
            if reference_answers and i < len(reference_answers):
                query_result["rag_metrics"] = self.evaluate_rag(
                    generated_answers[i],
                    reference_answers[i],
                    retrieved_docs_list[i] if i < len(retrieved_docs_list) else None
                )
            
            results["per_query"].append(query_result)
        
        # TODO: Compute aggregated metrics (mean, std, etc.)
        results["aggregated"] = {"note": "Aggregated metrics not yet implemented"}
        
        return results
