"""
Evaluation service for RAG metrics
"""
from typing import List, Dict, Optional
from information_retrieval.evaluation.retrieval_metrics import calculate_retrieval_metrics
from information_retrieval.evaluation.rag_metrics import calculate_rag_metrics
from information_retrieval.evaluation.auto_metrics import calculate_auto_metrics


class EvaluationService:
    """Business logic for evaluation metrics"""

    def auto_evaluate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[str],
    ) -> Dict:
        """Reference-free metrics computed automatically per query."""
        try:
            return calculate_auto_metrics(question, answer, retrieved_chunks or [])
        except Exception as e:
            return {"error": str(e)}

    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
    ) -> Dict[str, float]:
        try:
            return calculate_retrieval_metrics(retrieved_docs, relevant_docs)
        except Exception as e:
            return {"error": str(e)}

    def evaluate_rag(
        self,
        generated_answer: str,
        reference_answer: str,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        try:
            return calculate_rag_metrics(
                generated_answer,
                reference_answer,
                retrieved_chunks,
            )
        except Exception as e:
            return {"error": str(e)}

    def batch_evaluate(
        self,
        queries: List[str],
        retrieved_docs_list: List[List[str]],
        generated_answers: List[str],
        relevant_docs_list: Optional[List[List[str]]] = None,
        reference_answers: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        results = {"per_query": [], "aggregated": {}}

        for i, query in enumerate(queries):
            query_result = {
                "query": query,
                "retrieval_metrics": {},
                "rag_metrics": {},
            }

            if relevant_docs_list and i < len(relevant_docs_list):
                query_result["retrieval_metrics"] = self.evaluate_retrieval(
                    retrieved_docs_list[i],
                    relevant_docs_list[i],
                )

            if reference_answers and i < len(reference_answers):
                query_result["rag_metrics"] = self.evaluate_rag(
                    generated_answers[i],
                    reference_answers[i],
                    retrieved_docs_list[i] if i < len(retrieved_docs_list) else None,
                )

            results["per_query"].append(query_result)

        results["aggregated"] = {"note": "Aggregated metrics not yet implemented"}
        return results
