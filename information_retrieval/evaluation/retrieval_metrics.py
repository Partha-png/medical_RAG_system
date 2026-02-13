"""
Retrieval evaluation metrics
"""
from typing import List, Dict


def calculate_retrieval_metrics(
    retrieved_docs: List[str], 
    relevant_docs: List[str]
) -> Dict[str, float]:
    """
    Calculate standard retrieval metrics
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of ground truth relevant document IDs
        
    Returns:
        Dict with precision, recall, f1, and other metrics
    """
    if not retrieved_docs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_retrieved": 0,
            "num_relevant": len(relevant_docs)
        }
    
    if not relevant_docs:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": 0
        }
    
    # Convert to sets for intersection
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    # True positives
    tp = len(retrieved_set.intersection(relevant_set))
    
    # Precision: TP / (TP + FP) = TP / Retrieved
    precision = tp / len(retrieved_docs) if retrieved_docs else 0.0
    
    # Recall: TP / (TP + FN) = TP / Relevant
    recall = tp / len(relevant_docs) if relevant_docs else 0.0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_retrieved": len(retrieved_docs),
        "num_relevant": len(relevant_docs),
        "num_correct": tp
    }
