import os
import numpy as np
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertForMaskedLM
from rouge_score import rouge_scorer
from collections import defaultdict
import json
from datetime import datetime


class RAGEvaluator:
    """
    Comprehensive evaluation system for RAG pipeline.
    Calculates Perplexity, BERT Score, ROUGE scores, and Retrieval metrics.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize the evaluator with required models.
        
        Args:
            model_name: Model for perplexity calculation
            device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize BERT for perplexity
        print("Loading BERT model for perplexity...")
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.bert_model.eval()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Storage for evaluation results
        self.evaluation_history = []
        
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of generated text using BERT.
        Lower perplexity indicates better text quality.
        
        Args:
            text: Generated answer text
            
        Returns:
            Perplexity score
        """
        try:
            # Tokenize
            encodings = self.bert_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**encodings, labels=encodings['input_ids'])
                loss = outputs.loss
                
            perplexity = torch.exp(loss).item()
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and generated text.
        
        Args:
            reference: Ground truth or reference answer
            generated: Generated answer from RAG
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        try:
            scores = self.rouge_scorer.score(reference, generated)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_retrieval_precision(self, retrieved_chunks: List[str], 
                                     relevant_keywords: List[str]) -> float:
        """
        Calculate precision of retrieved chunks based on keyword presence.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            relevant_keywords: List of keywords that should be in relevant chunks
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_chunks or not relevant_keywords:
            return 0.0
        
        relevant_count = 0
        for chunk in retrieved_chunks:
            chunk_lower = chunk.lower()
            # Check if any keyword is in the chunk
            if any(keyword.lower() in chunk_lower for keyword in relevant_keywords):
                relevant_count += 1
        
        precision = relevant_count / len(retrieved_chunks)
        return precision
    
    def calculate_retrieval_recall(self, retrieved_chunks: List[str], 
                                   relevant_keywords: List[str]) -> float:
        """
        Calculate recall of retrieved chunks based on keyword coverage.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            relevant_keywords: List of keywords that should be retrieved
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_keywords:
            return 0.0
        
        combined_chunks = ' '.join(retrieved_chunks).lower()
        found_keywords = sum(
            1 for keyword in relevant_keywords 
            if keyword.lower() in combined_chunks
        )
        
        recall = found_keywords / len(relevant_keywords)
        return recall
    
    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Calculate semantic similarity between question and answer.
        Uses simple keyword overlap for basic relevance.
        
        Args:
            question: User question
            answer: Generated answer
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword-based relevance
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'when', 
                     'where', 'who', 'how', 'of', 'to', 'in', 'for', 'on', 'with'}
        
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words & answer_words)
        relevance = overlap / len(question_words)
        
        return min(relevance, 1.0)
    
    def evaluate_single_query(self, 
                            question: str, 
                            answer: str, 
                            retrieved_chunks: List[str],
                            reference_answer: str = None,
                            relevant_keywords: List[str] = None) -> Dict:
        """
        Perform comprehensive evaluation for a single query.
        
        Args:
            question: User question
            answer: Generated answer
            retrieved_chunks: Retrieved document chunks
            reference_answer: Optional ground truth answer for ROUGE
            relevant_keywords: Optional keywords for retrieval evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"\nEvaluating query: {question[:50]}...")
        
        metrics = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate perplexity
        print("  Computing perplexity...")
        metrics['perplexity'] = self.calculate_perplexity(answer)
        
        # Calculate ROUGE scores if reference is provided
        if reference_answer:
            print("  Computing ROUGE scores...")
            rouge_scores = self.calculate_rouge_scores(reference_answer, answer)
            metrics.update(rouge_scores)
        
        # Calculate retrieval metrics if keywords are provided
        if relevant_keywords:
            print("  Computing retrieval metrics...")
            metrics['retrieval_precision'] = self.calculate_retrieval_precision(
                retrieved_chunks, relevant_keywords
            )
            metrics['retrieval_recall'] = self.calculate_retrieval_recall(
                retrieved_chunks, relevant_keywords
            )
            metrics['retrieval_f1'] = (
                2 * metrics['retrieval_precision'] * metrics['retrieval_recall'] /
                (metrics['retrieval_precision'] + metrics['retrieval_recall'])
                if (metrics['retrieval_precision'] + metrics['retrieval_recall']) > 0 else 0.0
            )
        
        # Calculate answer relevance
        print("  Computing answer relevance...")
        metrics['answer_relevance'] = self.calculate_answer_relevance(question, answer)
        
        # Store in history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def calculate_average_metrics(self) -> Dict:
        """
        Calculate average metrics across all evaluated queries.
        
        Returns:
            Dictionary with average scores for all metrics
        """
        if not self.evaluation_history:
            return {}
        
        # Initialize accumulator
        averages = defaultdict(list)
        
        # Collect all metric values
        for result in self.evaluation_history:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['timestamp']:
                    averages[key].append(value)
        
        # Calculate means
        avg_metrics = {
            key: np.mean(values) 
            for key, values in averages.items()
        }
        
        avg_metrics['num_queries'] = len(self.evaluation_history)
        
        return avg_metrics
    
    def print_evaluation_summary(self, metrics: Dict):
        """
        Print a formatted summary of evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📝 Question: {metrics.get('question', 'N/A')}")
        print(f"\n💬 Answer: {metrics.get('answer', 'N/A')[:200]}...")
        
        print(f"\n📊 METRICS:")
        print(f"  • Perplexity: {metrics.get('perplexity', 'N/A'):.2f}")
        
        if 'rouge1' in metrics:
            print(f"  • ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"  • ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"  • ROUGE-L: {metrics['rougeL']:.4f}")
        
        if 'retrieval_precision' in metrics:
            print(f"  • Retrieval Precision: {metrics['retrieval_precision']:.4f}")
            print(f"  • Retrieval Recall: {metrics['retrieval_recall']:.4f}")
            print(f"  • Retrieval F1: {metrics['retrieval_f1']:.4f}")
        
        print(f"  • Answer Relevance: {metrics.get('answer_relevance', 'N/A'):.4f}")
        print("="*60 + "\n")
    
    def print_average_summary(self):
        """Print summary of average metrics across all queries."""
        avg_metrics = self.calculate_average_metrics()
        
        if not avg_metrics:
            print("No evaluation history available.")
            return
        
        print("\n" + "="*60)
        print("AVERAGE EVALUATION METRICS")
        print("="*60)
        print(f"\nTotal Queries Evaluated: {avg_metrics.get('num_queries', 0)}")
        
        print(f"\n📊 AVERAGE SCORES:")
        print(f"  • Perplexity: {avg_metrics.get('perplexity', 'N/A'):.2f}")
        
        if 'rouge1' in avg_metrics:
            print(f"  • ROUGE-1: {avg_metrics['rouge1']:.4f}")
            print(f"  • ROUGE-2: {avg_metrics['rouge2']:.4f}")
            print(f"  • ROUGE-L: {avg_metrics['rougeL']:.4f}")
        
        if 'retrieval_precision' in avg_metrics:
            print(f"  • Retrieval Precision: {avg_metrics['retrieval_precision']:.4f}")
            print(f"  • Retrieval Recall: {avg_metrics['retrieval_recall']:.4f}")
            print(f"  • Retrieval F1: {avg_metrics['retrieval_f1']:.4f}")
        
        print(f"  • Answer Relevance: {avg_metrics.get('answer_relevance', 'N/A'):.4f}")
        print("="*60 + "\n")
        
        return avg_metrics
    
    def save_results(self, filepath: str):
        """
        Save evaluation history to JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump({
                'evaluation_history': self.evaluation_history,
                'average_metrics': self.calculate_average_metrics()
            }, f, indent=2)
        
        print(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    from rag_system.langchain_rag import MedicalRAGCore
    
    # Initialize RAG system
    rag = MedicalRAGCore(
        model_type="biobert",
        faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container"
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Test queries with optional reference answers and keywords
    test_cases = [
        {
            "question": "What are the symptoms of diabetes?",
            "reference_answer": "Diabetes symptoms include elevated blood glucose, frequent urination, increased thirst, fatigue, and blurred vision.",
            "keywords": ["diabetes", "glucose", "symptoms", "blood"]
        },
        {
            "question": "How is hypertension treated?",
            "reference_answer": "Hypertension is treated with lifestyle modifications, dietary changes, and medications like ACE inhibitors and beta-blockers.",
            "keywords": ["hypertension", "treatment", "ACE inhibitors", "beta-blockers"]
        },
        {
            "question": "What causes asthma?",
            "reference_answer": "Asthma is caused by airway inflammation triggered by allergens, infections, pollutants, and exercise.",
            "keywords": ["asthma", "airway", "inflammation", "allergens"]
        }
    ]
    
    # Evaluate each query
    for test_case in test_cases:
        # Get RAG response
        result = rag.ask(test_case["question"], k=3)
        
        # Evaluate
        metrics = evaluator.evaluate_single_query(
            question=test_case["question"],
            answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
            reference_answer=test_case.get("reference_answer"),
            relevant_keywords=test_case.get("keywords")
        )
        
        # Print individual results
        evaluator.print_evaluation_summary(metrics)
    
    # Print average metrics
    evaluator.print_average_summary()
    
    # Save results
    evaluator.save_results("evaluation_results.json")