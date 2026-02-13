import numpy as np
from typing import List,Tuple
from sklearn.metrics.pairwise import cosine_similarity

class RAGEvaluator:
    def __init__(self,encoder):
        self.encoder=encoder
    def context_precision(self,retrieved_chunks,ground_truth,k=None):
        if not retrieved_chunks:
            return 0.0
        chunks_to_eval=retrieved_chunks[:k] if k else retrieved_chunks
        chunk_embeddings=self.encoder.encode(chunks_to_eval)
        gt_embedding=self.encoder.encode([ground_truth])

        similarities=cosine_similarity(chunk_embeddings,gt_embedding).flatten()

        threshold=0.5
        relevant_count=np.sum(similarities>=threshold)

        return relevant_count/len(chunks_to_eval)
    def context_recall(self,retrieved_chunks,ground_truth):
        if not retrieved_chunks or not ground_truth:
            return 0.0
        chunk_embeddings=self.encoder.encode(retrieved_chunks)
        gt_embedding=self.encoder.encode([ground_truth])

        similarities=cosine_similarity(chunk_embeddings,gt_embedding).flatten()
        max_similarity=np.max(similarities)

        return float(max_similarity)
    def answer_faithfulness(self,retrieved_chunks,ground_truth,answer):
        if not answer:
            return 0.0
        answer_embedding=self.encoder.encode([answer])
        chunk_embeddings=self.encoder.encode(retrieved_chunks)

        similarities=cosine_similarity(answer_embedding,chunk_embeddings).flatten()

        return float(np.mean(similarities))
    def answer_relevance(self,retrieved_chunks,ground_truth,answer,query):
        if not answer or not query:
            return 0.0
        answer_embedding=self.encoder.encode([answer])
        query_embedding=self.encoder.encode([query])

        similarity=cosine_similarity(answer_embedding,query_embedding).flatten()[0][0]

        return float(similarity)
    def semantic_similarity_score(self,text1,text2):
        emb1=self.encoder.encode([text1])
        emb2=self.encoder.encode([text2])
        return float(cosine_similarity(emb1,emb2)[0][0])
    def evaluate_retrieval(self,query,retrieved_chunks,ground_truth,answer):
        metrics={}
        if ground_truth:
            metrics['context_precision']=self.context_precision(retrieved_chunks,ground_truth,k=5)
            metrics['context_recall']=self.context_recall(retrieved_chunks,ground_truth)
            #generation metrics
            if answer:
                metrics['answer_relevance']=self.answer_relevance(answer,query)

                if retrieved_chunks:
                    metrics['answer_faithfulness']=self.answer_faithfulness(
                        answer,retrieved_chunks
                    )
            return metrics
    def calculate_rag_score(self,metrics):
        weights={
            'context_precision': 0.25,
            'context_recall': 0.25,
            'answer_relevance': 0.25,
            'answer_faithfulness': 0.25
        }
        available_metrics={k:v for k,v in metrics.items() if k in weights}
        if not available_metrics:
            return 0.0
        total_weight=sum(weights[k] for k in available_metrics)

        score=sum(
            metrics[k]*(weights[k]/total_weight)
            for k in available_metrics
        )
        return score
    
    def evaluate_batch(
            self,
            queries,
            retrieved_chunks_list,
            answers,
            ground_truths
    ):
        all_metrics = []
        for i, (query, chunks, answer) in enumerate(zip(queries, retrieved_chunks_list, answers)):
            gt = ground_truths[i] if ground_truths else None
            metrics = self.evaluate_retrieval(query, chunks, gt, answer)
            all_metrics.append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        if all_metrics:
            metric_names = set().union(*[m.keys() for m in all_metrics])
            for metric in metric_names:
                values = [m[metric] for m in all_metrics if metric in m]
                avg_metrics[f'avg_{metric}'] = np.mean(values) if values else 0.0
                avg_metrics[f'std_{metric}'] = np.std(values) if values else 0.0
        
        return avg_metrics
    class retrievalanalyzer:
        @staticmethod
        def analyze_retrieval()