from typing import List, Dict, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_rag_metrics(
    generated_answer: str,
    reference_answer: str,
    retrieved_chunks: Optional[List[str]] = None
) -> Dict[str, float]:

    gen_words = set(generated_answer.lower().split())
    ref_words = set(reference_answer.lower().split())

    if not gen_words or not ref_words:
        word_overlap = 0.0
    else:
        overlap = len(gen_words.intersection(ref_words))
        word_overlap = overlap / max(len(gen_words), len(ref_words))

    exact_match = 1.0 if generated_answer.strip() == reference_answer.strip() else 0.0
    length_ratio = len(generated_answer) / len(reference_answer) if reference_answer else 0.0

    try:
        reference_tokens = [reference_answer.lower().split()]
        generated_tokens = generated_answer.lower().split()
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothing)
    except Exception:
        bleu = 0.0

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_answer, generated_answer)
        rouge1_p = scores['rouge1'].precision
        rouge1_r = scores['rouge1'].recall
        rouge1_f = scores['rouge1'].fmeasure
        rouge2_p = scores['rouge2'].precision
        rouge2_r = scores['rouge2'].recall
        rouge2_f = scores['rouge2'].fmeasure
        rougeL_p = scores['rougeL'].precision
        rougeL_r = scores['rougeL'].recall
        rougeL_f = scores['rougeL'].fmeasure
    except Exception:
        rouge1_p = rouge1_r = rouge1_f = 0.0
        rouge2_p = rouge2_r = rouge2_f = 0.0
        rougeL_p = rougeL_r = rougeL_f = 0.0

    # Retrieval metrics from chunks
    precision_at_3 = 0.0
    recall_at_3 = 0.0
    f1_at_3 = 0.0

    if retrieved_chunks and reference_answer:
        ref_words_list = set(reference_answer.lower().split())
        matched = 0
        for chunk in retrieved_chunks[:3]:
            chunk_words = set(chunk.lower().split())
            if len(chunk_words.intersection(ref_words_list)) > 5:
                matched += 1
        precision_at_3 = matched / min(3, len(retrieved_chunks))
        recall_at_3 = matched / 3
        if precision_at_3 + recall_at_3 > 0:
            f1_at_3 = 2 * precision_at_3 * recall_at_3 / (precision_at_3 + recall_at_3)

    metrics = {
        "bleu_score":      round(bleu, 4),
        "rouge1_precision": round(rouge1_p, 4),
        "rouge1_recall":   round(rouge1_r, 4),
        "rouge1_f1":       round(rouge1_f, 4),
        "rouge2_precision": round(rouge2_p, 4),
        "rouge2_recall":   round(rouge2_r, 4),
        "rouge2_f1":       round(rouge2_f, 4),
        "rougeL_precision": round(rougeL_p, 4),
        "rougeL_recall":   round(rougeL_r, 4),
        "rougeL_f1":       round(rougeL_f, 4),
        "precision_at_3":  round(precision_at_3, 4),
        "recall_at_3":     round(recall_at_3, 4),
        "f1_at_3":         round(f1_at_3, 4),
        "word_overlap":    round(word_overlap, 4),
        "exact_match":     exact_match,
        "length_ratio":    round(length_ratio, 4),
        "generated_length": len(generated_answer),
        "reference_length": len(reference_answer),
        "num_chunks_used": len(retrieved_chunks) if retrieved_chunks else 0,
    }

    return metrics