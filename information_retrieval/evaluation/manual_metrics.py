"""
Reference-based ("manual") evaluation metrics.

These require ground truth supplied by the user:
  - a `reference_answer` (string)
  - optionally `relevant_passages` (list of strings considered "relevant")

Scores are emitted as floats in [0, 1] (where applicable). Heavy dependencies
(BERTScore, NLTK WordNet for METEOR) are lazy-imported so the rest of the app
keeps working if they aren't installed.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------- Generation metrics ----------------------------------------------

def _bleu(reference: str, generated: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = [_tokens(reference)]
        gen = _tokens(generated)
        if not ref[0] or not gen:
            return 0.0
        return float(sentence_bleu(ref, gen, smoothing_function=SmoothingFunction().method1))
    except Exception:
        return 0.0


def _rouge(reference: str, generated: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = scorer.score(reference, generated)
        return {
            "rouge_1": float(s["rouge1"].fmeasure),
            "rouge_2": float(s["rouge2"].fmeasure),
            "rouge_l": float(s["rougeL"].fmeasure),
        }
    except Exception:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


def _meteor(reference: str, generated: str) -> Optional[float]:
    """METEOR; requires nltk wordnet. Returns None if unavailable."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        # Make sure the small data files are present.
        for pkg, path in [("wordnet", "corpora/wordnet"),
                          ("omw-1.4", "corpora/omw-1.4")]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    return None
        ref = _tokens(reference)
        gen = _tokens(generated)
        if not ref or not gen:
            return 0.0
        return float(meteor_score([ref], gen))
    except Exception:
        return None


def _exact_match(reference: str, generated: str) -> float:
    return 1.0 if (reference or "").strip().lower() == (generated or "").strip().lower() else 0.0


def _token_f1(reference: str, generated: str) -> Dict[str, float]:
    ref = set(_tokens(reference))
    gen = set(_tokens(generated))
    if not ref or not gen:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    tp = len(ref & gen)
    p = tp / len(gen)
    r = tp / len(ref)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"f1": f1, "precision": p, "recall": r}


# Module-level cache so we don't re-download / re-load on every call.
_BERTSCORE = None


def _bertscore(reference: str, generated: str) -> Optional[Dict[str, float]]:
    """BERTScore; lazy imports + caches the scorer. Returns None on failure."""
    global _BERTSCORE
    try:
        if _BERTSCORE is None:
            from bert_score import BERTScorer
            _BERTSCORE = BERTScorer(lang="en", rescale_with_baseline=False)
        P, R, F = _BERTSCORE.score([generated or ""], [reference or ""])
        return {
            "bertscore_p":  float(P[0]),
            "bertscore_r":  float(R[0]),
            "bertscore_f1": float(F[0]),
        }
    except Exception:
        return None


def _completeness(reference: str, generated: str) -> float:
    """Recall of reference *content* tokens in the generated answer."""
    ref_content = {t for t in _tokens(reference) if len(t) > 2}
    if not ref_content:
        return 0.0
    gen = set(_tokens(generated))
    return len(ref_content & gen) / len(ref_content)


# ---------- Retrieval metrics (with ground-truth relevant passages) ---------

def _is_chunk_relevant(chunk: str, relevant_passages: List[str], threshold: float = 0.3) -> bool:
    """A chunk is "relevant" if it has high token overlap with any provided
    relevant passage (Jaccard over content tokens)."""
    chunk_t = {t for t in _tokens(chunk) if len(t) > 2}
    if not chunk_t:
        return False
    for rp in relevant_passages:
        rp_t = {t for t in _tokens(rp) if len(t) > 2}
        if not rp_t:
            continue
        j = len(chunk_t & rp_t) / len(chunk_t | rp_t)
        if j >= threshold:
            return True
    return False


def _retrieval_metrics(
    retrieved_chunks: List[str],
    relevant_passages: List[str],
    k: Optional[int] = None,
) -> Dict[str, float]:
    if k is None:
        k = len(retrieved_chunks)
    top_k = retrieved_chunks[:k] if k else []
    if not top_k or not relevant_passages:
        return {
            "precision_at_k": 0.0,
            "recall_at_k":    0.0,
            "hit_rate":       0.0,
            "mrr":            0.0,
            "ndcg_at_k":      0.0,
        }

    rel_flags = [1 if _is_chunk_relevant(c, relevant_passages) else 0 for c in top_k]
    num_retrieved_relevant = sum(rel_flags)
    total_relevant = len(relevant_passages)

    precision = num_retrieved_relevant / len(top_k)
    recall = num_retrieved_relevant / total_relevant if total_relevant else 0.0
    hit_rate = 1.0 if num_retrieved_relevant > 0 else 0.0

    mrr = 0.0
    for i, flag in enumerate(rel_flags, start=1):
        if flag:
            mrr = 1.0 / i
            break

    # nDCG with binary relevance.
    dcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(rel_flags, start=1))
    ideal_hits = min(total_relevant, len(top_k))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision_at_k": precision,
        "recall_at_k":    recall,
        "hit_rate":       hit_rate,
        "mrr":            mrr,
        "ndcg_at_k":      ndcg,
    }


# ---------- Public entry point ----------------------------------------------

def calculate_manual_metrics(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_chunks: Optional[List[str]] = None,
    relevant_passages: Optional[List[str]] = None,
) -> Dict:
    """Compute all reference-based metrics. Missing inputs => zeros / nulls."""

    rouge = _rouge(reference_answer, generated_answer)
    bleu = _bleu(reference_answer, generated_answer)
    meteor = _meteor(reference_answer, generated_answer)
    em = _exact_match(reference_answer, generated_answer)
    f1 = _token_f1(reference_answer, generated_answer)
    bert = _bertscore(reference_answer, generated_answer)
    completeness = _completeness(reference_answer, generated_answer)

    # answer_correctness: prefer BERTScore F1 (semantic); fall back to token F1.
    if bert and "bertscore_f1" in bert:
        answer_correctness = bert["bertscore_f1"]
    else:
        answer_correctness = f1["f1"]

    generation = {
        "rouge_1":            round(rouge["rouge_1"], 4),
        "rouge_2":            round(rouge["rouge_2"], 4),
        "rouge_l":            round(rouge["rouge_l"], 4),
        "bleu":               round(bleu, 4),
        "meteor":             round(meteor, 4) if meteor is not None else None,
        "exact_match":        em,
        "f1":                 round(f1["f1"], 4),
        "bertscore_f1":       round(bert["bertscore_f1"], 4) if bert else None,
        "answer_correctness": round(answer_correctness, 4),
        "completeness":       round(completeness, 4),
    }

    retrieval: Dict = {}
    if retrieved_chunks and relevant_passages:
        rm = _retrieval_metrics(retrieved_chunks, relevant_passages, k=len(retrieved_chunks))
        retrieval = {k: round(v, 4) for k, v in rm.items()}

    return {
        "generation": generation,
        "retrieval":  retrieval,
        "bertscore_available": bert is not None,
        "meteor_available":    meteor is not None,
    }
