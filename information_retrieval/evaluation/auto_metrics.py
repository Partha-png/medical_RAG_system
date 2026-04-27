"""
Reference-free automatic RAG quality metrics.

Computes lightweight quality indicators that do NOT require a ground-truth
answer or a relevance-judged document set. Suitable for surfacing alongside
every generated answer in production.

Three retrieval metrics + three answer metrics are returned, each in [0, 1]
where higher is better.

Retrieval:
  - context_relevance     : mean overlap between question content and each chunk
  - top_chunk_relevance   : best chunk-to-question overlap
  - diversity             : 1 - mean pairwise overlap between retrieved chunks

Answer:
  - faithfulness          : fraction of answer content words supported by chunks
  - answer_relevance      : overlap between question and answer content
  - context_utilization   : fraction of chunks that meaningfully contribute
"""
from typing import Dict, List, Set
import re


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")

# Small inline stopword list -- avoids an nltk download at runtime.
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "of", "for", "to", "in", "on", "at", "by", "with", "as", "and", "or", "but",
    "not", "no", "nor", "this", "that", "these", "those", "it", "its", "itself",
    "from", "into", "onto", "than", "then", "so", "such", "can", "could", "may",
    "might", "will", "would", "should", "shall", "must", "do", "does", "did",
    "done", "doing", "have", "has", "had", "having", "i", "you", "he", "she",
    "they", "we", "them", "us", "me", "him", "her", "my", "your", "his", "their",
    "our", "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "there", "here", "also", "about", "over", "under", "more", "most", "many",
    "much", "some", "any", "all", "each", "other", "another", "one", "two",
    "both", "if", "because", "while", "during", "just", "very", "only", "same",
    "yes", "off", "out", "up", "down", "again", "further", "once",
}


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage(needle: Set[str], haystack: Set[str]) -> float:
    """Fraction of `needle` tokens that also appear in `haystack`."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


def calculate_auto_metrics(
    question: str,
    answer: str,
    chunks: List[str],
) -> Dict:
    """Return reference-free retrieval + answer quality metrics."""
    q_tokens = _content_tokens(question)
    a_tokens = _content_tokens(answer)
    chunk_token_sets = [_content_tokens(c) for c in (chunks or [])]

    # --- Retrieval metrics ---
    if chunk_token_sets and q_tokens:
        chunk_rels = [_coverage(q_tokens, ct) for ct in chunk_token_sets]
        context_relevance = sum(chunk_rels) / len(chunk_rels)
        top_chunk_relevance = max(chunk_rels)
    else:
        context_relevance = 0.0
        top_chunk_relevance = 0.0

    if len(chunk_token_sets) >= 2:
        sims = []
        for i in range(len(chunk_token_sets)):
            for j in range(i + 1, len(chunk_token_sets)):
                sims.append(_jaccard(chunk_token_sets[i], chunk_token_sets[j]))
        diversity = 1.0 - (sum(sims) / len(sims))
    else:
        # A single chunk is trivially "diverse"; report 1.0 to avoid penalising k=1.
        diversity = 1.0 if chunk_token_sets else 0.0

    # --- Answer metrics ---
    chunk_union: Set[str] = set().union(*chunk_token_sets) if chunk_token_sets else set()
    faithfulness = _coverage(a_tokens, chunk_union)
    answer_relevance = _jaccard(q_tokens, a_tokens)

    if chunk_token_sets:
        # A chunk "contributes" if it shares >=3 distinct content words with the answer.
        used = sum(1 for ct in chunk_token_sets if len(ct & a_tokens) >= 3)
        context_utilization = used / len(chunk_token_sets)
    else:
        context_utilization = 0.0

    return {
        "retrieval": {
            "context_relevance": round(context_relevance, 4),
            "top_chunk_relevance": round(top_chunk_relevance, 4),
            "diversity": round(diversity, 4),
        },
        "answer": {
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_relevance, 4),
            "context_utilization": round(context_utilization, 4),
        },
        "counts": {
            "num_chunks": len(chunks or []),
            "answer_words": len(_tokens(answer)),
            "question_words": len(_tokens(question)),
        },
    }
