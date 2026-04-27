"""
Token-based document chunker.

Splits text into chunks of at most ``max_tokens`` tokens with a sliding
``overlap`` (in tokens) between consecutive chunks. The overlap is measured in
tokens — not sentences — which guarantees forward progress and avoids the
infinite-loop / tiny-chunk failure mode of a naive sentence-based scheme.
"""
import re

# Sentence boundaries: end-of-sentence punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def token_chunk(text, tokenizer, max_tokens: int = 150, overlap: int = 100):
    """Chunk ``text`` into pieces of at most ``max_tokens`` tokens.

    A trailing window of ``overlap`` tokens from the previous chunk is
    prepended to the next chunk to preserve context across boundaries.
    """
    if not text or not text.strip():
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    overlap = max(0, min(overlap, max_tokens - 1))

    sentences = _split_sentences(text) or [text.strip()]

    chunks = []
    current_tokens = []  # flat list of tokens for the in-progress chunk

    def _flush():
        if current_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_tokens).strip())

    for sent in sentences:
        sent_tokens = tokenizer.tokenize(sent)
        if not sent_tokens:
            continue

        # Sentence longer than the window: hard-split it into max_tokens pieces.
        if len(sent_tokens) > max_tokens:
            _flush()
            current_tokens = []
            for i in range(0, len(sent_tokens), max_tokens - overlap if max_tokens > overlap else max_tokens):
                piece = sent_tokens[i : i + max_tokens]
                if not piece:
                    break
                chunks.append(tokenizer.convert_tokens_to_string(piece).strip())
            continue

        if len(current_tokens) + len(sent_tokens) > max_tokens and current_tokens:
            _flush()
            # Carry the trailing ``overlap`` tokens forward.
            current_tokens = current_tokens[-overlap:] if overlap else []

        current_tokens.extend(sent_tokens)

    _flush()
    return chunks
