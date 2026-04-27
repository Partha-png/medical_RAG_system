import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import pickle
from .biobertqueryencoder import BioBERTQueryEncoder


# Module-level singleton: avoid re-loading BioBERT (~440MB) per retriever instance.
_QUERY_ENCODER = None


def _get_encoder() -> BioBERTQueryEncoder:
    global _QUERY_ENCODER
    if _QUERY_ENCODER is None:
        _QUERY_ENCODER = BioBERTQueryEncoder()
    return _QUERY_ENCODER


class BioBERTRetriever:
    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir
        self.encoder = _get_encoder()
        self.index = faiss.read_index(f"{faiss_dir}/biobertindex.faiss")
        with open(f"{faiss_dir}/biobertmetadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, k: int = 3):
        qv = self.encoder.encode(query).astype("float32")
        _, I = self.index.search(qv, k)
        return [self.metadata[i] for i in I[0]]
