from information_retrieval.document_encoding.bioBERT_encoder import BioBERTEncoder

class BioBERTQueryEncoder:

    def __init__(self):
        self.encoder = BioBERTEncoder()

    def encode(self, query: str):
        return self.encoder.encode([query])