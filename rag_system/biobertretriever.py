import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import numpy as np
from .bioBERTqueryencoder import BioBERTQueryEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class BioBERTRetriever:
    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir
        self.encoder = BioBERTQueryEncoder()
        self.index = faiss.read_index(f"{faiss_dir}/biobertindex.faiss")
        with open(f"{faiss_dir}/biobertmetadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, k=3):
        qv = self.encoder.encode(query).astype("float32")
        D, I = self.index.search(qv, k)
        return [self.metadata[i] for i in I[0]]
if __name__ == "__main__":
    retriever = BioBERTRetriever(faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container")
    results = retriever.retrieve("What are the symptoms of diabetes?", k=1)
    for res in results:
        print(res)