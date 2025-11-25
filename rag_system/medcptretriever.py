import faiss
import pickle
import numpy as np
from .medcptqueryencoder import MEDCPTQueryEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class MedCPTRetriever:
    """MedCPT FAISS vector retriever."""

    def __init__(self, faiss_dir: str):
        self.faiss_dir = faiss_dir
        self.encoder = MEDCPTQueryEncoder()
        self.index = faiss.read_index(f"{faiss_dir}/index.faiss")
        with open(f"{faiss_dir}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query: str, k=3):
        qv = self.encoder.encode(query).astype("float32")
        D, I = self.index.search(qv, k)
        return [self.metadata[i] for i in I[0]]

    # def as_langchain_retriever(self):
    #     embedder = HuggingFaceEmbeddings(model_name="ncbi/MedCPT-Query-Encoder")
    #     vectorstore = FAISS.load_local(
    #         self.faiss_dir,
    #         embedder,
    #         allow_dangerous_deserialization=True
    #     )
    #     return vectorstore.as_retriever(search_kwargs={"k": 3})
if __name__ == "__main__":
    retriever = MedCPTRetriever(faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container")
    results = retriever.retrieve("What are the symptoms of diabetes?", k=1)
    for res in results:
        print(res)