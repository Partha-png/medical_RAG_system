import os
import json
from groq import Groq
from .biobertretriever import BioBERTRetriever
from .medcptretriever import MedCPTRetriever
from dotenv import load_dotenv
load_dotenv() 
groq_api_key = os.getenv("GROQ_API_KEY")

class medicalrag:


    def __init__(self, model_type: str, faiss_dir: str, api_key: str = None):
        self.model_type = model_type.lower()
        self.faiss_dir = faiss_dir

        if self.model_type == "biobert":
            self.retriever = BioBERTRetriever(faiss_dir)
        else:
            self.retriever = MedCPTRetriever(faiss_dir)

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.client = Groq(api_key=self.api_key)
        self.model = "openai/gpt-oss-120b"

    def _format_docs(self, docs: list) -> str:

        out = []
        for idx, c in enumerate(docs):
            out.append(f"[DOC {idx+1}]\n{c}\n")
        return "\n".join(out)

    def ask(self, question: str, k: int = 3) -> dict:
    
        chunks = self.retriever.retrieve(question, k=k)

        context_section = self._format_docs(chunks)

        system_message = (
            "You are a medical assistant answer pin point to the question asked. "
            "Answer using your own knowledge taking help of context below and the answer should be related to the question.\n"
            f"Retrieved Documents:\n{context_section}"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": chunks
        }

if __name__ == "__main__":
    
    rag = medicalrag(
        model_type="medcpt", 
        faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container"
    )
    question = "hi who is this?"
    result = rag.ask(question, k=3)
    print(result["answer"])