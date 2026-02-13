import os
import json
from groq import Groq
from information_retrieval.retrievers.biobertretriever import BioBERTRetriever
from information_retrieval.retrievers.medcptretriever import MedCPTRetriever
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
    def query(self, question: str, chunks: list) -> str:
        """Query the RAG system with a question and retrieved chunks"""
        context = self._format_docs(chunks)
        
        rag_prompt = f"""ANSWER USING ONLY THE RETRIEVED MEDICAL DOCUMENTS BELOW AND ANSWER IT IN REQUIRED SENTENCES DONOT GIVE ANYOTHER INFROMATION RATHER THAN THAT JUST PROVIDE THE CLEAN ANSWER.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical assistant. Use ONLY the retrieved medical documents to answer."},
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    def ask(self, k: int = 3):
        messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant. Use ONLY the retrieved medical documents to answer."},
    ]

        while True:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["exit", "quit", "bye", "stop"]:

                print("Assistant: Goodbye!")
                break
            chunks = self.retriever.retrieve(user_input, k=k)
            context = self._format_docs(chunks)
            rag_prompt = f"""
ANSWER USING ONLY THE RETRIEVED MEDICAL DOCUMENTS BELOW AND ANSWER IT IN REQUIRED SENTENCES DONOT GIVE ANYOTHER INFROMATION RATHER THAN THAT JUST PROVIDE THE CLEAN ANSWER.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION: {user_input}

ANSWER:
"""
            messages.append({"role": "user", "content": rag_prompt})
            response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )

            answer = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": answer})

            print("\nAssistant:", answer)
if __name__ == "__main__":
    
    rag = medicalrag(
        model_type="biobert", 
        faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container")
    rag.ask(k=10)