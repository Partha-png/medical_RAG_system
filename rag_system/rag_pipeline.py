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

    def ask(self, k: int = 3):

        messages = [
        {"role": "system", "content": "You are a knowledgeable medical assistant. Use ONLY the retrieved medical documents to answer."},
    ]

        while True:

            user_input = input("\nUser: ").strip()

            if user_input.lower() in ["exit", "quit", "bye", "stop"]:

                print("Assistant: Goodbye!")
                break

        # Retrieve relevant chunks for this turn
            chunks = self.retriever.retrieve(user_input, k=k)
            context = self._format_docs(chunks)

        # Build RAG prompt
            rag_prompt = f"""
ANSWER USING ONLY THE RETRIEVED MEDICAL DOCUMENTS BELOW.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION: {user_input}

ANSWER:
"""

        # Add this turn to history (user turn)
            messages.append({"role": "user", "content": rag_prompt})

        # Query model
            response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )

            answer = response.choices[0].message.content.strip()

        # Add assistant reply to history
            messages.append({"role": "assistant", "content": answer})

            print("\nAssistant:", answer)


if __name__ == "__main__":
    
    rag = medicalrag(
        model_type="biobert", 
        faiss_dir=r"C:\Users\PARTHA SARATHI\Python\medical_rag\information_retrieval\fasiss_container")
    rag.ask(k=10)