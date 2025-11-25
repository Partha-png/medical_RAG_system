import streamlit as st
import sys
import os

# Make project importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IR_DIR = os.path.join(ROOT_DIR, "information_retrieval")
DOC_ENC_DIR = os.path.join(IR_DIR, "document_encoding")
RAG_DIR = os.path.join(ROOT_DIR, "rag_system")

sys.path.append(IR_DIR)
sys.path.append(DOC_ENC_DIR)
sys.path.append(RAG_DIR)

# Import retrievers
from .rag_system.biobertriever import BioBERTRetriever
from .rag_system.medcptretriever import MedCPTRetriever
from .rag_system.bm25_retriever import BM25Retriever
from .rag_system.hybrid_retriever import HybridRetriever

# LangChain RAG pipeline
from langchain_rag import MedicalRAG


FAISS_DIR = os.path.join(IR_DIR, "fasiss_container")

st.set_page_config(page_title="Medical RAG System", layout="wide")
st.title("🧬 Medical RAG System (BioBERT + MedCPT + Hybrid + BM25)")


# Sidebar
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Retrieval Model",
    ["BioBERT", "MedCPT", "BM25", "Hybrid"],
)

k_value = st.sidebar.slider("Top-k Results", 1, 10, 3)

use_rag_llm = st.sidebar.checkbox("Use LLM Answer (Groq)", value=True)

query = st.text_input("Enter your medical query")


# Load documents for BM25 / Hybrid
def load_metadata():
    meta_path = os.path.join(FAISS_DIR, "metadata.pkl")
    if os.path.exists(meta_path):
        import pickle
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return []


DOCS = load_metadata()


# Initialize retriever(s)
def get_retriever():
    if model_choice == "BioBERT":
        return BioBERTRetriever(FAISS_DIR)

    elif model_choice == "MedCPT":
        return MedCPTRetriever(FAISS_DIR)

    elif model_choice == "BM25":
        return BM25Retriever(DOCS)

    elif model_choice == "Hybrid":
        return HybridRetriever(FAISS_DIR, DOCS)


retriever = get_retriever()

# Initialize MedicalRAG (optional)
rag_chain = None
if use_rag_llm:
    try:
        rag_chain = MedicalRAG(model_choice.lower(), FAISS_DIR)
    except Exception as e:
        st.warning(f"LLM not available: {e}")


# Perform search
if st.button("Search"):
    if not query.strip():
        st.error("Please enter a query!")
    else:
        st.subheader("🔍 Retrieved Documents")
        results = retriever.retrieve(query, k=k_value)

        for i, r in enumerate(results):
            st.markdown(f"**{i+1}.** {r}\n\n---")

        if use_rag_llm and rag_chain:
            st.subheader("🤖 LLM Final Answer (Groq)")
            try:
                result = rag_chain.ask(query)
                answer = result.get("answer", "No answer produced.")
                st.success(answer)
            except Exception as e:
                st.error(f"LLM error: {e}")