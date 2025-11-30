import streamlit as st
import os
import time
from information_retrieval.document_encoding.encoder import encode_documents
from rag_system.rag_pipeline import medicalrag
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Medical RAG System", page_icon="🏥", layout="wide")

# Initialize session state
if 'encoded' not in st.session_state:
    st.session_state.encoded = False
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = None

FAISS_DIR = os.path.join(os.path.dirname(__file__), "information_retrieval", "faiss_container")
FAISS_DIR = os.path.abspath(FAISS_DIR)
TEMP_UPLOADS = os.path.abspath("temp_uploads")
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS, exist_ok=True)

st.title("🏥 Medical RAG System")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_type = st.selectbox("Encoder Model", ["biobert", "medcpt"])
    k_chunks = st.slider("Chunks to retrieve", 1, 10, 3)
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ Add GROQ_API_KEY to .env")
    
    if st.button("Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Tabs
tab1, tab2 = st.tabs(["Upload & Process", "Chat"])

with tab1:
    uploaded_file = st.file_uploader("Upload medical document", type=["pdf", "txt"])
    
    if uploaded_file:
        st.info(f"**{uploaded_file.name}** - {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("Process Document", type="primary"):
            try:
                temp_file_path = os.path.join(TEMP_UPLOADS, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing..."):
                    encode_documents(model_type, TEMP_UPLOADS, temp_file_path, batch_size=8)
                    time.sleep(1)
                    
                    index_path = os.path.join(FAISS_DIR, f"{model_type}index.faiss")
                    metadata_path = os.path.join(FAISS_DIR, f"{model_type}metadata.pkl")
                    
                    st.write(f"Looking for index at: {index_path}")
                    st.write(f"Looking for metadata at: {metadata_path}")
                    
                    if os.path.exists(FAISS_DIR):
                        st.write(f"FAISS directory contents: {os.listdir(FAISS_DIR)}")
                    
                    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                        raise FileNotFoundError(f"FAISS index not created. Check paths above.")
                    
                    st.session_state.rag_instance = medicalrag(model_type, FAISS_DIR, api_key)
                    st.session_state.encoded = True
                    st.session_state.file_processed = uploaded_file.name
                
                st.success("✅ Ready for questions!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                with st.expander("Debug"):
                    st.write("FAISS Dir:", FAISS_DIR)
                    st.write("Contents:", os.listdir(FAISS_DIR) if os.path.exists(FAISS_DIR) else "N/A")
    
    if st.session_state.encoded:
        st.success(f"Loaded: **{st.session_state.file_processed}** ({model_type.upper()})")

with tab2:
    if not st.session_state.encoded:
        st.warning("⚠️ Process a document first")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about your document..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                try:
                    # Retrieve chunks
                    chunks = st.session_state.rag_instance.retriever.retrieve(prompt, k=k_chunks)
                    
                    # Use the RAG pipeline's query method
                    answer = st.session_state.rag_instance.query(prompt, chunks)
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    with st.expander("View Retrieved Chunks"):
                        for idx, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {idx}:**")
                            st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})