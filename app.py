"""
Updated Streamlit Frontend - Uses FastAPI Backend
"""
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Backend API Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Medical RAG System", page_icon="🏥", layout="wide")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("🏥 Medical RAG System")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Encoder selection
    encoder_type = st.selectbox(
        "📦 Encoder Model",
        ["biobert", "medcpt", "bm25", "hybrid"],
        help="Model used to encode/index your document"
    )
    
    # Auto-set retriever based on encoder
    # For BioBERT, MedCPT, BM25 → retriever = encoder
    # For Hybrid → retriever = hybrid (uses BioBERT + BM25)
    if encoder_type in ["biobert", "medcpt", "bm25", "hybrid"]:
        retriever_type = encoder_type
    else:
        retriever_type = encoder_type
    
    # Show retriever (read-only display)
    st.text_input(
        "🔍 Retriever Model",
        value=retriever_type,
        disabled=True,
        help="Automatically matches encoder selection"
    )
    
    # Explanation
    if encoder_type == "hybrid":
        st.caption("ℹ️ Hybrid uses BioBERT encoder + BM25 for retrieval")
    elif encoder_type in ["biobert", "medcpt"]:
        st.caption(f"ℹ️ {encoder_type.upper()} used for both encoding and retrieval")
    elif encoder_type == "bm25":
        st.caption("ℹ️ BM25 is keyword-based (no neural encoding)")
    
    k_chunks = st.slider("Chunks to retrieve", 1, 10, 3)
    
    # API Key status
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ Add GROQ_API_KEY to .env")
    
    # Session info
    if st.session_state.session_id:
        st.info(f"📝 Session: {st.session_state.session_id[:8]}...")
        if st.button("📊 View Session Info"):
            try:
                response = requests.get(f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}")
                if response.status_code == 200:
                    session_data = response.json()
                    st.json(session_data)
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.button("Reset Session", use_container_width=True):
        if st.session_state.session_id:
            try:
                # Delete session on backend
                requests.delete(f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}")
            except:
                pass
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Process", "💬 Chat", "📊 Session History", "🧪 Evaluation"])

with tab1:
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader("Upload medical document", type=["pdf", "txt"])
    
    if uploaded_file:
        st.info(f"**{uploaded_file.name}** - {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("Process Document", type="primary"):
            try:
                with st.spinner("Creating session..."):
                    # Step 1: Create session
                    session_response = requests.post(
                        f"{API_BASE_URL}/api/sessions",
                        json={"encoder_type": encoder_type}
                    )
                    
                    if session_response.status_code != 201:
                        st.error(f"Failed to create session: {session_response.text}")
                        st.stop()
                    
                    session_data = session_response.json()
                    session_id = session_data["session_id"]
                    st.session_state.session_id = session_id
                    st.success(f"✅ Session created: {session_id[:8]}...")
                
                with st.spinner("Uploading and processing document..."):
                    # Step 2: Upload document
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"session_id": session_id}
                    
                    upload_response = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files=files,
                        data=data
                    )
                    
                    if upload_response.status_code != 200:
                        st.error(f"Failed to upload document: {upload_response.text}")
                        st.stop()
                    
                    result = upload_response.json()
                    st.success("✅ Document processed successfully!")
                    
                    # Show processing details
                    with st.expander("Processing Details"):
                        st.write(f"**Encoder**: {encoder_type}")
                        st.write(f"**Retriever**: {retriever_type}")
                        st.write(f"**Chunks created**: {result.get('num_chunks', 'N/A')}")
                        st.write(f"**Session ID**: {session_id}")
                
                st.info("👉 Go to the **Chat** tab to ask questions!")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure FastAPI server is running on port 8000")
                st.code("uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())

with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.session_id:
        st.warning("⚠️ Please upload and process a document first (Upload & Process tab)")
    else:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "chunks" in msg:
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            st.divider()
        
        # Chat input
        if question := st.chat_input("Ask about your document..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            with st.chat_message("user"):
                st.markdown(question)
            
            # Get answer from backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/query",
                            json={
                                "session_id": st.session_state.session_id,
                                "question": question,
                                "k": k_chunks
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            answer = result["answer"]
                            chunks = result["chunks"]  # Backend returns 'chunks', not 'retrieved_chunks'
                            
                            st.markdown(answer)
                            
                            # Store assistant response
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "chunks": chunks
                            })
                            
                            # Show retrieved chunks
                            with st.expander("View Retrieved Chunks"):
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    st.divider()
                        else:
                            error_msg = f"Error: {response.status_code} - {response.text}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
                    
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

with tab3:
    st.header("Conversation History")
    
    if st.session_state.session_id:
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation"
            )
            
            if response.status_code == 200:
                history = response.json()
                if history["messages"]:
                    st.write(f"**Total messages**: {len(history['messages'])}")
                    
                    for i, msg in enumerate(history["messages"]):
                        with st.expander(f"Message {i+1} - {msg['role'].title()}"):
                            st.write(f"**Question**: {msg['question']}")
                            st.write(f"**Answer**: {msg['answer']}")
                            st.write(f"**Timestamp**: {msg['timestamp']}")
                            
                            if msg.get("retrieved_chunks"):
                                st.write("**Retrieved Chunks:**")
                                for j, chunk in enumerate(msg["retrieved_chunks"], 1):
                                    st.text(f"{j}. {chunk[:200]}...")
                else:
                    st.info("No conversation history yet")
                
                if st.button("Clear History"):
                    delete_response = requests.delete(
                        f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation"
                    )
                    if delete_response.status_code == 200:
                        st.success("✅ History cleared")
                        st.rerun()
            else:
                st.error(f"Error fetching history: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("No active session")

with tab4:
    st.header("📊 Evaluation Metrics")
    st.markdown("Evaluate the quality of retrieved chunks and generated answers against reference data.")

    eval_tab1, eval_tab2 = st.tabs(["🤖 RAG Answer Quality", "🔍 Retrieval Quality"])

    with eval_tab1:
        st.subheader("RAG Answer Evaluation (BLEU / ROUGE)")
        st.caption("Compare a generated answer against a reference (ground-truth) answer.")

        generated_answer = st.text_area(
            "Generated Answer",
            placeholder="Paste the answer produced by the RAG system...",
            height=120,
            key="eval_generated"
        )
        reference_answer = st.text_area(
            "Reference Answer (Ground Truth)",
            placeholder="Paste the expected / correct answer...",
            height=120,
            key="eval_reference"
        )

        # Optionally attach current session chunks
        use_session_chunks = False
        if st.session_state.session_id:
            use_session_chunks = st.checkbox(
                "Include retrieved chunks from current session in context evaluation",
                value=False
            )

        if st.button("▶ Run RAG Evaluation", type="primary", key="run_rag_eval"):
            if not generated_answer.strip() or not reference_answer.strip():
                st.warning("⚠️ Please fill in both the Generated Answer and Reference Answer.")
            else:
                try:
                    payload = {
                        "generated_answer": generated_answer,
                        "reference_answer": reference_answer,
                    }
                    # attach chunks from last chat message if requested
                    if use_session_chunks:
                        for msg in reversed(st.session_state.chat_history):
                            if msg.get("role") == "assistant" and msg.get("chunks"):
                                payload["retrieved_chunks"] = msg["chunks"]
                                break

                    with st.spinner("Evaluating..."):
                        resp = requests.post(
                            f"{API_BASE_URL}/api/evaluation/rag",
                            json=payload
                        )

                    if resp.status_code == 200:
                        metrics = resp.json()
                        st.success("✅ Evaluation complete!")
                        if "error" in metrics:
                            st.error(f"Backend error: {metrics['error']}")
                        else:
                            cols = st.columns(len(metrics))
                            for col, (k, v) in zip(cols, metrics.items()):
                                label = k.replace("_", " ").upper()
                                col.metric(label=label, value=f"{v:.4f}" if isinstance(v, float) else str(v))
                            with st.expander("Raw JSON"):
                                st.json(metrics)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend (port 8000). Make sure FastAPI is running.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    with eval_tab2:
        st.subheader("Retrieval Quality Evaluation (Precision / Recall / F1)")
        st.caption(
            "Provide a list of retrieved document IDs/snippets and a list of truly relevant ones "
            "to compute Precision, Recall, and F1."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            retrieved_raw = st.text_area(
                "Retrieved Documents (one per line)",
                placeholder="chunk_1\nchunk_2\nchunk_3",
                height=160,
                key="eval_retrieved"
            )
        with col_b:
            relevant_raw = st.text_area(
                "Relevant Documents – Ground Truth (one per line)",
                placeholder="chunk_1\nchunk_4",
                height=160,
                key="eval_relevant"
            )

        if st.button("▶ Run Retrieval Evaluation", type="primary", key="run_ret_eval"):
            retrieved_list = [d.strip() for d in retrieved_raw.strip().splitlines() if d.strip()]
            relevant_list  = [d.strip() for d in relevant_raw.strip().splitlines()  if d.strip()]

            if not retrieved_list or not relevant_list:
                st.warning("⚠️ Please provide at least one entry in each field.")
            else:
                try:
                    payload = {
                        "retrieved_docs": retrieved_list,
                        "relevant_docs": relevant_list,
                    }
                    with st.spinner("Evaluating..."):
                        resp = requests.post(
                            f"{API_BASE_URL}/api/evaluation/retrieval",
                            json=payload
                        )

                    if resp.status_code == 200:
                        metrics = resp.json()
                        st.success("✅ Evaluation complete!")
                        if "error" in metrics:
                            st.error(f"Backend error: {metrics['error']}")
                        else:
                            m_cols = st.columns(3)
                            m_cols[0].metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            m_cols[1].metric("Recall",    f"{metrics.get('recall', 0):.4f}")
                            m_cols[2].metric("F1 Score",  f"{metrics.get('f1', 0):.4f}")

                            st.divider()
                            info_cols = st.columns(3)
                            info_cols[0].info(f"🗂 Retrieved: **{metrics.get('num_retrieved', 0)}**")
                            info_cols[1].info(f"✅ Relevant: **{metrics.get('num_relevant', 0)}**")
                            info_cols[2].info(f"🎯 Correct: **{metrics.get('num_correct', 0)}**")

                            with st.expander("Raw JSON"):
                                st.json(metrics)
                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend (port 8000). Make sure FastAPI is running.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption(f"📦 Encoder: **{encoder_type}**")
with col2:
    st.caption(f"🔍 Retriever: **{retriever_type}**")
with col3:
    st.caption(f"📊 Chunks: **{k_chunks}**")
with col4:
    if st.session_state.session_id:
        st.caption(f"📝 Session: **Active**")
    else:
        st.caption("📝 Session: **None**")