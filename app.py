"""
Streamlit Frontend - Talks to the FastAPI Backend
"""
import os
import traceback

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

# Backend API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical RAG System", page_icon="🏥", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🏥 Medical RAG System")


def _show_response_error(label: str, response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"{label} ({response.status_code}): {detail}")


# Sidebar
with st.sidebar:
    st.header("Configuration")

    encoder_type = st.selectbox(
        "📦 Encoder Model",
        ["biobert", "medcpt", "bm25", "hybrid"],
        help="Model used to encode/index your document",
    )

    retriever_type = encoder_type
    st.text_input(
        "🔍 Retriever Model",
        value=retriever_type,
        disabled=True,
        help="Automatically matches encoder selection",
    )

    if encoder_type == "hybrid":
        st.caption("ℹ️ Hybrid uses BioBERT encoder + BM25 for retrieval")
    elif encoder_type in ["biobert", "medcpt"]:
        st.caption(f"ℹ️ {encoder_type.upper()} used for both encoding and retrieval")
    elif encoder_type == "bm25":
        st.caption("ℹ️ BM25 is keyword-based (no neural encoding)")

    k_chunks = st.slider("Chunks to retrieve", 1, 10, 3)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ Add GROQ_API_KEY to your environment")

    if st.session_state.session_id:
        st.info(f"📝 Session: {st.session_state.session_id[:8]}...")
        if st.button("📊 View Session Info"):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}",
                    timeout=10,
                )
                if response.ok:
                    st.json(response.json())
                else:
                    _show_response_error("Failed to load session", response)
            except requests.RequestException as e:
                st.error(f"Network error: {e}")

    if st.button("Reset Session", use_container_width=True):
        if st.session_state.session_id:
            try:
                resp = requests.delete(
                    f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}",
                    timeout=10,
                )
                if not resp.ok and resp.status_code != 404:
                    _show_response_error("Failed to delete session", resp)
            except requests.RequestException as e:
                st.warning(f"Could not reach backend to delete session: {e}")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Upload & Process", "💬 Chat", "📊 Session History", "🧪 Evaluation"]
)

with tab1:
    st.header("Document Upload")

    uploaded_file = st.file_uploader("Upload medical document", type=["pdf", "txt"])

    if uploaded_file:
        st.info(f"**{uploaded_file.name}** - {uploaded_file.size / 1024:.2f} KB")

        if st.button("Process Document", type="primary"):
            try:
                with st.spinner("Creating session..."):
                    session_response = requests.post(
                        f"{API_BASE_URL}/api/sessions",
                        json={"encoder_type": encoder_type},
                        timeout=30,
                    )

                    if session_response.status_code != 201:
                        _show_response_error("Failed to create session", session_response)
                        st.stop()

                    session_data = session_response.json()
                    session_id = session_data["session_id"]
                    st.session_state.session_id = session_id
                    st.success(f"✅ Session created: {session_id[:8]}...")

                with st.spinner("Uploading and processing document..."):
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    data = {"session_id": session_id}

                    upload_response = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files=files,
                        data=data,
                        timeout=600,
                    )

                    if not upload_response.ok:
                        _show_response_error("Failed to upload document", upload_response)
                        st.stop()

                    result = upload_response.json()
                    st.success("✅ Document processed successfully!")

                    with st.expander("Processing Details"):
                        st.write(f"**Encoder**: {encoder_type}")
                        st.write(f"**Retriever**: {retriever_type}")
                        st.write(f"**Embeddings**: {result.get('num_embeddings', 'N/A')}")
                        st.write(f"**Session ID**: {session_id}")

                st.info("👉 Go to the **Chat** tab to ask questions!")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot connect to backend at {API_BASE_URL}. "
                    "Make sure the FastAPI server is running."
                )
            except requests.RequestException as e:
                st.error(f"Network error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())

with tab2:
    st.header("Ask Questions")

    if not st.session_state.session_id:
        st.warning("⚠️ Please upload and process a document first (Upload & Process tab)")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "chunks" in msg:
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            st.divider()

        if question := st.chat_input("Ask about your document..."):
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/query",
                            json={
                                "session_id": st.session_state.session_id,
                                "question": question,
                                "k": k_chunks,
                            },
                            timeout=300,
                        )

                        if response.ok:
                            result = response.json()
                            answer = result["answer"]
                            chunks = result["chunks"]

                            st.markdown(answer)

                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer, "chunks": chunks}
                            )

                            with st.expander("View Retrieved Chunks"):
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    st.divider()
                        else:
                            try:
                                detail = response.json().get("detail", response.text)
                            except ValueError:
                                detail = response.text
                            error_msg = f"Error {response.status_code}: {detail}"
                            st.error(error_msg)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": error_msg}
                            )
                    except requests.RequestException as e:
                        error_msg = f"Network error: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": error_msg}
                        )

with tab3:
    st.header("Conversation History")

    if st.session_state.session_id:
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation",
                timeout=15,
            )

            if response.ok:
                history = response.json()
                messages = history.get("messages", [])
                if messages:
                    st.write(f"**Total messages**: {len(messages)}")

                    for i, msg in enumerate(messages):
                        with st.expander(f"Message {i + 1} - {msg.get('role', '').title()}"):
                            st.write(f"**Role**: {msg.get('role', '')}")
                            st.write(f"**Content**: {msg.get('content', '')}")
                            st.write(f"**Timestamp**: {msg.get('timestamp', '')}")

                            if msg.get("retrieved_chunks"):
                                st.write("**Retrieved Chunks:**")
                                for j, chunk in enumerate(msg["retrieved_chunks"], 1):
                                    st.text(f"{j}. {chunk[:200]}...")
                else:
                    st.info("No conversation history yet")

                if st.button("Clear History"):
                    delete_response = requests.delete(
                        f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation",
                        timeout=15,
                    )
                    # Backend returns 204 No Content on success.
                    if delete_response.status_code in (200, 204):
                        st.success("✅ History cleared")
                        st.rerun()
                    else:
                        _show_response_error("Failed to clear history", delete_response)
            else:
                _show_response_error("Error fetching history", response)
        except requests.RequestException as e:
            st.error(f"Network error: {e}")
    else:
        st.info("No active session")

with tab4:
    st.header("📊 Evaluation Metrics")
    st.markdown(
        "Evaluate the quality of retrieved chunks and generated answers against reference data."
    )

    eval_tab1, eval_tab2 = st.tabs(["🤖 RAG Answer Quality", "🔍 Retrieval Quality"])

    with eval_tab1:
        st.subheader("RAG Answer Evaluation")
        st.caption(
            "Paste the generated answer and the ground truth answer to compute all metrics."
        )

        generated_answer = st.text_area(
            "Generated Answer",
            placeholder="Paste the answer produced by the RAG system...",
            height=120,
            key="eval_generated",
        )
        reference_answer = st.text_area(
            "Reference Answer (Ground Truth)",
            placeholder="Paste the expected correct answer...",
            height=120,
            key="eval_reference",
        )

        use_session_chunks = False
        if st.session_state.session_id:
            use_session_chunks = st.checkbox(
                "Include retrieved chunks from current session",
                value=True,
            )

        if st.button("Run Evaluation", type="primary", key="run_rag_eval"):
            if not generated_answer.strip() or not reference_answer.strip():
                st.warning("Please fill in both fields.")
            else:
                try:
                    payload = {
                        "generated_answer": generated_answer,
                        "reference_answer": reference_answer,
                    }
                    if use_session_chunks:
                        for msg in reversed(st.session_state.chat_history):
                            if msg.get("role") == "assistant" and msg.get("chunks"):
                                payload["retrieved_chunks"] = msg["chunks"]
                                break

                    with st.spinner("Evaluating..."):
                        resp = requests.post(
                            f"{API_BASE_URL}/api/evaluation/rag",
                            json=payload,
                            timeout=120,
                        )

                    if resp.ok:
                        m = resp.json()
                        st.success("Evaluation complete!")

                        if "error" in m:
                            st.error(f"Error: {m['error']}")
                        else:
                            st.markdown("#### BLEU Score")
                            st.metric("BLEU", f"{m.get('bleu_score', 0):.4f}")

                            st.divider()

                            st.markdown("#### ROUGE Scores")
                            rouge_data = {
                                "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
                                "Precision": [
                                    m.get("rouge1_precision", 0),
                                    m.get("rouge2_precision", 0),
                                    m.get("rougeL_precision", 0),
                                ],
                                "Recall": [
                                    m.get("rouge1_recall", 0),
                                    m.get("rouge2_recall", 0),
                                    m.get("rougeL_recall", 0),
                                ],
                                "F1": [
                                    m.get("rouge1_f1", 0),
                                    m.get("rouge2_f1", 0),
                                    m.get("rougeL_f1", 0),
                                ],
                            }
                            import pandas as pd

                            st.dataframe(
                                pd.DataFrame(rouge_data)
                                .set_index("Metric")
                                .style.format("{:.4f}"),
                                use_container_width=True,
                            )

                            st.divider()

                            st.markdown("#### Retrieval Metrics")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Precision@3", f"{m.get('precision_at_3', 0):.4f}")
                            c2.metric("Recall@3", f"{m.get('recall_at_3', 0):.4f}")
                            c3.metric("F1@3", f"{m.get('f1_at_3', 0):.4f}")

                            st.divider()

                            st.markdown("#### Answer Quality")
                            c4, c5, c6, c7 = st.columns(4)
                            c4.metric("Word Overlap", f"{m.get('word_overlap', 0):.4f}")
                            c5.metric("Length Ratio", f"{m.get('length_ratio', 0):.4f}")
                            c6.metric("Generated Length", m.get("generated_length", 0))
                            c7.metric("Chunks Used", m.get("num_chunks_used", 0))
                    else:
                        _show_response_error("API Error", resp)

                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to backend at {API_BASE_URL}.")
                except requests.RequestException as e:
                    st.error(f"Network error: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

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
                key="eval_retrieved",
            )
        with col_b:
            relevant_raw = st.text_area(
                "Relevant Documents – Ground Truth (one per line)",
                placeholder="chunk_1\nchunk_4",
                height=160,
                key="eval_relevant",
            )

        if st.button("▶ Run Retrieval Evaluation", type="primary", key="run_ret_eval"):
            retrieved_list = [d.strip() for d in retrieved_raw.strip().splitlines() if d.strip()]
            relevant_list = [d.strip() for d in relevant_raw.strip().splitlines() if d.strip()]

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
                            json=payload,
                            timeout=60,
                        )

                    if resp.ok:
                        metrics = resp.json()
                        st.success("✅ Evaluation complete!")
                        if "error" in metrics:
                            st.error(f"Backend error: {metrics['error']}")
                        else:
                            m_cols = st.columns(3)
                            m_cols[0].metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            m_cols[1].metric("Recall", f"{metrics.get('recall', 0):.4f}")
                            m_cols[2].metric("F1 Score", f"{metrics.get('f1', 0):.4f}")

                            st.divider()
                            info_cols = st.columns(3)
                            info_cols[0].info(f"🗂 Retrieved: **{metrics.get('num_retrieved', 0)}**")
                            info_cols[1].info(f"✅ Relevant: **{metrics.get('num_relevant', 0)}**")
                            info_cols[2].info(f"🎯 Correct: **{metrics.get('num_correct', 0)}**")

                            with st.expander("Raw JSON"):
                                st.json(metrics)
                    else:
                        _show_response_error("API Error", resp)
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to backend at {API_BASE_URL}.")
                except requests.RequestException as e:
                    st.error(f"Network error: {e}")
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
        st.caption("📝 Session: **Active**")
    else:
        st.caption("📝 Session: **None**")
