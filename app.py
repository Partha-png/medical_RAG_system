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


# ----- helpers -----------------------------------------------------------------

def _show_response_error(label: str, response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"{label} ({response.status_code}): {detail}")


_METRIC_HELP = {
    "context_relevance":   "How much of the question's content the retrieved chunks cover, on average.",
    "top_chunk_relevance": "Coverage of the question by the single best chunk.",
    "diversity":           "How non-redundant the retrieved chunks are (1.0 = all distinct).",
    "faithfulness":        "Fraction of the answer's content words that appear in the retrieved chunks.",
    "answer_relevance":    "Overlap between the question and the answer.",
    "context_utilization": "Fraction of retrieved chunks that meaningfully contributed to the answer.",
}


def _quality_label(score: float) -> str:
    if score >= 0.66:
        return "🟢 Strong"
    if score >= 0.33:
        return "🟡 Moderate"
    return "🔴 Weak"


def _render_metrics(metrics: dict, use_expander: bool = True) -> None:
    """Render the auto-evaluation metrics for one assistant turn.

    If `use_expander` is False, draws inline (used inside the History tab,
    where the surrounding message is already inside an expander -- Streamlit
    forbids nested expanders).
    """
    if not metrics or "error" in metrics:
        if metrics and "error" in metrics:
            st.caption(f"Could not compute quality metrics: {metrics['error']}")
        return

    retrieval = metrics.get("retrieval", {})
    answer = metrics.get("answer", {})
    counts = metrics.get("counts", {})

    retr_avg = sum(retrieval.values()) / len(retrieval) if retrieval else 0.0
    ans_avg = sum(answer.values()) / len(answer) if answer else 0.0

    header = (
        f"📈 Quality — Retrieval {_quality_label(retr_avg)} ({retr_avg:.2f})  ·  "
        f"Answer {_quality_label(ans_avg)} ({ans_avg:.2f})"
    )

    if use_expander:
        container = st.expander(header, expanded=False)
    else:
        st.markdown(f"**{header}**")
        container = st.container()

    with container:
        st.markdown("##### Retrieval Quality")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Context Relevance",
            f"{retrieval.get('context_relevance', 0):.2f}",
            help=_METRIC_HELP["context_relevance"],
        )
        c2.metric(
            "Top Chunk Relevance",
            f"{retrieval.get('top_chunk_relevance', 0):.2f}",
            help=_METRIC_HELP["top_chunk_relevance"],
        )
        c3.metric(
            "Diversity",
            f"{retrieval.get('diversity', 0):.2f}",
            help=_METRIC_HELP["diversity"],
        )

        st.markdown("##### Answer Quality")
        c4, c5, c6 = st.columns(3)
        c4.metric(
            "Faithfulness",
            f"{answer.get('faithfulness', 0):.2f}",
            help=_METRIC_HELP["faithfulness"],
        )
        c5.metric(
            "Answer Relevance",
            f"{answer.get('answer_relevance', 0):.2f}",
            help=_METRIC_HELP["answer_relevance"],
        )
        c6.metric(
            "Context Utilization",
            f"{answer.get('context_utilization', 0):.2f}",
            help=_METRIC_HELP["context_utilization"],
        )

        if counts:
            st.caption(
                f"chunks: {counts.get('num_chunks', 0)}  ·  "
                f"answer words: {counts.get('answer_words', 0)}  ·  "
                f"question words: {counts.get('question_words', 0)}"
            )


# ----- sidebar ----------------------------------------------------------------

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


# ----- main tabs --------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "💬 Chat", "📊 Session History"])

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
        # Render existing history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "chunks" in msg:
                    with st.expander("View Retrieved Chunks"):
                        for i, chunk in enumerate(msg["chunks"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            st.divider()
                if "metrics" in msg and msg["metrics"]:
                    _render_metrics(msg["metrics"])

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
                            metrics = result.get("metrics")

                            st.markdown(answer)

                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": answer,
                                    "chunks": chunks,
                                    "metrics": metrics,
                                }
                            )

                            with st.expander("View Retrieved Chunks"):
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    st.divider()

                            if metrics:
                                _render_metrics(metrics)
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

                            if msg.get("metrics"):
                                st.write("**Quality Metrics:**")
                                _render_metrics(msg["metrics"], use_expander=False)
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
