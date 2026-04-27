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
if "comparison_rows" not in st.session_state:
    st.session_state.comparison_rows = []

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
    "retrieval_ms":        "Time spent fetching the top-k chunks from the index.",
    "generation_ms":       "Time the LLM took to generate the answer.",
    "total_ms":            "Total wall-clock time for retrieval + generation.",
}


def _fmt_ms(ms: float) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


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

        latency = metrics.get("latency") or {}
        if latency:
            st.markdown("##### Latency")
            l1, l2, l3 = st.columns(3)
            l1.metric(
                "Retrieval",
                _fmt_ms(latency.get("retrieval_ms")),
                help=_METRIC_HELP["retrieval_ms"],
            )
            l2.metric(
                "Generation",
                _fmt_ms(latency.get("generation_ms")),
                help=_METRIC_HELP["generation_ms"],
            )
            l3.metric(
                "Total",
                _fmt_ms(latency.get("total_ms")),
                help=_METRIC_HELP["total_ms"],
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

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Upload & Process",
    "💬 Chat",
    "📊 Session History",
    "🆚 Model Comparison",
])

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

                            # Log this run into the comparison table.
                            if metrics and "error" not in metrics:
                                from datetime import datetime as _dt
                                retrieval = metrics.get("retrieval", {})
                                ans_m = metrics.get("answer", {})
                                lat = metrics.get("latency", {})
                                st.session_state.comparison_rows.append({
                                    "Time": _dt.now().strftime("%H:%M:%S"),
                                    "Question": question,
                                    "Encoder": metrics.get("encoder", encoder_type),
                                    "k": k_chunks,
                                    "Faithfulness": ans_m.get("faithfulness"),
                                    "Top Chunk Rel.": retrieval.get("top_chunk_relevance"),
                                    "Context Rel.": retrieval.get("context_relevance"),
                                    "Diversity": retrieval.get("diversity"),
                                    "Answer Rel.": ans_m.get("answer_relevance"),
                                    "Ctx Util.": ans_m.get("context_utilization"),
                                    "Retrieval (ms)": lat.get("retrieval_ms"),
                                    "Generation (ms)": lat.get("generation_ms"),
                                    "Total (ms)": lat.get("total_ms"),
                                    "Answer": (answer[:120] + "…") if len(answer) > 120 else answer,
                                })

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

with tab4:
    st.header("Model Comparison")
    st.caption(
        "Every query you run is logged here as a row. "
        "To compare encoders/retrievers, ask the **same question**, then change the "
        "encoder in the sidebar and ask it again — each run appears side-by-side."
    )

    rows = st.session_state.comparison_rows

    if not rows:
        st.info("No runs yet. Ask a question in the Chat tab — it'll show up here.")
    else:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            questions = ["(all)"] + sorted({r["Question"] for r in rows})
            selected_q = st.selectbox("Filter by question", questions, index=0)
        with col_b:
            group_mode = st.checkbox("Group by question", value=True)
        with col_c:
            if st.button("🗑 Clear table", use_container_width=True):
                st.session_state.comparison_rows = []
                st.rerun()

        filtered = rows if selected_q == "(all)" else [r for r in rows if r["Question"] == selected_q]

        try:
            import pandas as pd
            df = pd.DataFrame(filtered)
            if group_mode and "Question" in df.columns:
                df = df.sort_values(by=["Question", "Time"]).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download as CSV",
                csv,
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        except ImportError:
            # Fall back to Streamlit's built-in renderer if pandas is missing.
            st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Per-question best-encoder summary
        if group_mode and len(filtered) >= 2:
            st.markdown("##### Best encoder per question (by Faithfulness)")
            try:
                import pandas as pd
                summary_df = pd.DataFrame(filtered)
                summary_df["Faithfulness"] = pd.to_numeric(summary_df["Faithfulness"], errors="coerce")
                idx = summary_df.groupby("Question")["Faithfulness"].idxmax().dropna()
                best = summary_df.loc[idx, [
                    "Question", "Encoder", "Faithfulness",
                    "Top Chunk Rel.", "Context Rel.", "Total (ms)",
                ]].reset_index(drop=True)
                st.dataframe(best, use_container_width=True, hide_index=True)
            except Exception:
                pass


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
