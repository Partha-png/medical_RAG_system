"""
Streamlit Frontend - Talks to the FastAPI Backend
"""
import os
import uuid
import traceback
from datetime import datetime

import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical RAG System", page_icon="🏥", layout="wide")

# --- Session state ------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "comparison_rows" not in st.session_state:
    # Each row: dict keyed by friendly column names; "_run_id" is internal.
    st.session_state.comparison_rows = []

st.title("🏥 Medical RAG System")


# ----- helpers ----------------------------------------------------------------

def _show_response_error(label: str, response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"{label} ({response.status_code}): {detail}")


def _fmt_ms(ms):
    if ms is None:
        return "—"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


def _fmt_pct(v, suffix="%"):
    return "—" if v is None else f"{v:.1f}{suffix}"


def _fmt_mb(v):
    return "—" if v is None else f"{v:.0f} MB"


def _fmt_score(v):
    return "—" if v is None else f"{v:.2f}"


def _quality_label(score: float) -> str:
    if score is None:
        return "—"
    if score >= 0.66:
        return "🟢"
    if score >= 0.33:
        return "🟡"
    return "🔴"


_HELP = {
    "faithfulness":         "Fraction of the answer's content words supported by the retrieved chunks. Hallucination guard.",
    "hallucination_rate":   "1 − Faithfulness. Lower is better.",
    "top_chunk_relevance":  "Coverage of the question by the single best chunk.",
    "context_relevance":    "Mean coverage of the question across all retrieved chunks.",
    "context_coverage":     "How much of the retrieved material the answer actually drew from.",
    "context_diversity":    "How non-redundant the chunks are (1 = all distinct).",
    "context_utilization":  "Fraction of retrieved chunks that meaningfully contributed.",
    "answer_relevance":     "Token overlap between the question and the answer.",
}


def _render_metrics_minimal(metrics: dict) -> None:
    """Compact one-glance metrics view rendered under the chat answer."""
    if not metrics or not isinstance(metrics, dict) or "error" in metrics:
        if metrics and "error" in metrics:
            st.caption(f"Quality metrics unavailable: {metrics['error']}")
        return

    retr = metrics.get("retrieval", {})
    ans = metrics.get("answer", {})
    lat = metrics.get("latency", {}) or {}

    faith = ans.get("faithfulness")
    hall = ans.get("hallucination_rate")
    ctx_rel = retr.get("context_relevance")
    top_rel = retr.get("top_chunk_relevance")

    # Single compact line of badges.
    line = (
        f"{_quality_label(faith)} **Faithfulness** {_fmt_score(faith)}  ·  "
        f"🧪 **Hallucination** {_fmt_score(hall)}  ·  "
        f"{_quality_label(top_rel)} **Top Chunk** {_fmt_score(top_rel)}  ·  "
        f"{_quality_label(ctx_rel)} **Context Rel.** {_fmt_score(ctx_rel)}  ·  "
        f"⏱ {_fmt_ms(lat.get('total_ms'))}"
    )
    st.caption(line)

    with st.expander("Full metrics breakdown", expanded=False):
        _render_metrics_full(metrics, in_expander=False)


def _render_metrics_full(metrics: dict, in_expander: bool = True) -> None:
    """Full table + sub-tables view — used in History + 'View all' expanders."""
    if not metrics or "error" in metrics:
        if metrics and "error" in metrics:
            st.caption(f"Quality metrics unavailable: {metrics['error']}")
        return

    retr = metrics.get("retrieval", {})
    ans = metrics.get("answer", {})
    lat = metrics.get("latency", {}) or {}
    resources_r = lat.get("retrieval_resources", {})
    resources_g = lat.get("generation_resources", {})

    container = st.expander("📈 Quality + Resources", expanded=False) if in_expander else st.container()

    with container:
        st.markdown("**Answer / RAG**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness",         _fmt_score(ans.get("faithfulness")),       help=_HELP["faithfulness"])
        c2.metric("Hallucination Rate",   _fmt_score(ans.get("hallucination_rate")), help=_HELP["hallucination_rate"])
        c3.metric("Answer Relevance",     _fmt_score(ans.get("answer_relevance")),   help=_HELP["answer_relevance"])
        c4.metric("Context Utilization",  _fmt_score(ans.get("context_utilization")),help=_HELP["context_utilization"])

        st.markdown("**Retrieval**")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Top Chunk Relevance",  _fmt_score(retr.get("top_chunk_relevance")), help=_HELP["top_chunk_relevance"])
        c6.metric("Context Relevance",    _fmt_score(retr.get("context_relevance")),   help=_HELP["context_relevance"])
        c7.metric("Context Coverage",     _fmt_score(retr.get("context_coverage")),    help=_HELP["context_coverage"])
        c8.metric("Context Diversity",    _fmt_score(retr.get("context_diversity")),   help=_HELP["context_diversity"])

        st.markdown("**Latency**")
        c9, c10, c11 = st.columns(3)
        c9.metric("Retrieval",  _fmt_ms(lat.get("retrieval_ms")))
        c10.metric("Generation", _fmt_ms(lat.get("generation_ms")))
        c11.metric("Total",      _fmt_ms(lat.get("total_ms")))

        st.markdown("**Resources**")
        c12, c13, c14, c15 = st.columns(4)
        c12.metric("CPU (retrieval)",  _fmt_pct(resources_r.get("cpu_percent")))
        c13.metric("CPU (generation)", _fmt_pct(resources_g.get("cpu_percent")))
        c14.metric("RAM used",         _fmt_mb(lat.get("ram_used_mb")))
        if lat.get("gpu_available", True) is False:
            c15.metric("GPU", "N/A", help="No CUDA device available on this machine")
        else:
            c15.metric("GPU memory", _fmt_mb(lat.get("gpu_mem_mb")))


def _record_comparison_row(question, answer, metrics, encoder, k):
    """Append one row to the comparison/evaluation table."""
    if not metrics or "error" in metrics:
        return None
    retr = metrics.get("retrieval", {})
    ans = metrics.get("answer", {})
    lat = metrics.get("latency", {}) or {}

    run_id = str(uuid.uuid4())
    row = {
        "_run_id":          run_id,
        "Time":             datetime.now().strftime("%H:%M:%S"),
        "Question":         question,
        "Encoder":          metrics.get("encoder", encoder),
        "k":                k,
        # AUTO — answer
        "Faithfulness":     ans.get("faithfulness"),
        "Halluc. Rate":     ans.get("hallucination_rate"),
        "Answer Rel.":      ans.get("answer_relevance"),
        "Ctx Util.":        ans.get("context_utilization"),
        # AUTO — retrieval
        "Top Chunk Rel.":   retr.get("top_chunk_relevance"),
        "Context Rel.":     retr.get("context_relevance"),
        "Context Cov.":     retr.get("context_coverage"),
        "Diversity":        retr.get("context_diversity"),
        # Latency
        "Retrieval (ms)":   lat.get("retrieval_ms"),
        "Generation (ms)":  lat.get("generation_ms"),
        "Total (ms)":       lat.get("total_ms"),
        # Resources
        "CPU %":            lat.get("cpu_percent"),
        "RAM (MB)":         lat.get("ram_used_mb"),
        "GPU mem (MB)":     None if lat.get("gpu_available", True) is False else lat.get("gpu_mem_mb"),
        # MANUAL — generation (filled in by Manual Eval tab)
        "BLEU":             None,
        "ROUGE-1":          None,
        "ROUGE-2":          None,
        "ROUGE-L":          None,
        "METEOR":           None,
        "Exact Match":      None,
        "Token F1":         None,
        "BERTScore F1":     None,
        "Ans. Correctness": None,
        "Completeness":     None,
        # MANUAL — retrieval (filled in only when relevant_passages provided)
        "Precision@k":      None,
        "Recall@k":          None,
        "Hit Rate":         None,
        "MRR":              None,
        "nDCG@k":           None,
        # Reference for Manual Eval lookup
        "_question":        question,
        "_answer":          answer,
        "_chunks":          metrics.get("_chunks", []),
        "Answer":           (answer[:120] + "…") if len(answer) > 120 else answer,
    }
    st.session_state.comparison_rows.append(row)
    return run_id


def _merge_manual_into_row(run_id: str, manual: dict):
    if not manual or "error" in manual:
        return False
    gen = manual.get("generation", {}) or {}
    retr = manual.get("retrieval", {}) or {}
    for r in st.session_state.comparison_rows:
        if r.get("_run_id") == run_id:
            r["BLEU"]             = gen.get("bleu")
            r["ROUGE-1"]          = gen.get("rouge_1")
            r["ROUGE-2"]          = gen.get("rouge_2")
            r["ROUGE-L"]          = gen.get("rouge_l")
            r["METEOR"]           = gen.get("meteor")
            r["Exact Match"]      = gen.get("exact_match")
            r["Token F1"]         = gen.get("f1")
            r["BERTScore F1"]     = gen.get("bertscore_f1")
            r["Ans. Correctness"] = gen.get("answer_correctness")
            r["Completeness"]     = gen.get("completeness")
            if retr:
                r["Precision@k"] = retr.get("precision_at_k")
                r["Recall@k"]    = retr.get("recall_at_k")
                r["Hit Rate"]    = retr.get("hit_rate")
                r["MRR"]         = retr.get("mrr")
                r["nDCG@k"]      = retr.get("ndcg_at_k")
            return True
    return False


# ----- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    encoder_type = st.selectbox(
        "📦 Encoder Model",
        ["biobert", "medcpt", "bm25", "hybrid"],
        help="Model used to encode/index your document",
    )
    retriever_type = encoder_type
    st.text_input("🔍 Retriever Model", value=retriever_type, disabled=True)

    if encoder_type == "hybrid":
        st.caption("ℹ️ Hybrid uses BioBERT encoder + BM25 for retrieval")
    elif encoder_type in ["biobert", "medcpt"]:
        st.caption(f"ℹ️ {encoder_type.upper()} used for both encoding and retrieval")
    elif encoder_type == "bm25":
        st.caption("ℹ️ BM25 is keyword-based (no neural encoding)")

    k_chunks = st.slider("Chunks to retrieve", 1, 10, 3)

    if os.getenv("GROQ_API_KEY"):
        st.success("✅ API Key loaded")
    else:
        st.error("❌ Add GROQ_API_KEY to your environment")

    if st.session_state.session_id:
        st.info(f"📝 Session: {st.session_state.session_id[:8]}...")

    if st.button("Reset Session", use_container_width=True):
        if st.session_state.session_id:
            try:
                resp = requests.delete(
                    f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}", timeout=10
                )
                if not resp.ok and resp.status_code != 404:
                    _show_response_error("Failed to delete session", resp)
            except requests.RequestException as e:
                st.warning(f"Could not reach backend: {e}")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ----- main tabs --------------------------------------------------------------

tab_upload, tab_chat, tab_history, tab_eval, tab_manual = st.tabs([
    "📄 Upload & Process",
    "💬 Chat",
    "📊 Session History",
    "🆚 Evaluation Table",
    "📝 Manual Evaluation",
])


# ----- Upload tab -------------------------------------------------------------

with tab_upload:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload medical document", type=["pdf", "txt"])
    if uploaded_file:
        st.info(f"**{uploaded_file.name}** - {uploaded_file.size / 1024:.2f} KB")
        if st.button("Process Document", type="primary"):
            try:
                with st.spinner("Creating session..."):
                    sr = requests.post(
                        f"{API_BASE_URL}/api/sessions",
                        json={"encoder_type": encoder_type},
                        timeout=30,
                    )
                    if sr.status_code != 201:
                        _show_response_error("Failed to create session", sr)
                        st.stop()
                    session_id = sr.json()["session_id"]
                    st.session_state.session_id = session_id
                    st.success(f"✅ Session created: {session_id[:8]}...")

                with st.spinner("Uploading and processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    ur = requests.post(
                        f"{API_BASE_URL}/api/documents/upload",
                        files=files,
                        data={"session_id": session_id},
                        timeout=600,
                    )
                    if not ur.ok:
                        _show_response_error("Failed to upload document", ur)
                        st.stop()
                    res = ur.json()
                    st.success("✅ Document processed successfully!")
                    with st.expander("Processing Details"):
                        st.write(f"**Encoder**: {encoder_type}")
                        st.write(f"**Embeddings**: {res.get('num_embeddings', 'N/A')}")
                        st.write(f"**Session ID**: {session_id}")
                st.info("👉 Go to the **Chat** tab to ask questions!")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at {API_BASE_URL}.")
            except requests.RequestException as e:
                st.error(f"Network error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())


# ----- Chat tab ---------------------------------------------------------------

with tab_chat:
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
                if msg.get("metrics"):
                    _render_metrics_minimal(msg["metrics"])

        if question := st.chat_input("Ask about your document..."):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/api/query",
                            json={
                                "session_id": st.session_state.session_id,
                                "question": question,
                                "k": k_chunks,
                            },
                            timeout=300,
                        )
                        if r.ok:
                            result = r.json()
                            answer = result["answer"]
                            chunks = result["chunks"]
                            metrics = result.get("metrics") or {}
                            metrics["_chunks"] = chunks  # tucked away for Manual Eval

                            st.markdown(answer)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "chunks": chunks,
                                "metrics": metrics,
                            })

                            _record_comparison_row(question, answer, metrics, encoder_type, k_chunks)

                            with st.expander("View Retrieved Chunks"):
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.text(chunk)
                                    st.divider()

                            _render_metrics_minimal(metrics)
                        else:
                            try:
                                detail = r.json().get("detail", r.text)
                            except ValueError:
                                detail = r.text
                            err = f"Error {r.status_code}: {detail}"
                            st.error(err)
                            st.session_state.chat_history.append({"role": "assistant", "content": err})
                    except requests.RequestException as e:
                        err = f"Network error: {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role": "assistant", "content": err})


# ----- Session History tab ----------------------------------------------------

with tab_history:
    st.header("Conversation History")
    if st.session_state.session_id:
        try:
            r = requests.get(
                f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation",
                timeout=15,
            )
            if r.ok:
                history = r.json()
                messages = history.get("messages", [])
                if messages:
                    st.write(f"**Total messages**: {len(messages)}")
                    for i, msg in enumerate(messages):
                        with st.expander(f"Message {i + 1} — {msg.get('role', '').title()}"):
                            st.write(f"**Role**: {msg.get('role', '')}")
                            st.write(f"**Content**: {msg.get('content', '')}")
                            st.write(f"**Timestamp**: {msg.get('timestamp', '')}")
                            if msg.get("retrieved_chunks"):
                                st.write("**Retrieved Chunks:**")
                                for j, chunk in enumerate(msg["retrieved_chunks"], 1):
                                    st.text(f"{j}. {chunk[:200]}…")
                            if msg.get("metrics"):
                                st.write("**Quality Metrics:**")
                                _render_metrics_full(msg["metrics"], in_expander=False)
                else:
                    st.info("No conversation history yet")
                if st.button("Clear History"):
                    dr = requests.delete(
                        f"{API_BASE_URL}/api/sessions/{st.session_state.session_id}/conversation",
                        timeout=15,
                    )
                    if dr.status_code in (200, 204):
                        st.success("✅ History cleared")
                        st.rerun()
                    else:
                        _show_response_error("Failed to clear history", dr)
            else:
                _show_response_error("Error fetching history", r)
        except requests.RequestException as e:
            st.error(f"Network error: {e}")
    else:
        st.info("No active session")


# ----- Evaluation Table tab ---------------------------------------------------

with tab_eval:
    st.header("Evaluation Table")
    st.caption(
        "Every query you run is auto-logged here. Switch the encoder in the sidebar "
        "and re-ask the same question to compare. Reference-based metrics "
        "(BLEU, ROUGE, BERTScore, Precision@k, …) are filled in by the **Manual "
        "Evaluation** tab."
    )

    rows = st.session_state.comparison_rows
    if not rows:
        st.info("No runs yet. Ask a question in the Chat tab — it'll appear here.")
    else:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            qs = ["(all)"] + sorted({r["Question"] for r in rows})
            selected_q = st.selectbox("Filter by question", qs, index=0)
        with col_b:
            group_mode = st.checkbox("Group by question", value=True)
        with col_c:
            if st.button("🗑 Clear table", use_container_width=True):
                st.session_state.comparison_rows = []
                st.rerun()

        filtered = rows if selected_q == "(all)" else [r for r in rows if r["Question"] == selected_q]
        # Strip internal-only keys for display.
        display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in filtered]

        try:
            import pandas as pd
            df = pd.DataFrame(display_rows)
            if group_mode and "Question" in df.columns:
                df = df.sort_values(by=["Question", "Time"]).reset_index(drop=True)

            sections = {
                "Identity":            ["Time", "Question", "Encoder", "k"],
                "Auto — Answer/RAG":   ["Faithfulness", "Halluc. Rate", "Answer Rel.", "Ctx Util."],
                "Auto — Retrieval":    ["Top Chunk Rel.", "Context Rel.", "Context Cov.", "Diversity"],
                "Latency":             ["Retrieval (ms)", "Generation (ms)", "Total (ms)"],
                "Resources":           ["CPU %", "RAM (MB)", "GPU mem (MB)"],
                "Manual — Generation": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR",
                                        "Exact Match", "Token F1", "BERTScore F1",
                                        "Ans. Correctness", "Completeness"],
                "Manual — Retrieval":  ["Precision@k", "Recall@k", "Hit Rate", "MRR", "nDCG@k"],
                "Preview":             ["Answer"],
            }
            picked = st.multiselect(
                "Sections to show",
                list(sections.keys()),
                default=["Identity", "Auto — Answer/RAG", "Auto — Retrieval", "Latency", "Resources"],
            )
            cols_to_show = [c for s in picked for c in sections[s] if c in df.columns]
            st.dataframe(df[cols_to_show] if cols_to_show else df,
                         use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download full table as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_table.csv",
                mime="text/csv",
            )

            if group_mode and len(filtered) >= 2:
                st.markdown("##### Best encoder per question (by Faithfulness)")
                fdf = df.copy()
                fdf["Faithfulness"] = pd.to_numeric(fdf["Faithfulness"], errors="coerce")
                idx = fdf.groupby("Question")["Faithfulness"].idxmax().dropna()
                summary_cols = [c for c in ["Question", "Encoder", "Faithfulness",
                                            "Top Chunk Rel.", "Context Rel.",
                                            "Total (ms)", "CPU %", "RAM (MB)"] if c in fdf.columns]
                st.dataframe(fdf.loc[idx, summary_cols].reset_index(drop=True),
                             use_container_width=True, hide_index=True)
        except ImportError:
            st.dataframe(display_rows, use_container_width=True, hide_index=True)


# ----- Manual Evaluation tab --------------------------------------------------

with tab_manual:
    st.header("Manual (Reference-Based) Evaluation")
    st.caption(
        "Pick one of your previous answers, paste the **ground-truth answer**, "
        "and optionally paste **relevant passages** (one per line). "
        "We'll compute BLEU, ROUGE-1/2/L, METEOR, Exact Match, Token F1, BERTScore, "
        "Answer Correctness, Completeness, and (if relevant passages are given) "
        "Precision@k, Recall@k, Hit Rate, MRR, nDCG@k. Results are merged into the "
        "Evaluation Table."
    )

    rows = st.session_state.comparison_rows
    if not rows:
        st.info("Run at least one query in the Chat tab first.")
    else:
        labels = [
            f"[{r['Time']}] [{r['Encoder']}] {r['Question'][:80]}"
            + ("  ✅ already evaluated" if r.get("BLEU") is not None else "")
            for r in rows
        ]
        idx = st.selectbox("Pick a run to evaluate", range(len(rows)),
                           format_func=lambda i: labels[i])
        target = rows[idx]

        with st.expander("Selected run", expanded=True):
            st.write(f"**Question:** {target['_question']}")
            st.write(f"**Encoder:** {target['Encoder']}, **k:** {target['k']}")
            st.markdown("**Generated Answer:**")
            st.write(target["_answer"])

        reference_answer = st.text_area(
            "Ground-truth answer (required)",
            height=160,
            placeholder="Paste the correct / expected answer here…",
        )
        relevant_text = st.text_area(
            "Relevant passages (optional, one per line)",
            height=160,
            placeholder="Paste passages (snippets of text) you consider relevant for this question.\n"
                        "If provided, retrieval metrics (Precision@k, Recall@k, Hit Rate, MRR, nDCG@k) are computed.",
        )

        if st.button("🧮 Compute metrics", type="primary"):
            if not reference_answer.strip():
                st.error("Please paste a ground-truth answer first.")
            else:
                relevant_passages = [
                    line.strip() for line in relevant_text.splitlines() if line.strip()
                ]
                with st.spinner("Computing reference-based metrics (first BERTScore run downloads a model)…"):
                    try:
                        resp = requests.post(
                            f"{API_BASE_URL}/api/evaluation/manual",
                            json={
                                "question":          target["_question"],
                                "generated_answer":  target["_answer"],
                                "reference_answer":  reference_answer,
                                "retrieved_chunks":  target.get("_chunks", []),
                                "relevant_passages": relevant_passages,
                            },
                            timeout=600,
                        )
                        if not resp.ok:
                            _show_response_error("Manual evaluation failed", resp)
                        else:
                            manual = resp.json()
                            ok = _merge_manual_into_row(target["_run_id"], manual)
                            if ok:
                                st.success("✅ Metrics computed and merged into the Evaluation Table.")
                            gen = manual.get("generation", {})
                            retr_m = manual.get("retrieval", {})

                            st.markdown("##### Generation metrics")
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("BLEU",          _fmt_score(gen.get("bleu")))
                            c2.metric("ROUGE-1",       _fmt_score(gen.get("rouge_1")))
                            c3.metric("ROUGE-2",       _fmt_score(gen.get("rouge_2")))
                            c4.metric("ROUGE-L",       _fmt_score(gen.get("rouge_l")))
                            c5.metric("METEOR",        _fmt_score(gen.get("meteor")))
                            c6, c7, c8, c9, c10 = st.columns(5)
                            c6.metric("Exact Match",   _fmt_score(gen.get("exact_match")))
                            c7.metric("Token F1",      _fmt_score(gen.get("f1")))
                            c8.metric("BERTScore F1",  _fmt_score(gen.get("bertscore_f1")))
                            c9.metric("Ans. Correctness", _fmt_score(gen.get("answer_correctness")))
                            c10.metric("Completeness", _fmt_score(gen.get("completeness")))

                            if not manual.get("bertscore_available"):
                                st.caption("ℹ️ BERTScore unavailable in this environment — Answer Correctness fell back to Token F1.")
                            if not manual.get("meteor_available"):
                                st.caption("ℹ️ METEOR unavailable (NLTK WordNet missing).")

                            if retr_m:
                                st.markdown("##### Retrieval metrics (vs. provided relevant passages)")
                                d1, d2, d3, d4, d5 = st.columns(5)
                                d1.metric("Precision@k", _fmt_score(retr_m.get("precision_at_k")))
                                d2.metric("Recall@k",    _fmt_score(retr_m.get("recall_at_k")))
                                d3.metric("Hit Rate",    _fmt_score(retr_m.get("hit_rate")))
                                d4.metric("MRR",         _fmt_score(retr_m.get("mrr")))
                                d5.metric("nDCG@k",      _fmt_score(retr_m.get("ndcg_at_k")))
                            else:
                                st.caption("ℹ️ Retrieval metrics skipped — no relevant passages provided.")
                    except requests.RequestException as e:
                        st.error(f"Network error: {e}")


# ----- Footer -----------------------------------------------------------------

st.divider()
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    st.caption(f"📦 Encoder: **{encoder_type}**")
with fc2:
    st.caption(f"🔍 Retriever: **{retriever_type}**")
with fc3:
    st.caption(f"📊 Chunks: **{k_chunks}**")
with fc4:
    st.caption(f"📝 Session: **{'Active' if st.session_state.session_id else 'None'}**")
