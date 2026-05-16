"""Streamlit frontend for the Financial Reports Q&A RAG system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from retrieval.rag_pipeline import FinancialRAG

st.set_page_config(
    page_title="Financial Reports Q&A",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading vector store...")
def get_rag() -> FinancialRAG:
    """Load and cache the RAG pipeline."""
    return FinancialRAG()


_COMPANY_NAMES = {"apple", "tesla", "microsoft", "nvidia", "google", "amazon", "meta"}

def _is_comparison_query(q: str) -> bool:
    """Return True if the question appears to compare multiple companies."""
    q_lower = q.lower()
    companies_mentioned = sum(1 for c in _COMPANY_NAMES if c in q_lower)
    compare_keywords = any(w in q_lower for w in ["compare", "vs", "versus", "difference", "both"])
    return companies_mentioned >= 2 or compare_keywords


def _render_answer_block(entry: dict, expanded: bool = True) -> None:
    """Render a single Q&A history entry."""
    st.markdown(f"**Q: {entry['question']}**")
    if entry.get("mode_label"):
        st.caption(entry["mode_label"])
    st.markdown(entry["answer"])
    if entry["sources"]:
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for i, src in enumerate(entry["sources"], 1):
                st.markdown(
                    f"**{i}. {src['company']} {src['year']}** — `{src['source_file']}`"
                )
                st.text(src["excerpt"])
                if i < len(entry["sources"]):
                    st.markdown("---")


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Filters")

    # Index stats
    try:
        _stats = get_rag().stats()
        year_range = (
            f"{_stats['years'][0]}–{_stats['years'][-1]}"
            if len(_stats["years"]) > 1
            else str(_stats["years"][0]) if _stats["years"] else "N/A"
        )
        company_str = ", ".join(
            f"{c} ({n:,})" for c, n in sorted(_stats["companies"].items())
        )
        st.caption(
            f"**{_stats['total_chunks']:,} chunks indexed** · {company_str} · Years: {year_range}"
        )
    except Exception:
        pass

    st.markdown("---")

    try:
        _companies = sorted(_stats["companies"].keys())
        _years = _stats["years"]
    except Exception:
        _companies = ["Apple", "Tesla"]
        _years = [2023, 2024, 2025]

    company_options = ["All"] + _companies
    selected_companies = st.multiselect(
        "Company",
        options=company_options,
        default=["All"],
    )

    year_options = ["All"] + _years
    selected_years = st.multiselect(
        "Year",
        options=year_options,
        default=["All"],
    )

    mode = st.radio(
        "Mode",
        options=["Single Query", "Compare Companies"],
    )

    st.markdown("---")
    st.markdown("**Example Questions**")
    examples = [
        "What was Apple's total revenue in 2024?",
        "How did Nvidia's revenue grow from 2022 to 2025?",
        "Compare Microsoft and Google's cloud revenue",
        "What risks did Meta highlight in their 2024 annual report?",
        "How did Amazon explain its profitability improvement?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["question_input"] = ex

    st.markdown("---")
    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("Financial Reports Q&A")
st.caption("Ask questions about Apple and Tesla 10-K filings, grounded in the source documents.")

tab_qa, tab_about = st.tabs(["Q&A", "About"])

with tab_qa:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What was Apple's total revenue in 2024?",
        key="question_input",
    )

    submit = st.button("Submit", type="primary")

    if submit and question.strip():
        rag = get_rag()

        # Resolve filters
        company_filter = None
        filtered_companies = [c for c in selected_companies if c != "All"]
        if filtered_companies:
            company_filter = filtered_companies[0] if len(filtered_companies) == 1 else filtered_companies

        year_filter = None
        filtered_years = [int(y) for y in selected_years if y != "All"]
        if filtered_years:
            year_filter = filtered_years[0] if len(filtered_years) == 1 else filtered_years

        # Auto-upgrade to Compare mode when query is clearly about multiple companies
        effective_mode = mode
        mode_label = None
        if mode == "Single Query" and _is_comparison_query(question):
            effective_mode = "Compare Companies"
            mode_label = "Comparison question detected — using Compare Companies mode automatically."
            st.info(mode_label)

        try:
            with st.spinner("Searching documents and generating answer..."):
                if effective_mode == "Compare Companies":
                    companies = [c for c in selected_companies if c != "All"]
                    if len(companies) < 2:
                        companies = ["Apple", "Tesla"]
                    result = rag.compare(question, companies)
                    answer_text = result["comparison"]
                    sources = result["sources"]
                else:
                    result = rag.query(question, company_filter=company_filter, year_filter=year_filter)
                    answer_text = result["answer"]
                    sources = result["sources"]
        except Exception as e:
            st.error(f"Something went wrong while generating the answer: {e}")
            st.stop()

        # Prepend to history (newest first)
        st.session_state.history.insert(0, {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "mode_label": mode_label,
        })

        st.markdown("### Answer")
        st.markdown(answer_text)

        if sources:
            st.markdown(f"### Sources ({len(sources)})")
            for i, src in enumerate(sources, 1):
                with st.expander(f"Source {i} — {src['company']} {src['year']} ({src['source_file']})"):
                    st.text(src["excerpt"])

    elif submit and not question.strip():
        st.warning("Please enter a question.")

    # ── History ───────────────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### Previous Questions")
        for entry in st.session_state.history[1:]:
            with st.expander(f"Q: {entry['question']}"):
                _render_answer_block(entry)


with tab_about:
    st.markdown("## How it works")
    st.markdown(
        "This app uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer "
        "questions grounded strictly in Apple and Tesla 10-K filings. No figures are "
        "fabricated — every answer cites the source document it came from."
    )

    st.markdown("### Pipeline")
    st.markdown(
        """
| Step | What happens |
|------|-------------|
| **1. Parse** | SEC 10-K HTML filings are parsed with BeautifulSoup. Financial tables are converted to labeled rows so figures stay attached to their row labels. |
| **2. Chunk** | Each document is split into ~800-character overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. |
| **3. Embed** | Chunks are embedded with OpenAI `text-embedding-3-small` and stored in a local ChromaDB vector store. |
| **4. Retrieve** | At query time, the top-K most similar chunks are retrieved. Comparison queries fetch chunks separately per company to ensure both are represented. |
| **5. Generate** | Retrieved chunks are passed as context to `gpt-4o-mini`, which answers strictly based on what the documents say. |
"""
    )

    st.markdown("### Documents indexed")
    try:
        _stats = get_rag().stats()
        cols = st.columns(len(_stats["companies"]))
        for col, (company, count) in zip(cols, sorted(_stats["companies"].items())):
            col.metric(label=company, value=f"{count:,}", delta="chunks")
        st.caption(f"Years covered: {', '.join(str(y) for y in _stats['years'])}")
    except Exception:
        st.info("Load the Q&A tab first to see stats.")

    st.markdown("### Tech stack")
    st.markdown(
        "- **LangChain** — document loading, splitting, retrieval\n"
        "- **ChromaDB** — local persistent vector store\n"
        "- **OpenAI** — embeddings (`text-embedding-3-small`) + generation (`gpt-4o-mini`)\n"
        "- **Streamlit** — frontend\n"
        "- **RAGAS** — evaluation framework for faithfulness and relevancy scoring"
    )
