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


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Filters")

    company_options = ["All", "Apple", "Tesla"]
    selected_companies = st.multiselect(
        "Company",
        options=company_options,
        default=["All"],
    )

    year_options = ["All", 2023, 2024, 2025]
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
        "How did Tesla explain declining margins?",
        "Compare Apple and Tesla's R&D spending",
        "What risks did Apple highlight in their latest annual report?",
    ]
    for ex in examples:
        st.markdown(f"- _{ex}_")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("Financial Reports Q&A")
st.caption("Ask questions about Apple and Tesla 10-K filings, grounded in the source documents.")

question = st.text_input(
    "Your question",
    placeholder="e.g. What was Apple's total revenue in 2024?",
)

submit = st.button("Submit", type="primary")

if submit and question.strip():
    rag = get_rag()

    # Resolve filters
    company_filter = None
    if "All" not in selected_companies and len(selected_companies) == 1:
        company_filter = selected_companies[0]

    year_filter = None
    if "All" not in selected_years and len(selected_years) == 1:
        year_filter = int(selected_years[0])

    with st.spinner("Searching documents and generating answer..."):
        if mode == "Compare Companies":
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

    st.markdown("### Answer")
    st.info(answer_text)

    if sources:
        st.markdown("### Sources")
        for i, src in enumerate(sources, 1):
            with st.expander(f"Source {i} — {src['company']} {src['year']} ({src['source_file']})"):
                st.markdown(f"```\n{src['excerpt']}\n```")

elif submit and not question.strip():
    st.warning("Please enter a question.")
