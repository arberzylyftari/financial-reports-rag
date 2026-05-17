"""Streamlit frontend for the Financial Reports Q&A RAG system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import json
import plotly.graph_objects as go
import streamlit as st
from retrieval.rag_pipeline import FinancialRAG

HISTORY_FILE = Path(__file__).parent / "chat_history.json"


def _load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def _save_history(history: list[dict]) -> None:
    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception:
        pass

st.set_page_config(
    page_title="Financial Reports Q&A",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading vector store...")
def get_rag() -> FinancialRAG:
    """Load and cache the RAG pipeline."""
    return FinancialRAG()


_COMPANY_MAP = {
    "apple": "Apple", "tesla": "Tesla", "microsoft": "Microsoft",
    "nvidia": "Nvidia", "google": "Google", "amazon": "Amazon", "meta": "Meta",
}
_COMPANY_NAMES = set(_COMPANY_MAP.keys())


def _is_comparison_query(q: str) -> bool:
    """Return True if the question appears to compare multiple companies."""
    q_lower = q.lower()
    companies_mentioned = sum(1 for c in _COMPANY_NAMES if c in q_lower)
    compare_keywords = any(w in q_lower for w in ["compare", "vs", "versus", "difference", "both"])
    return companies_mentioned >= 2 or compare_keywords


def _extract_companies_from_query(q: str) -> list[str]:
    """Return properly-cased company names mentioned in the query."""
    q_lower = q.lower()
    return [name for key, name in _COMPANY_MAP.items() if key in q_lower]


def _history_to_markdown(history: list[dict]) -> str:
    """Convert chat history to a readable Markdown document."""
    lines = ["# Financial Reports Q&A — Conversation Export", ""]
    for entry in history:
        lines.append(f"## Q: {entry['question']}")
        if entry.get("mode_label"):
            lines.append(f"*{entry['mode_label']}*")
        lines.append("")
        lines.append(entry["answer"])
        lines.append("")
        if entry["sources"]:
            lines.append(f"**Sources ({len(entry['sources'])}):**")
            for src in entry["sources"]:
                lines.append(
                    f"- {src['company']} {src['year']} — `{src['source_file']}`"
                )
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = _load_history()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Filters")

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
            f"**{_stats['total_chunks']:,} chunks indexed** · Years: {year_range}"
        )
    except Exception:
        _stats = None

    st.markdown("---")

    try:
        _companies = sorted(_stats["companies"].keys())
        _years = _stats["years"]
    except Exception:
        _companies = list(_COMPANY_NAMES)
        _years = list(range(2018, 2026))

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
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.session_state.history:
        st.download_button(
            label="Download conversation",
            data=_history_to_markdown(st.session_state.history),
            file_name="financial_qa_conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            _save_history([])
            st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("Financial Reports Q&A")
st.caption(
    "Chat with 7 companies' SEC 10-K filings (2018–2025). "
    "Every answer is grounded in the source documents."
)

tab_qa, tab_charts, tab_about = st.tabs(["Chat", "Charts", "About"])

with tab_qa:

    # Render full conversation history (oldest first → newest at bottom, just above input)
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
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

    # Prefill from sidebar example buttons
    prefill = st.session_state.pop("prefill", "")

    question = st.chat_input(
        "Ask a question about any company's financials…",
        key="chat_input",
    )

    # Use prefill if example button was clicked
    if prefill and not question:
        question = prefill

    if question:
        # Resolve filters
        company_filter = None
        filtered_companies = [c for c in selected_companies if c != "All"]
        if filtered_companies:
            company_filter = filtered_companies[0] if len(filtered_companies) == 1 else filtered_companies

        year_filter = None
        filtered_years = [int(y) for y in selected_years if y != "All"]
        if filtered_years:
            year_filter = filtered_years[0] if len(filtered_years) == 1 else filtered_years

        effective_mode = mode
        mode_label = None
        if mode == "Single Query" and _is_comparison_query(question):
            effective_mode = "Compare Companies"
            mode_label = "Comparison question detected — switching to Compare Companies mode."

        # Build chat history for multi-turn context (last 3 turns)
        chat_history = []
        for entry in st.session_state.history[-3:]:
            chat_history.append({"role": "user", "content": entry["question"]})
            chat_history.append({"role": "assistant", "content": entry["answer"]})

        try:
            rag = get_rag()
            with st.spinner("Retrieving documents…"):
                if effective_mode == "Compare Companies":
                    companies = [c for c in selected_companies if c != "All"]
                    if len(companies) < 2:
                        companies = _extract_companies_from_query(question)
                    if len(companies) < 2:
                        companies = list(_COMPANY_MAP.values())
                    stream, sources = rag.compare_stream(question, companies, chat_history=chat_history)
                else:
                    stream, sources = rag.query_stream(
                        question,
                        company_filter=company_filter,
                        year_filter=year_filter,
                        chat_history=chat_history,
                    )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        with st.chat_message("assistant"):
            if mode_label:
                st.caption(mode_label)
            answer_text = st.write_stream(
                chunk.content for chunk in stream if hasattr(chunk, "content")
            )
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}. {src['company']} {src['year']}** — `{src['source_file']}`"
                        )
                        st.text(src["excerpt"])
                        if i < len(sources):
                            st.markdown("---")

        # Save to history, persist to disk, rerun so new entry renders in the loop above
        st.session_state.history.append({
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "mode_label": mode_label,
        })
        _save_history(st.session_state.history)
        st.rerun()


with tab_charts:
    st.markdown("### Financial Metrics Explorer")
    st.caption("Extract and visualize key metrics from the indexed 10-K filings.")

    METRICS = [
        "Total Revenue",
        "Net Income",
        "R&D Expense",
        "Operating Income",
        "Gross Profit",
    ]

    col1, col2 = st.columns([1, 2])
    with col1:
        chart_metric = st.selectbox("Metric", METRICS)
        chart_type = st.radio("Chart type", ["Bar", "Line"])
    with col2:
        chart_companies = st.multiselect(
            "Companies",
            options=_companies,
            default=_companies[:3] if len(_companies) >= 3 else _companies,
        )
        if _years:
            year_range = st.slider(
                "Year range",
                min_value=int(_years[0]),
                max_value=int(_years[-1]),
                value=(int(_years[0]), int(_years[-1])),
            )
            selected_chart_years = [y for y in _years if year_range[0] <= y <= year_range[1]]
        else:
            selected_chart_years = []

    if st.button("Generate Chart", type="primary"):
        if not chart_companies:
            st.warning("Select at least one company.")
        elif not selected_chart_years:
            st.warning("Select a valid year range.")
        else:
            cache_key = f"chart_{chart_metric}_{'_'.join(chart_companies)}_{selected_chart_years[0]}_{selected_chart_years[-1]}"
            if cache_key not in st.session_state:
                rag = get_rag()
                results = {}
                progress = st.progress(0, text="Fetching data…")
                for idx, company in enumerate(chart_companies):
                    results[company] = rag.extract_metric_series(
                        company, selected_chart_years, chart_metric
                    )
                    progress.progress((idx + 1) / len(chart_companies), text=f"Fetched {company}…")
                progress.empty()
                st.session_state[cache_key] = results
            else:
                results = st.session_state[cache_key]

            fig = go.Figure()
            for company, series in results.items():
                years = [y for y in selected_chart_years if series.get(y) is not None]
                values = [series[y] / 1000 for y in years]  # convert M → B
                if chart_type == "Bar":
                    fig.add_trace(go.Bar(name=company, x=years, y=values))
                else:
                    fig.add_trace(go.Scatter(name=company, x=years, y=values, mode="lines+markers"))

            fig.update_layout(
                title=f"{chart_metric} (USD billions)",
                xaxis_title="Fiscal Year",
                yaxis_title="USD (billions)",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Values extracted by LLM from SEC 10-K filings. Verify against source documents.")


with tab_about:
    st.markdown("## How it works")
    st.markdown(
        "This app uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer "
        "questions grounded strictly in SEC 10-K annual filings from 7 major tech companies. "
        "No figures are fabricated — every answer cites the source document it came from."
    )

    st.markdown("### Pipeline")
    st.markdown(
        """
| Step | What happens |
|------|-------------|
| **1. Parse** | SEC 10-K HTML filings are parsed with BeautifulSoup. Financial tables are converted to labeled rows so figures stay attached to their row labels and column headers. |
| **2. Chunk** | Each document is split into ~800-character overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`. |
| **3. Embed** | Chunks are embedded with OpenAI `text-embedding-3-small` and stored in a local ChromaDB vector store. |
| **4. Retrieve** | At query time, the top-K most similar chunks are retrieved. Comparison queries fetch chunks separately per company to ensure all companies are represented. |
| **5. Generate** | Retrieved chunks are passed as context to `gpt-4o-mini`, which answers strictly based on what the documents say. |
"""
    )

    st.markdown("### Documents indexed")
    try:
        _stats = get_rag().stats()
        cols = st.columns(min(len(_stats["companies"]), 4))
        for i, (company, count) in enumerate(sorted(_stats["companies"].items())):
            cols[i % len(cols)].metric(label=company, value=f"{count:,}", delta="chunks")
        st.caption(f"Years covered: {', '.join(str(y) for y in _stats['years'])}")
    except Exception:
        st.info("Load the Chat tab first to see stats.")

    st.markdown("### Tech stack")
    st.markdown(
        "- **LangChain** — document loading, splitting, retrieval\n"
        "- **ChromaDB** — local persistent vector store\n"
        "- **OpenAI** — embeddings (`text-embedding-3-small`) + generation (`gpt-4o-mini`)\n"
        "- **Streamlit** — frontend\n"
        "- **RAGAS** — evaluation framework for faithfulness and relevancy scoring"
    )
