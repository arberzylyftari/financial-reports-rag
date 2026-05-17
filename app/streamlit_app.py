"""Streamlit frontend for the Financial Reports Q&A RAG system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import csv
import io
import json
import plotly.graph_objects as go
import streamlit as st
from retrieval.rag_pipeline import FinancialRAG

HISTORY_FILE = Path(__file__).parent / "chat_history.json"

# GPT-4o-mini pricing (USD per token)
_INPUT_COST_PER_TOKEN  = 0.150 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 0.600 / 1_000_000


def _stream_and_capture_usage(stream, usage_out: dict):
    """Yield text chunks from an LLM stream, capturing token usage into usage_out."""
    for chunk in stream:
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage_out.update(chunk.usage_metadata)
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


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

COMPANY_COLORS = {
    "Apple":     "#007AFF",
    "Microsoft": "#00BCF2",
    "Nvidia":    "#76B900",
    "Google":    "#EA4335",
    "Amazon":    "#FF9900",
    "Meta":      "#0866FF",
    "Tesla":     "#CC0000",
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

    # Session cost tracker
    session_input  = sum(e.get("input_tokens",  0) for e in st.session_state.history)
    session_output = sum(e.get("output_tokens", 0) for e in st.session_state.history)
    session_cost   = session_input * _INPUT_COST_PER_TOKEN + session_output * _OUTPUT_COST_PER_TOKEN
    if st.session_state.history:
        col_a, col_b = st.columns(2)
        col_a.metric("Session tokens", f"{session_input + session_output:,}")
        col_b.metric("Est. cost", f"${session_cost:.4f}")

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

    TOPIC_FILTERS = {
        "Any topic": "",
        "Revenue & Profit": "Focus on revenue, net income, and profitability.",
        "Cash Flow": "Focus on cash flow from operations, free cash flow, and liquidity.",
        "R&D & Innovation": "Focus on research and development spending and innovation strategy.",
        "Risk Factors": "Focus on risks, uncertainties, and challenges disclosed in the filing.",
        "Guidance & Outlook": "Focus on forward-looking statements, guidance, and strategic outlook.",
        "Segments & Geography": "Focus on business segments, product lines, and geographic revenue breakdown.",
    }
    selected_topic = st.selectbox("Topic focus", options=list(TOPIC_FILTERS.keys()))

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

tab_qa, tab_charts, tab_dashboard, tab_about = st.tabs(["Chat", "Charts", "Dashboard", "About"])

with tab_qa:

    # Render full conversation history (oldest first → newest at bottom, just above input)
    for h_idx, entry in enumerate(st.session_state.history):
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
            if entry.get("followups"):
                st.caption("You might also ask:")
                for f_idx, fq in enumerate(entry["followups"]):
                    if st.button(fq, key=f"fq_{h_idx}_{f_idx}", use_container_width=True):
                        st.session_state["prefill"] = fq

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

        # Apply topic prefix to steer the LLM focus
        topic_prefix = TOPIC_FILTERS.get(selected_topic, "")
        augmented_question = f"{topic_prefix} {question}".strip() if topic_prefix else question

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
                    stream, sources = rag.compare_stream(augmented_question, companies, chat_history=chat_history)
                else:
                    stream, sources = rag.query_stream(
                        augmented_question,
                        company_filter=company_filter,
                        year_filter=year_filter,
                        chat_history=chat_history,
                    )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        usage_out: dict = {}
        with st.chat_message("assistant"):
            if mode_label:
                st.caption(mode_label)
            answer_text = st.write_stream(_stream_and_capture_usage(stream, usage_out))
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}. {src['company']} {src['year']}** — `{src['source_file']}`"
                        )
                        st.text(src["excerpt"])
                        if i < len(sources):
                            st.markdown("---")
            with st.spinner("Generating follow-up suggestions…"):
                followups = rag.suggest_followups(question, answer_text)

        input_tokens  = usage_out.get("input_tokens",  0)
        output_tokens = usage_out.get("output_tokens", 0)

        # Save to history, persist to disk, rerun so new entry renders in the loop above
        st.session_state.history.append({
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "mode_label": mode_label,
            "followups": followups,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
        _save_history(st.session_state.history)
        st.rerun()


with tab_charts:
    st.markdown("### Financial Metrics Explorer")
    st.caption("Extract and visualize key metrics from the indexed 10-K filings.")

    METRICS = [
        "Total Revenue",
        "Net Income",
        "Gross Profit",
        "Operating Income",
        "R&D Expense",
        "Free Cash Flow",
        "Total Assets",
        "Cash and Equivalents",
        "Gross Margin %",
        "Operating Margin %",
    ]
    PERCENTAGE_METRICS = {"Gross Margin %", "Operating Margin %"}

    col1, col2 = st.columns([1, 2])
    with col1:
        chart_metric = st.selectbox("Metric", METRICS)
        chart_type = st.radio("Chart type", ["Bar", "Line", "Area"])
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
            # Store render params so chart survives download button reruns
            st.session_state["chart_render"] = {
                "key": cache_key,
                "metric": chart_metric,
                "type": chart_type,
                "years": selected_chart_years,
            }

    # Render chart from session state — persists across download button reruns
    if "chart_render" in st.session_state:
        render = st.session_state["chart_render"]
        results = st.session_state.get(render["key"], {})
        r_metric = render["metric"]
        r_type   = render["type"]
        r_years  = render["years"]

        if results:
            is_pct  = r_metric in PERCENTAGE_METRICS
            y_label = "%" if is_pct else "USD (billions)"
            title   = f"{r_metric} ({'%' if is_pct else 'USD billions'})"

            fig = go.Figure()
            for company, series in results.items():
                years  = [y for y in r_years if series.get(y) is not None]
                values = [series[y] if is_pct else series[y] / 1000 for y in years]
                color  = COMPANY_COLORS.get(company, "#888888")
                if r_type == "Bar":
                    fig.add_trace(go.Bar(name=company, x=years, y=values, marker_color=color))
                elif r_type == "Area":
                    fig.add_trace(go.Scatter(name=company, x=years, y=values,
                                             mode="lines", fill="tozeroy",
                                             line=dict(color=color), opacity=0.7))
                else:
                    fig.add_trace(go.Scatter(name=company, x=years, y=values,
                                             mode="lines+markers", line=dict(color=color),
                                             marker=dict(color=color)))

            fig.update_layout(
                title=title,
                xaxis_title="Fiscal Year",
                yaxis_title=y_label,
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "toImageButtonOptions": {"format": "png", "filename": r_metric.replace(" ", "_"), "scale": 2},
                    "displaylogo": False,
                },
            )

            # Light-themed copy for HTML export (readable on white background)
            fig_export = go.Figure(fig)
            fig_export.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#222222"),
                xaxis=dict(gridcolor="#EEEEEE", linecolor="#CCCCCC"),
                yaxis=dict(gridcolor="#EEEEEE", linecolor="#CCCCCC"),
            )

            dl_col1, dl_col2 = st.columns([1, 5])
            with dl_col1:
                st.download_button(
                    "Download chart (HTML)",
                    data=fig_export.to_html(include_plotlyjs="cdn"),
                    file_name=f"{r_metric.replace(' ', '_')}.html",
                    mime="text/html",
                    key="dl_html",
                )
            st.caption("Values extracted by LLM from SEC 10-K filings. Verify against source documents.")

            # Raw data table
            st.markdown("#### Data Table")
            table_rows = {}
            for company, series in results.items():
                table_rows[company] = {
                    str(y): (
                        f"{series[y]:.1f}%" if is_pct and series.get(y) is not None
                        else f"${series[y]/1000:.2f}B" if series.get(y) is not None
                        else "—"
                    )
                    for y in r_years
                }
            st.dataframe(table_rows, use_container_width=True)

            # CSV — clean layout with metadata header and labeled columns
            unit_label = "%" if is_pct else "M USD"
            col_headers = [f"{y} ({unit_label})" for y in r_years]

            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["=== Financial Reports RAG ==="])
            writer.writerow([f"Metric: {r_metric}"])
            writer.writerow([f"Unit: {'Percentage (%)' if is_pct else 'Millions USD — chart displays values as Billions USD'}"])
            writer.writerow(["Source: SEC 10-K annual filings. Values extracted by LLM — verify against source documents before use."])
            writer.writerow([])
            writer.writerow(["Company"] + col_headers)
            for company, series in results.items():
                def _fmt(v):
                    if v is None:
                        return "N/A"
                    return f"{v:.2f}%" if is_pct else f"{v:,.0f}"
                writer.writerow([company] + [_fmt(series.get(y)) for y in r_years])
            st.download_button(
                "Download data (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"{r_metric.replace(' ', '_')}_data.csv",
                mime="text/csv",
                key="dl_csv",
            )

            # YoY growth overlay
            growth_data = {}
            for company, series in results.items():
                yrs = sorted(y for y in r_years if series.get(y) is not None)
                if len(yrs) >= 2:
                    growth_data[company] = {
                        yrs[i]: round(
                            (series[yrs[i]] - series[yrs[i - 1]]) / abs(series[yrs[i - 1]]) * 100, 1
                        )
                        for i in range(1, len(yrs))
                        if series[yrs[i - 1]] not in (None, 0)
                    }

            if growth_data:
                st.markdown("#### Year-over-Year Growth (%)")
                fig_growth = go.Figure()
                for company, g_series in growth_data.items():
                    g_years  = sorted(g_series.keys())
                    g_values = [g_series[y] for y in g_years]
                    color    = COMPANY_COLORS.get(company, "#888888")
                    fig_growth.add_trace(go.Scatter(
                        name=company, x=g_years, y=g_values,
                        mode="lines+markers",
                        line=dict(color=color),
                        marker=dict(color=color),
                        hovertemplate="%{y:.1f}%<extra>" + company + "</extra>",
                    ))
                fig_growth.update_layout(
                    xaxis_title="Fiscal Year",
                    yaxis_title="YoY Growth (%)",
                    yaxis=dict(ticksuffix="%"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FAFAFA"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                fig_growth.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(
                    fig_growth,
                    use_container_width=True,
                    config={"displaylogo": False},
                )


with tab_dashboard:
    st.markdown("### Company Dashboard")
    st.caption("Snapshot of key financial metrics for selected companies in a given year.")

    DASHBOARD_METRICS = ["Total Revenue", "Net Income", "Gross Profit", "R&D Expense", "Operating Income"]

    d_col1, d_col2 = st.columns([1, 3])
    with d_col1:
        dash_year = st.selectbox("Year", options=list(reversed(_years)) if _years else [2024], key="dash_year")
    with d_col2:
        dash_companies = st.multiselect(
            "Companies",
            options=_companies,
            default=_companies,
            key="dash_companies",
        )

    if st.button("Load Dashboard", type="primary"):
        if not dash_companies:
            st.warning("Select at least one company.")
        else:
            dash_key = f"dashboard_{dash_year}_{'_'.join(dash_companies)}"
            if dash_key not in st.session_state:
                rag = get_rag()
                dash_results = {}
                prog = st.progress(0, text="Loading…")
                for idx, company in enumerate(dash_companies):
                    dash_results[company] = {}
                    for metric in DASHBOARD_METRICS:
                        series = rag.extract_metric_series(company, [dash_year], metric)
                        dash_results[company][metric] = series.get(dash_year)
                    prog.progress((idx + 1) / len(dash_companies), text=f"Loaded {company}…")
                prog.empty()
                st.session_state[dash_key] = dash_results
            else:
                dash_results = st.session_state[dash_key]

            # Metric cards — one row per metric, one column per company
            for metric in DASHBOARD_METRICS:
                st.markdown(f"**{metric}**")
                cols = st.columns(len(dash_companies))
                for col, company in zip(cols, dash_companies):
                    val = dash_results[company].get(metric)
                    display = f"${val/1000:.2f}B" if val is not None else "N/A"
                    col.metric(label=company, value=display)
                st.markdown("---")

            # Radar chart — compare all companies across all metrics for selected year
            st.markdown("#### Multi-Metric Comparison (Radar)")
            radar_fig = go.Figure()
            for company in dash_companies:
                raw_vals = [dash_results[company].get(m) for m in DASHBOARD_METRICS]
                # Normalise each metric to 0–1 across all companies so the radar is comparable
                norm_vals = []
                for m_idx, metric in enumerate(DASHBOARD_METRICS):
                    all_vals = [dash_results[c].get(metric) for c in dash_companies if dash_results[c].get(metric) is not None]
                    if not all_vals or raw_vals[m_idx] is None:
                        norm_vals.append(0)
                    else:
                        mn, mx = min(all_vals), max(all_vals)
                        norm_vals.append((raw_vals[m_idx] - mn) / (mx - mn) if mx != mn else 1.0)

                radar_fig.add_trace(go.Scatterpolar(
                    r=norm_vals + [norm_vals[0]],
                    theta=DASHBOARD_METRICS + [DASHBOARD_METRICS[0]],
                    fill="toself",
                    name=company,
                    opacity=0.6,
                ))

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(radar_fig, use_container_width=True, config={"displaylogo": False})
            st.caption(f"Values normalised within each metric across selected companies. Fiscal year {dash_year}.")


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
