# Financial Reports RAG

A production-quality **Retrieval-Augmented Generation** system that lets you chat with SEC 10-K annual filings from seven major tech companies. Every answer is grounded strictly in the source documents — no hallucinated figures, no fabricated citations.

---

## Overview

This project answers complex financial questions like:

- *"What was Apple's total revenue in 2024?"*
- *"How did Nvidia's net income grow from 2022 to 2025?"*
- *"Compare Microsoft and Google's cloud revenue over the last three years."*
- *"What risks did Meta highlight in their 2024 annual report?"*
- *"How did Amazon explain its profitability improvement?"*

It achieves this through a multi-stage pipeline: parse → chunk → embed → HyDE → hybrid retrieve → cross-encoder rerank → generate. The result is a full-featured interactive application backed by 57 SEC filings covering 2018–2025, with a chat interface, financial charting, a company dashboard, and a RAGAS evaluation suite.

---

## Companies & Data Coverage

| Company   | Filings Covered | Ticker |
|-----------|----------------|--------|
| Apple     | 2018 – 2025    | AAPL   |
| Amazon    | 2018 – 2025    | AMZN   |
| Google    | 2018 – 2025    | GOOGL  |
| Meta      | 2018 – 2025    | META   |
| Microsoft | 2018 – 2025    | MSFT   |
| Nvidia    | 2018 – 2026    | NVDA   |
| Tesla     | 2018 – 2025    | TSLA   |

> Nvidia's fiscal year ends in late January, so their FY2026 filing (ended January 25, 2026) is the most recent available.

All filings are SEC Form 10-K (annual reports), sourced directly from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar).

---

## Architecture

```
SEC EDGAR HTML
      │
      ▼
┌─────────────┐
│   Parser    │  BeautifulSoup — extracts text + converts HTML tables
│             │  to labeled sentences ("Revenue: 2023 $394B, 2024 $391B")
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Chunker   │  RecursiveCharacterTextSplitter — 800-char chunks, 100-char overlap
│             │  Each chunk inherits company / year / source_file metadata
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Embedder  │  OpenAI text-embedding-3-small — batched at 50 chunks/call
│             │  Persisted in local ChromaDB collection
└──────┬──────┘
       │  (at query time)
       ▼
┌─────────────────────────────┐
│    HyDE Query Expansion     │  GPT-4o-mini generates a short hypothetical
│                             │  answer — that text is embedded instead of
│                             │  the raw question for vector search
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Hybrid Retrieval       │
│                              │
│  Vector search (ChromaDB)    │  ──► Reciprocal Rank Fusion (RRF)
│       +                      │
│  Keyword search (BM25)       │
└──────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Cross-Encoder Reranker    │  FlashRank ms-marco-MiniLM-L-12-v2
│   (top 8 of 24 candidates)  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         GPT-4o-mini         │  Streaming response with source citations
│   (grounded generation)     │  Multi-turn conversation memory
└─────────────────────────────┘
```

---

## Pipeline Details

### 1. Parsing (`ingestion/parser.py`)

HTML 10-K filings are parsed with BeautifulSoup. The key innovation is table handling: rather than dumping raw HTML table text (which destroys the meaning of every cell), each table is converted into labeled sentences:

```
Revenue: 2023 $394,328M, 2024 $391,035M
Net Income: 2023 $96,995M, 2024 $93,736M
```

This keeps financial figures attached to their row labels and column year headers — critical for accurate retrieval of specific numbers. The parser also handles encoding quirks common in SEC filings (UTF-8 with cp1252 fallback) and normalises all non-ASCII characters.

### 2. Chunking (`ingestion/chunker.py`)

Documents are split using LangChain's `RecursiveCharacterTextSplitter` with:
- **Chunk size**: 800 characters
- **Overlap**: 100 characters

Every chunk inherits the parent document's metadata (`company`, `year`, `source_file`), enabling precise metadata filtering at query time.

### 3. Embedding (`vector_store/embedder.py`)

Chunks are embedded with OpenAI's `text-embedding-3-small` model and persisted in a local ChromaDB vector store. Embedding is batched (50 chunks per call, 2s pause between batches) to stay within API rate limits. The vector store is fetched in pages of 500 to avoid SQLite variable limits on large collections.

### 4. HyDE — Hypothetical Document Embeddings (`retrieval/rag_pipeline.py`)

Before retrieval, the pipeline uses GPT-4o-mini to generate a short, plausible financial report passage that *would* answer the user's question. That hypothetical passage is then embedded and used as the vector search query instead of the raw question.

This works because hypothetical answers live in the same embedding space as real document chunks, whereas questions are phrased differently from answers. The generated text is never shown to the user — it is used purely to improve retrieval.

### 5. Hybrid Retrieval (`retrieval/rag_pipeline.py`)

Two search strategies run in parallel:

**Vector search** — ChromaDB cosine similarity over `text-embedding-3-small` embeddings of the HyDE-generated passage. Excels at semantic understanding ("profitability improvement" → retrieves chunks about operating leverage, margin expansion).

**BM25 keyword search** — Classic TF-IDF-based ranking over all indexed chunks, built in-memory at startup. Runs against the original question (not HyDE) to preserve exact keyword matching for specific financial terms, figures, and metric names.

Results from both searches are merged with **Reciprocal Rank Fusion (RRF)**:

```
score(doc) = Σ  1 / (k + rank_i)
```

where `k=60` dampens the influence of high-ranked outliers. RRF consistently outperforms either strategy alone, especially for queries that blend semantic intent with exact financial terminology.

Metadata filters (`company`, `year`) are applied natively in ChromaDB for vector results, and via post-retrieval filtering for BM25 results — ensuring both legs respect user-selected filters.

### 6. Cross-Encoder Reranking

The merged candidate set (up to 24 documents) is reranked by a **FlashRank cross-encoder** (`ms-marco-MiniLM-L-12-v2`). Unlike bi-encoder similarity (which compresses query and document separately), a cross-encoder sees the full (query, document) pair jointly, capturing subtle relevance signals that bi-encoders miss. The top 8 documents are passed to the LLM.

### 7. Multi-Turn Generation

Retrieved chunks are formatted as a numbered context block with source labels. Before retrieval, follow-up questions are rewritten into standalone questions using the conversation history (e.g. "what about 2023?" → "What was Apple's total revenue in 2023?"). The full conversation history (last 3 turns) is then included in the LLM prompt so the model can reference previous answers.

GPT-4o-mini streams its response token-by-token with a strict system prompt that requires:
- Citation of company and year for every fact stated
- No fabrication or estimation of figures
- Partial-information answers over outright refusals when the context contains relevant data

After each answer, a separate LLM call generates 3 contextually relevant follow-up questions displayed as clickable buttons.

---

## Application Features

### Chat Tab
- Natural language Q&A with real-time streaming responses
- **Multi-turn conversation memory** — follow-up questions like "what about 2023?" work correctly because prior exchanges are included in context
- **Suggested follow-up questions** — 3 clickable related questions appear after each answer
- **Topic focus filter** — steer answers toward specific areas: Revenue & Profit, Cash Flow, R&D & Innovation, Risk Factors, Guidance & Outlook, Segments & Geography
- Company and year sidebar filters with multi-select support
- Auto-detection of comparison queries — switches to per-company retrieval mode automatically
- **Token & cost tracker** — session token count and estimated USD cost shown in the sidebar
- Persistent chat history saved to disk across sessions
- One-click conversation export to Markdown

### Charts Tab
- **10 extractable metrics**: Total Revenue, Net Income, Gross Profit, Operating Income, R&D Expense, Free Cash Flow, Total Assets, Cash and Equivalents, Gross Margin %, Operating Margin %
- **3 chart types**: Bar, Line, Area
- **Year-over-year growth overlay** — automatic % growth rate chart appears below every main chart
- **Raw data table** — formatted values shown below the chart (billions for absolute metrics, % for margins)
- **Download chart** as PNG (via toolbar camera button) or interactive HTML
- **Download data** as CSV with metric metadata, unit labels, and formatted numbers
- Year-range slider for temporal filtering
- Results cached per session to avoid redundant LLM calls

### Dashboard Tab
- Select a fiscal year and any combination of companies
- **Metric cards** — side-by-side snapshot of Revenue, Net Income, Gross Profit, R&D Expense, and Operating Income for all selected companies
- **Radar/spider chart** — multi-metric comparison across all selected companies, normalised within each metric so the shape reveals relative strengths

### About Tab
- Full pipeline explanation with step-by-step table
- Live indexed document stats (chunks per company, years covered)
- Tech stack overview

---

## Project Structure

```
financial-reports-rag/
├── app/
│   └── streamlit_app.py        # Streamlit frontend (Chat, Charts, Dashboard, About)
├── ingestion/
│   ├── parser.py               # HTML → clean text with table-to-text conversion
│   └── chunker.py              # LangChain text splitter with metadata propagation
├── vector_store/
│   └── embedder.py             # OpenAI embeddings + ChromaDB persistence
├── retrieval/
│   └── rag_pipeline.py         # HyDE, hybrid BM25+vector, RRF, reranking, LLM
├── evaluation/
│   ├── eval_dataset.py         # Hand-crafted Q&A pairs with ground-truth answers
│   └── ragas_eval.py           # RAGAS evaluation (faithfulness, relevancy, recall)
├── scripts/
│   └── ingest.py               # End-to-end ingestion runner
├── tests/
│   └── test_pipeline.py        # Unit tests for parser and chunker
├── data/
│   └── raw/                    # HTML 10-K filings (gitignored)
│       ├── apple/
│       ├── amazon/
│       ├── google/
│       ├── meta/
│       ├── microsoft/
│       ├── nvidia/
│       └── tesla/
├── chroma_db/                  # Persistent vector store (gitignored)
├── requirements.txt
└── .env                        # OPENAI_API_KEY (gitignored)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/arberzylyftari/financial-reports-rag.git
cd financial-reports-rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### Download the Data

Download 10-K filings from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar):

1. Search for a company (e.g. "Apple Inc") and set filing type to `10-K`
2. Click a filing → click the `.htm` document link
3. Save the HTML file into the corresponding folder under `data/raw/`

```
data/raw/apple/aapl-20240928.htm
data/raw/nvidia/nvda-20250126.htm
data/raw/microsoft/msft-20240630.htm
...
```

### Run Ingestion

```bash
python scripts/ingest.py
```

This parses all HTML files, chunks them, embeds them in batches, and persists the ChromaDB vector store. Expect 15–25 minutes for a full run of 57 documents depending on OpenAI rate limits. Progress is printed per batch.

### Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Evaluation

The project includes a RAGAS-based evaluation suite to objectively measure retrieval and generation quality:

```bash
python evaluation/ragas_eval.py
```

**Metrics evaluated:**

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Fraction of answer claims that are supported by the retrieved context — catches hallucinations |
| **Answer Relevancy** | How well the answer addresses the question that was actually asked |
| **Context Recall** | Fraction of the ground-truth answer that can be attributed to retrieved chunks — measures retrieval completeness |

The evaluation dataset (`evaluation/eval_dataset.py`) contains hand-crafted Q&A pairs with ground-truth answers, covering revenue figures, year-over-year growth, segment breakdowns, and risk factors across all seven companies. Results are saved to `evaluation/results.json`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover year extraction from SEC filenames, text cleaning, chunking behaviour, and metadata propagation through the pipeline.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local persistent) |
| Keyword Search | BM25 via rank-bm25 |
| Query Expansion | HyDE (Hypothetical Document Embeddings) |
| Result Merging | Reciprocal Rank Fusion (RRF) |
| Reranking | FlashRank (ms-marco-MiniLM-L-12-v2) |
| Orchestration | LangChain |
| HTML Parsing | BeautifulSoup4 + lxml |
| Charts | Plotly |
| Evaluation | RAGAS |

---

## Design Decisions

**Why HyDE?**
Users phrase questions differently from how answers appear in documents. "What was Apple's revenue?" is structurally different from "Apple's total net sales were $391 billion." Generating a hypothetical answer first and embedding that brings the search query into the same embedding space as real document chunks, improving retrieval precision — especially for vague or high-level questions.

**Why hybrid retrieval?**
Pure vector search is strong on semantic queries but misses exact financial terms (specific dollar amounts, metric names, fiscal year identifiers). BM25 is the opposite — great at exact matches, blind to synonyms. RRF fusion gets both without any additional training.

**Why cross-encoder reranking?**
Bi-encoder similarity compresses query and document into separate vectors — fast, but it loses all inter-token interactions between them. A cross-encoder sees the full (query, document) pair jointly and produces much more accurate relevance scores. Reranking 24 candidates down to 8 adds only ~100ms while meaningfully improving the quality of context passed to the LLM.

**Why table-to-text conversion?**
Plain `get_text()` on HTML financial tables produces streams of numbers with no labels — the embedder cannot know that `394328` refers to Apple's 2022 revenue in millions. Converting each table row to `"Total Revenue: 2022 $394,328M, 2023 $383,285M"` makes the relationship explicit and dramatically improves retrieval accuracy for numerical queries.

**Why multi-turn memory with standalone question rewriting?**
Simply appending history to the LLM context is not enough — the retrieval step still sees the ambiguous follow-up question ("what about 2023?") and fetches irrelevant chunks. Rewriting the follow-up into a fully standalone question before retrieval ensures the vector search and BM25 index receive a complete, unambiguous query.

**Why `gpt-4o-mini`?**
For strictly grounded generation over a well-formed context window, `gpt-4o-mini` performs on par with larger models at a fraction of the cost. The system prompt constraints and retrieval quality do more work than raw model capability.

---

## License

MIT
