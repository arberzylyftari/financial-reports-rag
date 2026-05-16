# Financial Reports RAG

A production-quality **Retrieval-Augmented Generation** system that lets you chat with SEC 10-K annual filings from seven major tech companies. Every answer is grounded strictly in the source documents — no hallucinated figures, no fabricated citations.

---

## Overview

This project answers complex financial questions like:

- *"What was Apple's total revenue in 2024?"*
- *"How did Nvidia's net income grow from 2022 to 2025?"*
- *"Compare Microsoft and Google's cloud revenue over the last three years."*
- *"What risks did Meta highlight in their 2024 annual report?"*

It achieves this through a multi-stage pipeline: parse → chunk → embed → hybrid retrieve → cross-encoder rerank → generate. The result is an interactive chat interface backed by ~57 SEC filings covering 2018–2025.

---

## Companies & Data Coverage

| Company   | Filings Covered |
|-----------|----------------|
| Apple     | 2018 – 2024    |
| Amazon    | 2018 – 2024    |
| Google    | 2018 – 2024    |
| Meta      | 2018 – 2024    |
| Microsoft | 2018 – 2024    |
| Nvidia    | 2018 – 2024    |
| Tesla     | 2018 – 2024    |

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
┌──────────────────────────────┐
│       Hybrid Retrieval       │
│                              │
│  Vector search (ChromaDB)    │
│       +                      │  ──► Reciprocal Rank Fusion
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
│         GPT-4o-mini         │  Streaming response, source citations
│   (grounded generation)     │
└─────────────────────────────┘
```

---

## Pipeline Details

### 1. Parsing (`ingestion/parser.py`)

HTML 10-K filings are parsed with BeautifulSoup. The key innovation is table handling: rather than dumping raw HTML table text (which loses the meaning of every cell), each table is converted into labeled sentences:

```
Revenue: 2023 $394,328M, 2024 $391,035M
Net Income: 2023 $96,995M, 2024 $93,736M
```

This ensures financial figures remain attached to their row labels and column year headers — critical for accurate retrieval. The parser also handles encoding quirks common in SEC filings (UTF-8 with cp1252 fallback) and normalises non-ASCII characters.

### 2. Chunking (`ingestion/chunker.py`)

Documents are split using LangChain's `RecursiveCharacterTextSplitter` with:
- **Chunk size**: 800 characters
- **Overlap**: 100 characters

Every chunk inherits the parent document's metadata (`company`, `year`, `source_file`), enabling precise metadata filtering at query time.

### 3. Embedding (`vector_store/embedder.py`)

Chunks are embedded with OpenAI's `text-embedding-3-small` model and persisted in a local ChromaDB vector store. Embedding is batched (50 chunks per call, 2s pause between batches) to stay within API rate limits.

### 4. Hybrid Retrieval (`retrieval/rag_pipeline.py`)

At query time, two search strategies run in parallel:

**Vector search** — ChromaDB cosine similarity over `text-embedding-3-small` embeddings. Excels at semantic understanding ("profitability improvement" → retrieves chunks about operating leverage, margin expansion).

**BM25 keyword search** — Classic TF-IDF-based ranking over all indexed chunks. Excels at exact-match queries ("R&D expense 2023", specific dollar figures, technical terms).

Results from both searches are merged with **Reciprocal Rank Fusion (RRF)**:

```
score(doc) = Σ  1 / (k + rank_i)
```

where `k=60` dampens the influence of high-ranked outliers. This consistently outperforms either strategy alone, especially for queries that blend semantic intent with specific financial terminology.

Metadata filters (`company`, `year`) are applied natively in ChromaDB for vector results, and via post-retrieval filtering for BM25 results — ensuring both legs respect user-selected filters.

### 5. Cross-Encoder Reranking

The merged candidate set (up to 24 documents) is reranked by a **FlashRank cross-encoder** (`ms-marco-MiniLM-L-12-v2`). Unlike bi-encoder similarity, the cross-encoder scores each (query, document) pair jointly, capturing subtle relevance signals. The top 8 documents are passed to the LLM.

### 6. Generation

Retrieved chunks are formatted as a numbered context block with source labels and passed to `gpt-4o-mini` alongside a strict system prompt that requires:
- Citation of company and year for every fact
- No fabrication or estimation of figures
- Partial-information answers over refusals when the context contains relevant data

Responses are streamed token-by-token to the Streamlit interface.

---

## Features

### Chat Interface
- Natural language Q&A with streaming responses
- Persistent chat history (saved to disk across sessions)
- One-click conversation export to Markdown
- Auto-detection of comparison queries (switches to per-company retrieval mode automatically)
- Sidebar filters for company and year

### Compare Mode
- Retrieves a balanced set of chunks per company
- Ensures all requested companies are represented in the context
- Auto-detects comparison intent from query text ("compare", "vs", "versus", multiple company names)

### Financial Charts
- Interactive Plotly charts: bar and line views
- Metrics: Total Revenue, Net Income, R&D Expense, Operating Income, Gross Profit
- LLM extracts values as structured JSON from filing context (values in millions USD, converted to billions for display)
- Year-range slider for temporal filtering
- Results cached per session to avoid redundant LLM calls

---

## Project Structure

```
financial-reports-rag/
├── app/
│   └── streamlit_app.py        # Streamlit frontend (Chat, Charts, About tabs)
├── ingestion/
│   ├── parser.py               # HTML → clean text with table-to-text conversion
│   └── chunker.py              # LangChain text splitter with metadata propagation
├── vector_store/
│   └── embedder.py             # OpenAI embeddings + ChromaDB persistence
├── retrieval/
│   └── rag_pipeline.py         # Hybrid BM25+vector retrieval, RRF, reranking, LLM
├── evaluation/
│   ├── eval_dataset.py         # Hand-crafted Q&A pairs with ground-truth answers
│   └── ragas_eval.py           # RAGAS evaluation script (faithfulness, relevancy, recall)
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
...
```

### Run Ingestion

```bash
python scripts/ingest.py
```

This parses all HTML files, chunks them, embeds them, and persists the vector store. Expect ~10–20 minutes for a full run of 50+ documents (depending on OpenAI rate limits).

### Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Evaluation

The project includes a RAGAS-based evaluation suite to measure retrieval and generation quality:

```bash
python evaluation/ragas_eval.py
```

**Metrics evaluated:**

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Fraction of answer claims that are supported by the retrieved context |
| **Answer Relevancy** | How well the answer addresses the question asked |
| **Context Recall** | Fraction of the ground-truth answer that can be attributed to retrieved chunks |

The evaluation dataset (`evaluation/eval_dataset.py`) contains hand-crafted Q&A pairs covering revenue figures, year-over-year growth, segment breakdowns, and risk factors across all seven companies.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover year extraction from SEC filenames, text cleaning, chunking, and metadata propagation.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local persistent) |
| Keyword Search | BM25 via rank-bm25 |
| Result Merging | Reciprocal Rank Fusion |
| Reranking | FlashRank (ms-marco-MiniLM-L-12-v2) |
| Orchestration | LangChain |
| HTML Parsing | BeautifulSoup4 + lxml |
| Charts | Plotly |
| Evaluation | RAGAS |

---

## Design Decisions

**Why hybrid retrieval?**
Pure vector search is strong on semantic queries but weak on exact-match financial terms (specific dollar amounts, metric names). BM25 is the opposite. RRF fusion reliably outperforms either approach alone without requiring any additional model training.

**Why cross-encoder reranking?**
Bi-encoder similarity (used in vector search) compresses query and document into separate embeddings — fast, but it loses inter-token interactions. A cross-encoder sees the full (query, document) pair and produces much more accurate relevance scores. Reranking a pool of 24 candidates to 8 adds minimal latency while meaningfully improving the quality of context passed to the LLM.

**Why table-to-text conversion?**
Plain `get_text()` on HTML tables produces streams of numbers with no labels — the embedder has no way to know that `394328` refers to Apple's 2022 revenue. Converting each row to `"Revenue: 2022 $394,328M, 2023 $383,285M"` makes the relationship explicit and dramatically improves retrieval precision for numerical queries.

**Why `gpt-4o-mini`?**
For strictly grounded generation with a well-formed context window, `gpt-4o-mini` performs on par with larger models at a fraction of the cost. The system prompt and retrieval quality do more work than model size.

---

## License

MIT
