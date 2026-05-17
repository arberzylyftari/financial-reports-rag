# Financial Reports RAG — Full Presentation Guide

This document covers everything you need to know to confidently present and answer questions about this project.

---

## 1. What Is This Project?

This is a **Financial Reports Q&A system** built with a technique called **RAG (Retrieval-Augmented Generation)**.

In simple words:
- You type a question like *"What was Apple's revenue in 2023?"*
- The system searches through real SEC annual report documents
- It finds the most relevant passages
- It sends those passages to an AI (GPT-4o-mini) which reads them and writes an answer
- Every answer is based **only** on the documents — nothing is made up

**Why is this useful?**
Normal AI like ChatGPT has a knowledge cutoff and can hallucinate (make up) financial numbers. This system is grounded in real documents, so every figure cited can be traced back to a source file.

---

## 2. What Data Does It Use?

The system uses **SEC Form 10-K** filings — these are annual reports that every public company in the United States must file with the SEC (Securities and Exchange Commission). They contain:
- Full financial statements (revenue, net income, cash flow, etc.)
- Business segment breakdowns
- Risk factors
- Management discussion and analysis
- Forward-looking guidance

**7 companies covered:**

| Company   | Ticker | Years Covered |
|-----------|--------|---------------|
| Apple     | AAPL   | 2018 – 2025   |
| Amazon    | AMZN   | 2018 – 2025   |
| Google    | GOOGL  | 2018 – 2025   |
| Meta      | META   | 2018 – 2025   |
| Microsoft | MSFT   | 2018 – 2025   |
| Nvidia    | NVDA   | 2018 – 2026   |
| Tesla     | TSLA   | 2018 – 2025   |

> Note: Nvidia's fiscal year ends in late January. Their FY2026 filing ended January 25, 2026 — that is why the system shows data up to 2026 for Nvidia specifically. All other companies go up to fiscal year 2025.

**Total: 57 HTML documents downloaded from SEC EDGAR (sec.gov)**

---

## 3. What Technologies Were Used and Why

| Technology | What It Does | Why I Chose It |
|-----------|-------------|----------------|
| **Python** | Programming language | Industry standard for AI/ML |
| **OpenAI API** | Powers embeddings and LLM responses | Best-in-class quality, easy API |
| **LangChain** | Framework that connects all the pieces | Handles document splitting, retrieval abstractions |
| **ChromaDB** | Stores document embeddings locally | Free, runs locally, no cloud needed |
| **BM25 (rank-bm25)** | Keyword-based search | Catches exact financial terms that vector search misses |
| **FlashRank** | Cross-encoder reranker | Improves relevance of final documents sent to LLM |
| **Streamlit** | Web interface | Fast to build, Python-native, looks professional |
| **Plotly** | Interactive charts | Beautiful interactive visualizations |
| **RAGAS** | Evaluation framework | Objectively measures how good the system is |
| **BeautifulSoup** | Parses HTML SEC filings | Robust HTML parsing with table extraction |

---

## 4. How I Built It — Step by Step

### Phase 1: Basic Foundation
1. Downloaded 10-K HTML files from SEC EDGAR for Apple and Tesla
2. Built a parser (`ingestion/parser.py`) that reads HTML, removes navigation/scripts, and extracts clean text
3. Built a chunker (`ingestion/chunker.py`) that splits long documents into 800-character pieces with 100-character overlap
4. Built an embedder (`vector_store/embedder.py`) that converts text chunks into numerical vectors (embeddings) and stores them in ChromaDB
5. Built a basic query function that finds similar chunks and passes them to GPT-4o-mini

### Phase 2: Fixing Real Problems
- Added **metadata filtering** so you can filter by company and year
- Fixed **encoding errors** in Tesla files (some characters were corrupted — fixed with cp1252 fallback)
- Fixed **table parsing** — HTML financial tables were losing their labels, so I built a table-to-text converter
- Fixed the **system prompt** so the model gives partial answers instead of refusing when only some data is available

### Phase 3: Improving Retrieval Quality
- Added **BM25 keyword search** alongside vector search
- Merged results with **Reciprocal Rank Fusion (RRF)**
- Added **FlashRank cross-encoder reranking** — this re-scores all 24 candidates and picks the best 8
- Added **HyDE** — generates a hypothetical answer first, uses that for vector search

### Phase 4: Advanced Features
- Added **multi-turn conversation memory** with standalone question rewriting
- Added **suggested follow-up questions** after each answer
- Added **token and cost tracking** in the sidebar
- Added **streaming responses** (text appears word by word)
- Added **Charts tab** with 10 metrics, YoY growth, data table, CSV and HTML downloads
- Added **Dashboard tab** with metric cards and radar chart
- Added **Topic focus filter** in sidebar

### Phase 5: Data Expansion
- Expanded from 2 companies to 7 companies
- Added filings going back to 2018 (7 years per company)
- Re-ran full ingestion pipeline for all 57 documents

---

## 5. How the Pipeline Works — Simple Explanation

Imagine you have a massive library of 57 books (the 10-K filings). A user walks in and asks a question. Here is what happens:

### Step 1 — Parse
Each book is read and cleaned. Financial tables are converted to readable sentences so numbers keep their labels.

### Step 2 — Chunk
Each book is cut into small pieces of ~800 characters. Each piece gets a label: which company, which year, which file.

### Step 3 — Embed
Each piece is converted to a list of 1,536 numbers (a vector/embedding) that represents its meaning. Similar pieces get similar numbers. This is done once and stored permanently in ChromaDB.

### Step 4 — HyDE (at query time)
When a question comes in, the system first generates a short *hypothetical answer* using the LLM. Example:
- Question: *"What was Apple's revenue in 2023?"*
- Hypothetical answer: *"Apple reported total net sales of approximately $383 billion for fiscal year 2023."*

This hypothetical answer is then embedded instead of the raw question — because answers are phrased more like document chunks than questions are.

### Step 5 — Hybrid Search
Two searches run at the same time:
- **Vector search**: finds chunks whose embeddings are mathematically close to the HyDE embedding (semantic similarity)
- **BM25 search**: finds chunks that share exact keywords with the original question

Results from both are merged using **RRF** — a formula that rewards documents that rank highly in both searches.

### Step 6 — Reranking
The top 24 merged results are passed to a cross-encoder model (FlashRank). This model reads each (question, document) pair together and gives a precise relevance score. The top 8 are selected.

### Step 7 — Generate
The 8 most relevant chunks are formatted as context and sent to GPT-4o-mini with a strict system prompt. The model streams its answer word by word, citing company and year for every fact.

---

## 6. What Are Tokens and How Does the Cost Work?

**What is a token?**
Tokens are how AI models measure text. Roughly:
- 1 token ≈ 4 characters of English text
- 1 token ≈ ¾ of a word
- "Hello world" = 2 tokens
- A full 10-K filing paragraph = hundreds of tokens

**Why does it matter?**
Every time you send text to OpenAI's API, you pay per token — both for what you send (input) and what the model writes back (output).

**GPT-4o-mini pricing:**
| Direction | Cost |
|-----------|------|
| Input tokens | $0.15 per 1 million tokens |
| Output tokens | $0.60 per 1 million tokens |

**Example of one query:**
- You ask: "What was Apple's revenue in 2023?" (8 tokens)
- System adds: 8 retrieved document chunks (~1,600 tokens) + system prompt (~100 tokens)
- Total input: ~1,700 tokens → costs about $0.000255
- Model writes answer: ~150 tokens → costs about $0.000090
- **Total per query: ~$0.0003 (less than a tenth of a cent)**

**Why does the app show a session cost?**
Each query makes multiple LLM calls:
1. Standalone question rewriting (if follow-up)
2. HyDE hypothetical document generation
3. Main answer generation
4. Follow-up question suggestions

So a full query might use $0.001–$0.003 total. The sidebar tracker adds these up so you can see the running total for your session.

---

## 7. What Types of Questions Can You Ask?

### Revenue and Financial Performance
- "What was Apple's total revenue in 2024?"
- "How did Nvidia's net income grow from 2022 to 2024?"
- "What was Amazon's gross profit in 2023?"
- "How did Microsoft's operating income change over the last 3 years?"

### Company Comparisons
- "Compare Apple and Microsoft's total revenue in 2023"
- "Who had higher net income in 2022 — Google or Meta?"
- "How does Nvidia's R&D spending compare to Tesla's?"
- "Compare the profitability of all 7 companies in 2023"

### Risk and Business Topics
- "What risks did Meta highlight in their 2024 annual report?"
- "What challenges did Tesla face according to their 2022 filing?"
- "What did Apple say about supply chain risks?"

### Strategy and Guidance
- "How did Amazon explain its profitability improvement?"
- "What growth strategy did Microsoft outline for their cloud business?"
- "What did Nvidia say about their AI chip demand outlook?"

### Segments and Geography
- "What were Microsoft's three main business segments in 2024?"
- "How much of Apple's revenue came from international markets?"
- "What was Amazon Web Services revenue in 2023?"

### Follow-up Questions (Multi-turn)
After getting an answer, you can ask:
- "What about 2022?" (system knows which company you mean)
- "How does that compare to Microsoft?" (picks up company from context)
- "What caused that increase?" (refers to previously discussed figure)

---

## 8. How to Use Each Feature

### Chat Tab — Sidebar Filters
- **Company filter**: Narrow answers to one or more specific companies
- **Year filter**: Narrow answers to specific fiscal years
- **Mode**: "Single Query" for one company, "Compare Companies" for cross-company questions
- **Topic Focus**: Choose a topic to steer the answer — useful when you want to focus on risks, cash flow, guidance, etc.

### Chat Tab — What to Expect
- Answer streams in word by word (like ChatGPT)
- Under the answer: a collapsible "Sources" section shows exactly which documents were used and what text was retrieved
- Below sources: 3 follow-up question buttons — click any to ask it immediately
- Sidebar shows live token count and estimated cost for the session

### Charts Tab — Step by Step
1. Select a **Metric** from the dropdown (e.g., Total Revenue)
2. Select **Chart type** (Bar, Line, or Area)
3. Select **Companies** to compare
4. Adjust the **Year range** slider
5. Click **Generate Chart**
6. The chart appears with each company in a distinct color
7. Below the chart: a YoY growth % chart appears automatically
8. Below that: a formatted data table and download buttons (HTML chart, CSV data)

### Dashboard Tab — Step by Step
1. Select a **Year**
2. Select **Companies**
3. Click **Load Dashboard**
4. Metric cards appear showing Revenue, Net Income, Gross Profit, R&D, Operating Income side by side per company
5. A radar chart at the bottom compares all companies across all metrics at once (normalised so the shape shows relative strengths)

---

## 9. Problems I Encountered and How I Solved Them

| Problem | What Happened | How I Fixed It |
|---------|--------------|----------------|
| Tesla files had corrupted characters | Tesla's HTM files used Windows encoding (cp1252) not UTF-8 | Added encoding fallback: try UTF-8, fall back to cp1252 |
| Bullet characters (•, ●, ◉) appeared in answers | Non-ASCII characters weren't being cleaned | Extended `_clean_text()` with a full replacement map |
| Financial tables lost their labels | `get_text()` on HTML tables produces raw numbers with no context | Built `_table_to_text()` that converts each row to "Label: year1 value1, year2 value2" |
| "I don't have enough information" even when sources were retrieved | System prompt was too strict — refused partial answers | Softened system prompt to provide partial answers when context has relevant data |
| Compare mode only showed Apple data | Comparison defaulted to Apple/Tesla from old code | Added `_extract_companies_from_query()` to detect which companies are mentioned in the question |
| New messages appeared below chat input | Streamlit renders in order — new message was at bottom | Saved message to history then called `st.rerun()` so it renders in the history loop above the input |
| SQLite "too many variables" error | Fetching all chunks at once from ChromaDB exceeded SQLite's limit of 999 variables | Changed `_build_bm25()` and `stats()` to fetch in pages of 500 |
| Charts turned all black | No explicit colors set — Plotly inherited dark theme black | Added a `COMPANY_COLORS` dict with distinct colors per company, applied to every trace |
| Download button reset the page | `st.download_button` triggers a full Streamlit rerun, losing the chart | Stored render parameters in `st.session_state` so the chart re-renders after any rerun |
| Downloaded HTML chart had invisible white text | Figure had white font color for dark theme — exported as-is | Created a light-themed copy of the figure before exporting to HTML |
| Compare mode showed no data for Microsoft/Nvidia | Vector store only had Apple and Tesla from initial ingestion | Re-ran full ingestion with all 7 companies |

---

## 10. What Is Left Unsolved / Limitations

- **Apple 2024 missing** — The 2024 Apple filing didn't get indexed correctly (shows N/A in charts). Would need to verify the file and re-ingest.
- **LLM extraction accuracy** — The chart values are extracted by the LLM from document text, not from a structured database. They are usually correct but should be verified against the source before use in any real analysis.
- **No real-time data** — The system only knows what's in the indexed filings. It has no access to live stock prices, news, or recent quarterly results.
- **Ingestion takes 15–25 minutes** — Embedding 57 documents is slow due to OpenAI rate limits. A production system would use async batching or a faster embedding model.
- **RAGAS evaluation not run yet** — The evaluation framework is built but we haven't run it to get benchmark scores for faithfulness and relevancy.
- **Not deployed yet** — The app runs locally. Deployment to Streamlit Cloud would make it publicly accessible.

---

## 11. Likely Questions From Your Teacher — With Answers

**Q: What is RAG?**
RAG stands for Retrieval-Augmented Generation. Instead of relying on what the AI model memorised during training, you first retrieve relevant documents from a database and then pass those documents to the model as context. The model answers based on the documents, not its training data. This prevents hallucination and keeps answers up to date.

**Q: Why not just use ChatGPT directly?**
ChatGPT doesn't have access to specific company filings from 2018–2024 in a structured way. It might hallucinate numbers or give outdated information. Our system grounds every answer in the actual filed documents — you can see exactly which source was used and read the exact text that backed the answer.

**Q: What is an embedding?**
An embedding converts text into a list of numbers (a vector) that represents the meaning of that text. Similar texts get similar vectors. This lets us do mathematical similarity search — instead of matching keywords, we match meaning. For example, "revenue" and "total sales" would have similar embeddings even though they don't share a word.

**Q: What is vector search?**
Vector search finds documents whose embedding vectors are mathematically close (in cosine similarity) to the query embedding. It's fast because ChromaDB pre-indexes all the vectors and uses approximate nearest-neighbor algorithms to find matches without comparing every single document.

**Q: Why do you use BM25 alongside vector search?**
Vector search is great at semantic similarity but sometimes misses exact financial terms. If someone asks about "R&D expense 2023," vector search might return chunks about "research investment" which is semantically similar but doesn't have the exact number. BM25 (which is based on keyword frequency) would find the exact term. Using both and merging results with RRF gives the best of both worlds.

**Q: What is HyDE and why does it help?**
HyDE (Hypothetical Document Embeddings) works on the insight that questions and answers live in different embedding spaces. A question like "What was Apple's revenue?" is phrased very differently from how the answer appears in a document ("Apple's total net sales were $391 billion"). By generating a hypothetical answer first and embedding that, we put the search query in the same space as the document chunks, finding better matches.

**Q: What is cross-encoder reranking?**
When we retrieve 24 candidate documents, we need to pick the best 8 to send to the LLM. A cross-encoder model reads each (question + document) pair together and scores how relevant the document is to the question. This is more accurate than the original vector similarity score because the model sees both texts together and can catch subtle mismatches. The cost is that it's slower — but we only run it on 24 candidates, not the full database.

**Q: How do you prevent hallucination?**
Three layers of protection:
1. The system prompt explicitly forbids fabricating or estimating figures
2. Every retrieved chunk is labeled with company and year, so the LLM always knows what it's reading
3. Sources are shown to the user so they can verify any claim against the original document text

**Q: What are the main limitations of your system?**
- Accuracy depends on retrieval quality — if the right chunk isn't retrieved, the answer will be incomplete
- LLM-extracted chart values can occasionally be wrong — should be verified against source documents
- The system only knows what's in the indexed filings — no live data, no recent news
- It's computationally expensive: each query makes 3–4 LLM calls (HyDE, rewriting, answering, follow-ups)

**Q: How much does it cost to run?**
Each full query costs roughly $0.001–$0.003 (fractions of a cent). For a demo session with 20 questions, total cost would be under $0.10. The sidebar shows the real-time session cost so you can track it.

**Q: What would you improve if you had more time?**
- Deploy to Streamlit Cloud for a public URL
- Run RAGAS evaluation and add benchmark scores to the README
- Add async/parallel retrieval to reduce latency
- Add a proper database for structured financial data to avoid relying on LLM extraction for charts
- Add PDF support alongside HTML filings

---

## 12. Key Terms to Know

| Term | Plain English Explanation |
|------|--------------------------|
| **RAG** | Retrieve documents first, then use AI to answer based on them |
| **Embedding** | Converting text to numbers that represent its meaning |
| **Vector store** | A database that stores embeddings and supports fast similarity search |
| **BM25** | Keyword search that scores documents by how often query words appear |
| **RRF** | A formula that merges rankings from multiple search systems |
| **HyDE** | Generate a fake answer first, use that for search |
| **Cross-encoder** | A model that reads (question + document) together to score relevance |
| **Reranking** | Re-scoring retrieved documents to pick the best ones |
| **Chunk** | A small piece of a document (800 characters in this project) |
| **Token** | The unit AI models use to measure text (~4 characters each) |
| **10-K** | Annual report that US public companies must file with the SEC |
| **ChromaDB** | The local vector database used in this project |
| **Streamlit** | The Python library used to build the web interface |
| **RAGAS** | Framework for evaluating RAG systems with measurable metrics |
| **Faithfulness** | Does the answer only say things that are in the retrieved documents? |
| **Context Recall** | Did retrieval find all the information needed to answer correctly? |

---

## 13. How to Give a Live Demo

**Recommended demo flow (5–7 minutes):**

1. **Open the Chat tab**
   - Show the sidebar: company filter, year filter, mode, topic focus
   - Ask: *"What was Apple's total revenue in 2024?"*
   - Show the streaming response, expand Sources to show where it came from
   - Click one of the follow-up question buttons

2. **Show the topic filter working**
   - Set Topic Focus to "Risk Factors"
   - Ask: *"Tell me about Tesla"*
   - Show how the answer focuses entirely on risks, not general overview

3. **Show comparison mode**
   - Ask: *"Compare Microsoft and Google's cloud revenue in 2023"*
   - Show it auto-detects comparison intent and switches mode
   - Show sources from both companies

4. **Open the Charts tab**
   - Select Total Revenue, Bar chart, Apple + Microsoft + Nvidia, 2020–2024
   - Click Generate Chart
   - Show the bar chart with company colors
   - Scroll down to show YoY growth chart, data table, download buttons
   - Download the CSV and show its structure

5. **Open the Dashboard tab**
   - Select 2023, select all companies
   - Click Load Dashboard
   - Show metric cards side by side
   - Show the radar chart at the bottom

6. **Point to the sidebar cost tracker**
   - Explain tokens and how the cost adds up across the session

---

*This document covers everything needed for the presentation. Study the "Likely Questions" section carefully — those are the most common questions you will get.*
