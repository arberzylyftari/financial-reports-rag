"""RAG query pipeline: retrieve relevant chunks and generate grounded answers."""

from __future__ import annotations

import json
import re
from collections import Counter

from flashrank import Ranker, RerankRequest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from vector_store.embedder import load_vector_store

# Loaded once, shared across all FinancialRAG instances
_reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions based on the provided context from company financial reports.

Rules:
- Only use information present in the provided context.
- Always cite the company name and year for every fact you state.
- If the context contains partial information, provide what is available and clearly note what is missing rather than refusing entirely.
- Only use this exact refusal — "I don't have enough information in the provided documents to answer this question." — when the context contains absolutely no relevant information at all.
- Never fabricate, estimate, or hallucinate financial figures.
- Be concise and precise."""

TOP_K = 8           # final chunks sent to LLM after reranking
TOP_K_FETCH = 24    # candidates fetched before reranking
TOP_K_COMPARE = 12  # per-company chunks in compare mode
ALL_COMPANIES = ["Amazon", "Apple", "Google", "Meta", "Microsoft", "Nvidia", "Tesla"]


def _format_sources(docs) -> list[dict]:
    """Convert retrieved LangChain documents into serializable source dicts."""
    return [
        {
            "company": doc.metadata.get("company"),
            "year": doc.metadata.get("year"),
            "source_file": doc.metadata.get("source_file"),
            "excerpt": doc.page_content[:400],
        }
        for doc in docs
    ]


def _build_context(docs) -> str:
    """Build a context string from retrieved document chunks."""
    parts = []
    for i, doc in enumerate(docs, 1):
        company = doc.metadata.get("company", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        parts.append(f"[Source {i} — {company} {year}]\n{doc.page_content}")
    return "\n\n".join(parts)


class FinancialRAG:
    """Retrieval-Augmented Generation pipeline for financial reports."""

    def __init__(self):
        """Initialize the vector store connection, BM25 index, and LLM."""
        self._vector_store = load_vector_store()
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._bm25 = self._build_bm25()

    def _build_bm25(self) -> BM25Retriever:
        """Build a BM25 index from all chunks stored in ChromaDB, fetched in batches."""
        BATCH = 500
        docs = []
        offset = 0
        while True:
            data = self._vector_store.get(
                include=["documents", "metadatas"], limit=BATCH, offset=offset
            )
            if not data["documents"]:
                break
            docs.extend(
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data["documents"], data["metadatas"])
            )
            if len(data["documents"]) < BATCH:
                break
            offset += BATCH
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = TOP_K_FETCH
        return retriever

    @staticmethod
    def _doc_matches_where(doc: Document, where: dict) -> bool:
        """Return True if doc metadata satisfies a ChromaDB-style where clause."""
        meta = doc.metadata
        if "$and" in where:
            return all(FinancialRAG._doc_matches_where(doc, clause) for clause in where["$and"])
        for key, condition in where.items():
            val = meta.get(key)
            if isinstance(condition, dict):
                if "$in" in condition and val not in condition["$in"]:
                    return False
            elif val != condition:
                return False
        return True

    @staticmethod
    def _rrf_merge(ranked_lists: list[list], k: int = 60) -> list:
        """Merge multiple ranked doc lists with Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, 1):
                key = doc.page_content[:120]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                doc_map[key] = doc
        merged = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in merged]

    def _retrieve(
        self,
        question: str,
        company_filter: str | list[str] | None,
        year_filter: int | list[int] | None,
    ):
        """Retrieve top-K chunks via hybrid BM25 + vector search with RRF, then rerank."""
        def _company_clause(f):
            if isinstance(f, list):
                return {"company": {"$in": f}}
            return {"company": f}

        def _year_clause(f):
            if isinstance(f, list):
                return {"year": {"$in": f}}
            return {"year": f}

        where: dict = {}
        if company_filter and year_filter:
            where = {"$and": [_company_clause(company_filter), _year_clause(year_filter)]}
        elif company_filter:
            where = _company_clause(company_filter)
        elif year_filter:
            where = _year_clause(year_filter)

        # Vector search (with metadata filter)
        retriever_kwargs = {"k": TOP_K_FETCH}
        if where:
            retriever_kwargs["filter"] = where
        vector_docs = self._vector_store.similarity_search(question, **retriever_kwargs)

        # BM25 search (filter applied post-retrieval)
        self._bm25.k = TOP_K_FETCH
        bm25_docs = self._bm25.invoke(question)
        if where:
            bm25_docs = [d for d in bm25_docs if self._doc_matches_where(d, where)]

        merged = self._rrf_merge([vector_docs, bm25_docs])
        return self._rerank(question, merged, TOP_K)

    def _rerank(self, query: str, docs: list, top_n: int) -> list:
        """Re-score docs with a cross-encoder and return the top_n most relevant."""
        if not docs:
            return docs
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = _reranker.rerank(request)
        top_ids = [r["id"] for r in results[:top_n]]
        return [docs[i] for i in top_ids]

    def stats(self) -> dict:
        """Return counts of indexed chunks by company and the years covered."""
        BATCH = 500
        metadatas = []
        offset = 0
        while True:
            data = self._vector_store.get(include=["metadatas"], limit=BATCH, offset=offset)
            if not data["metadatas"]:
                break
            metadatas.extend(data["metadatas"])
            if len(data["metadatas"]) < BATCH:
                break
            offset += BATCH
        companies = dict(Counter(m.get("company", "Unknown") for m in metadatas))
        years = sorted(set(m.get("year") for m in metadatas if m.get("year")))
        return {
            "total_chunks": len(metadatas),
            "companies": companies,
            "years": years,
        }

    def _standalone_question(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question as a self-contained question using chat history."""
        if not history:
            return question
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]
        )
        prompt = (
            f"Given this conversation:\n{history_str}\n\n"
            f"Rewrite the following follow-up question as a fully standalone question "
            f"that can be understood with no prior context. "
            f"If it is already standalone, return it unchanged. "
            f"Return only the rewritten question, nothing else.\n"
            f"Follow-up: {question}"
        )
        return self._llm.invoke([HumanMessage(content=prompt)]).content.strip()

    def _build_messages(self, context: str, prompt: str, history: list[dict] | None = None) -> list:
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in (history or []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=f"Context:\n{context}\n\n{prompt}"))
        return messages

    def query(self, question: str, company_filter: str | None = None, year_filter: int | None = None, chat_history: list[dict] | None = None) -> dict:
        """Answer a question using retrieved context.

        Returns {'answer': str, 'sources': list[dict]}.
        """
        retrieval_question = self._standalone_question(question, chat_history or [])
        docs = self._retrieve(retrieval_question, company_filter, year_filter)
        context = _build_context(docs)
        messages = self._build_messages(context, f"Question: {question}", chat_history)
        response = self._llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": _format_sources(docs),
        }

    def query_stream(self, question: str, company_filter=None, year_filter=None, chat_history: list[dict] | None = None):
        """Like query() but streams the LLM response.

        Returns (stream_generator, sources). Pass the generator to st.write_stream().
        """
        retrieval_question = self._standalone_question(question, chat_history or [])
        docs = self._retrieve(retrieval_question, company_filter, year_filter)
        context = _build_context(docs)
        messages = self._build_messages(context, f"Question: {question}", chat_history)
        return self._llm.stream(messages), _format_sources(docs)

    def extract_metric_series(
        self, company: str, years: list[int], metric: str
    ) -> dict[int, float | None]:
        """Extract a financial metric across multiple years for one company.

        Makes a single LLM call per company and returns a year→value mapping
        (values in millions USD). Returns None for years where data is absent.
        """
        docs = self._vector_store.similarity_search(
            f"{metric} {company} annual total",
            k=TOP_K_COMPARE,
            filter={"company": company},
        )
        if not docs:
            return {y: None for y in years}

        context = _build_context(docs)
        years_str = ", ".join(str(y) for y in years)
        prompt = (
            f"From the context below, extract '{metric}' for {company} "
            f"for each of these fiscal years: {years_str}.\n"
            f"Return a JSON object mapping year (as string) to the value in millions of USD "
            f"(integers only, no symbols). Use null for missing years.\n"
            f"Example: {{\"2022\": 394328, \"2023\": 383285, \"2024\": null}}\n"
            f"Return ONLY the JSON object, nothing else.\n\n"
            f"Context:\n{context}"
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        try:
            match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if match:
                raw = json.loads(match.group())
                return {
                    int(k): (float(v) if v is not None else None)
                    for k, v in raw.items()
                    if int(k) in years
                }
        except Exception:
            pass
        return {y: None for y in years}

    def _compare_docs_and_prompt(self, question: str, companies: list[str]):
        all_docs = []
        for company in companies:
            docs = self._vector_store.similarity_search(
                question, k=TOP_K_COMPARE, filter={"company": company}
            )
            all_docs.extend(docs)
        context = _build_context(all_docs)
        prompt = (
            f"Compare the following companies on this topic: {', '.join(companies)}\n\n"
            f"Question: {question}"
        )
        return all_docs, context, prompt

    def compare(self, question: str, companies: list[str], chat_history: list[dict] | None = None) -> dict:
        """Compare multiple companies by retrieving context for each.

        Returns {'comparison': str, 'sources': list[dict]}.
        """
        all_docs, context, prompt = self._compare_docs_and_prompt(question, companies)
        messages = self._build_messages(context, prompt, chat_history)
        response = self._llm.invoke(messages)
        return {
            "comparison": response.content,
            "sources": _format_sources(all_docs),
        }

    def compare_stream(self, question: str, companies: list[str], chat_history: list[dict] | None = None):
        """Like compare() but streams the LLM response.

        Returns (stream_generator, sources). Pass the generator to st.write_stream().
        """
        all_docs, context, prompt = self._compare_docs_and_prompt(question, companies)
        messages = self._build_messages(context, prompt, chat_history)
        return self._llm.stream(messages), _format_sources(all_docs)
