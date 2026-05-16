"""RAG query pipeline: retrieve relevant chunks and generate grounded answers."""

from __future__ import annotations

from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from vector_store.embedder import load_vector_store

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions based on the provided context from company financial reports.

Rules:
- Only use information present in the provided context.
- Always cite the company name and year for every fact you state.
- If the context contains partial information, provide what is available and clearly note what is missing rather than refusing entirely.
- Only use this exact refusal — "I don't have enough information in the provided documents to answer this question." — when the context contains absolutely no relevant information at all.
- Never fabricate, estimate, or hallucinate financial figures.
- Be concise and precise."""

TOP_K = 8
TOP_K_COMPARE = 12
ALL_COMPANIES = ["Apple", "Tesla"]


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
        """Initialize the vector store connection and LLM."""
        self._vector_store = load_vector_store()
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _retrieve(
        self,
        question: str,
        company_filter: str | list[str] | None,
        year_filter: int | list[int] | None,
    ):
        """Retrieve top-K chunks, optionally filtered by company and/or year."""
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

        retriever_kwargs = {"k": TOP_K}
        if where:
            retriever_kwargs["filter"] = where

        return self._vector_store.similarity_search(question, **retriever_kwargs)

    def stats(self) -> dict:
        """Return counts of indexed chunks by company and the years covered."""
        data = self._vector_store.get(include=["metadatas"])
        metadatas = data["metadatas"]
        companies = dict(Counter(m.get("company", "Unknown") for m in metadatas))
        years = sorted(set(m.get("year") for m in metadatas if m.get("year")))
        return {
            "total_chunks": len(metadatas),
            "companies": companies,
            "years": years,
        }

    def query(self, question: str, company_filter: str | None = None, year_filter: int | None = None) -> dict:
        """Answer a question using retrieved context.

        Returns {'answer': str, 'sources': list[dict]}.
        """
        docs = self._retrieve(question, company_filter, year_filter)
        context = _build_context(docs)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        response = self._llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": _format_sources(docs),
        }

    def compare(self, question: str, companies: list[str]) -> dict:
        """Compare multiple companies by retrieving context for each.

        Returns {'comparison': str, 'sources': list[dict]}.
        """
        all_docs = []
        for company in companies:
            docs = self._vector_store.similarity_search(
                question, k=TOP_K_COMPARE, filter={"company": company}
            )
            all_docs.extend(docs)

        context = _build_context(all_docs)
        comparison_prompt = (
            f"Compare the following companies on this topic: {', '.join(companies)}\n\n"
            f"Question: {question}"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\n{comparison_prompt}"),
        ]

        response = self._llm.invoke(messages)
        return {
            "comparison": response.content,
            "sources": _format_sources(all_docs),
        }
