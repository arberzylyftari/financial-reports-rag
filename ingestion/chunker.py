"""Split parsed documents into chunks with metadata using LangChain."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Matches a clean parsed financial-statement line:
#   "Operating income: 2024 123,216, 2023 114,301, 2022 119,437"
_FACT_LINE = re.compile(r"^(?P<label>[^:]{2,80}): (?P<body>(?:19|20)\d{2} .+)$")
_PAIR_SPLIT = re.compile(r",\s+(?=(?:19|20)\d{2}\s)")

# Canonical income-statement / balance-sheet / cash-flow line items worth
# indexing as standalone facts. Targeted on purpose: emitting a fact for
# every numeric row (percentages, per-share, share counts, footnote
# sublines) roughly doubled the corpus and destabilized ingestion.
_KEY_METRICS = (
    "total revenue", "total revenues", "total net sales", "net sales",
    "net revenue", "revenue",
    "cost of sales", "cost of revenue", "cost of revenues",
    "gross margin", "gross profit",
    "research and development",
    "selling, general and administrative", "sales and marketing",
    "general and administrative",
    "operating income", "income from operations", "operating expenses",
    "operating loss",
    "net income", "net loss",
    "income before provision for income taxes", "income before income taxes",
    "provision for income taxes",
    "total assets", "total liabilities", "total current assets",
    "total current liabilities", "stockholders' equity", "long-term debt",
    "cash and cash equivalents", "inventories", "goodwill",
    "net cash provided by operating activities",
    "net cash used in operating activities",
    "free cash flow", "capital expenditures",
    "purchases of property and equipment",
)
# Labels containing these are ratios/noise, not headline figures.
_NOISE_MARKERS = (
    "percentage", "% of", "as a percent", "per share",
    "weighted-average", "shares used", "number of shares", "credit, net",
)


def _is_key_metric(label_lc: str) -> bool:
    """True only for headline financial-statement line items."""
    if any(n in label_lc for n in _NOISE_MARKERS):
        return False
    return any(k in label_lc for k in _KEY_METRICS)


def _atomic_facts(text: str, metadata: dict) -> list[Document]:
    """Emit one short, self-contained Document per (key line-item, year).

    A dense income-statement chunk holds ~15 line items, so its embedding
    is a blur that weakly matches any single-metric question. These atomic
    facts are query-shaped ("Apple - fiscal year 2024 - Operating income:
    123,216 ...") so they embed right next to the natural question. The
    fact's year is taken from the value's own column, so comparative
    figures (e.g. 2022 data inside a 2024 filing) become year-filterable.
    Only headline metrics are emitted (see _KEY_METRICS) to keep the
    corpus a size the vector store ingests reliably.
    """
    company = metadata.get("company", "Unknown")
    source_file = metadata.get("source_file", "")
    facts: list[Document] = []
    seen: set[tuple[str, int]] = set()

    for line in text.splitlines():
        m = _FACT_LINE.match(line.strip())
        if not m:
            continue
        label = m.group("label").strip()
        if not _is_key_metric(label.lower()):
            continue
        for part in _PAIR_SPLIT.split(m.group("body")):
            year_str, _, value = part.partition(" ")
            value = re.sub(r"\(\s+", "(", value).replace(" )", ")").strip(" ,")
            if not value or not year_str.isdigit():
                continue
            year = int(year_str)
            if (label, year) in seen:
                continue
            seen.add((label, year))
            facts.append(
                Document(
                    page_content=(
                        f"{company} - fiscal year {year} - "
                        f"{label}: {value} (in millions USD, "
                        f"from {source_file})."
                    ),
                    metadata={**metadata, "year": year},
                )
            )
    return facts


def chunk_documents(documents: list[dict]) -> list[Document]:
    """Split a list of parsed document dicts into LangChain Document chunks.

    Each chunk inherits the metadata (company, year, source_file) from its
    source. In addition to the prose/table chunks, atomic per-line-item
    fact chunks are emitted for headline financial metrics so specific
    figures are reliably retrievable for single-metric questions.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        chunks = splitter.create_documents(
            texts=[doc["text"]],
            metadatas=[doc["metadata"]],
        )
        facts = _atomic_facts(doc["text"], doc["metadata"])
        company = doc["metadata"].get("company", "Unknown")
        print(
            f"[chunker] {company} - {doc['metadata'].get('source_file')} -> "
            f"{len(chunks)} chunks + {len(facts)} atomic facts"
        )
        all_chunks.extend(chunks)
        all_chunks.extend(facts)

    print(f"[chunker] Done. {len(all_chunks)} total chunks.")
    return all_chunks
