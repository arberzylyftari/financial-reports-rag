"""Split parsed documents into chunks with metadata using LangChain."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_documents(documents: list[dict]) -> list[Document]:
    """Split a list of parsed document dicts into LangChain Document chunks.

    Each chunk inherits the metadata (company, year, source_file) from its source.
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
        company = doc["metadata"].get("company", "Unknown")
        print(f"[chunker] {company} — {doc['metadata'].get('source_file')} → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"[chunker] Done. {len(all_chunks)} total chunks.")
    return all_chunks
