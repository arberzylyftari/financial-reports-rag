"""Embed document chunks and persist them in ChromaDB."""

import time
from collections import Counter
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

COLLECTION_NAME = "financial_reports"
CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "chroma_db")
BATCH_SIZE = 50   # chunks per API call — keeps each batch well under 40k TPM
BATCH_PAUSE = 2   # seconds between batches


def get_embeddings() -> OpenAIEmbeddings:
    """Return the OpenAI embedding model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def embed_and_store(chunks: list[Document]) -> Chroma:
    """Embed chunks in small batches and store them in a persistent ChromaDB collection.

    Clears any existing collection before re-ingesting. Batching avoids
    hitting the 40k tokens-per-minute rate limit on new API accounts.
    """
    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[embedder] Cleared existing collection '{COLLECTION_NAME}'.")

    company_counts: Counter = Counter(
        chunk.metadata.get("company", "Unknown") for chunk in chunks
    )
    print(f"[embedder] Embedding {len(chunks)} chunks in batches of {BATCH_SIZE} ...")
    for company, count in company_counts.items():
        print(f"[embedder]   {company}: {count} chunks")

    vector_store: Chroma | None = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"[embedder] Batch {batch_num}/{total_batches} ({len(batch)} chunks) ...", flush=True)

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DIR,
            )
        else:
            vector_store.add_documents(batch)

        if i + BATCH_SIZE < len(chunks):
            time.sleep(BATCH_PAUSE)

    print(f"[embedder] Done. Collection '{COLLECTION_NAME}' persisted to {CHROMA_DIR}.")
    return vector_store


def load_vector_store() -> Chroma:
    """Load the existing persistent ChromaDB vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
