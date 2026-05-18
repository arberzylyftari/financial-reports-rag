"""Embed document chunks and persist them in ChromaDB."""

import os
import time
from collections import Counter
from pathlib import Path

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

COLLECTION_NAME = "financial_reports"
CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "chroma_db")
BATCH_SIZE = 100  # chunks per API call — balanced for typical TPM limits
BATCH_PAUSE = 1   # seconds between batches — keeps under per-minute limits


def get_embeddings() -> OpenAIEmbeddings:
    """Return the OpenAI embedding model.

    An explicit request_timeout ensures a stalled API call fails fast and
    is retried rather than hanging the whole ingest indefinitely.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        request_timeout=60,
        max_retries=6,
    )


def _get_chroma_client():
    """Return a Chroma client.

    Uses Chroma Cloud when CHROMA_API_KEY is set (production / deployed),
    otherwise a local persistent client (local development).
    """
    api_key = os.getenv("CHROMA_API_KEY")
    if api_key:
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"],
        )
    return chromadb.PersistentClient(path=CHROMA_DIR)


def embed_and_store(chunks: list[Document]) -> Chroma:
    """Embed chunks in small batches and store them in a persistent ChromaDB collection.

    Clears any existing collection before re-ingesting. Batching avoids
    hitting the 40k tokens-per-minute rate limit on new API accounts.
    """
    embeddings = get_embeddings()

    client = _get_chroma_client()
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
                client=client,
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
        client=_get_chroma_client(),
    )
