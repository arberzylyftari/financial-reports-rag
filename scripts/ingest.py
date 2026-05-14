"""End-to-end ingestion: parse HTML files → chunk → embed into ChromaDB."""

import sys
from pathlib import Path

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from ingestion.parser import parse_directory
from ingestion.chunker import chunk_documents
from vector_store.embedder import embed_and_store
from collections import Counter


def main():
    """Run the full ingestion pipeline."""
    data_dir = Path(__file__).resolve().parents[1] / "data"

    print("=" * 60)
    print("FINANCIAL REPORTS INGESTION PIPELINE")
    print("=" * 60)

    print("\n[1/3] Parsing HTML documents ...")
    documents = parse_directory(data_dir)
    if not documents:
        print("ERROR: No documents parsed. Check that HTML files exist in data/raw/apple/ and data/raw/tesla/.")
        sys.exit(1)

    print("\n[2/3] Chunking documents ...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Embedding and storing in ChromaDB ...")
    embed_and_store(chunks)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Total documents : {len(documents)}")
    print(f"  Total chunks    : {len(chunks)}")
    companies = Counter(doc["metadata"]["company"] for doc in documents)
    for company, count in companies.items():
        print(f"  {company:<10}: {count} document(s)")
    print("=" * 60)
    print("Ingestion complete. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
