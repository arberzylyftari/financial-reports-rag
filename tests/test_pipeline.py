"""Basic tests for the ingestion and RAG pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
from ingestion.parser import _extract_year_from_filename, _clean_text
from ingestion.chunker import chunk_documents


def test_extract_year_from_filename():
    """Year extraction handles standard SEC filename formats."""
    assert _extract_year_from_filename("aapl-20240928.htm") == 2024
    assert _extract_year_from_filename("tsla-20231231.htm") == 2023
    assert _extract_year_from_filename("unknown.htm") is None


def test_clean_text():
    """Text cleaner collapses whitespace and removes blank lines."""
    raw = "  Hello  \n\n  World  \n\n"
    result = _clean_text(raw)
    assert result == "Hello\nWorld"


def test_chunk_documents_returns_documents():
    """Chunker returns LangChain Document objects with correct metadata."""
    sample = [
        {
            "text": "Apple reported revenue of $391 billion. " * 50,
            "metadata": {"company": "Apple", "year": 2024, "source_file": "aapl-2024.htm"},
        }
    ]
    chunks = chunk_documents(sample)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["company"] == "Apple"
        assert chunk.metadata["year"] == 2024


def test_chunk_metadata_propagated():
    """Every chunk carries the metadata of its source document."""
    sample = [
        {
            "text": "Tesla delivered 1.79 million vehicles. " * 50,
            "metadata": {"company": "Tesla", "year": 2024, "source_file": "tsla-2024.htm"},
        }
    ]
    chunks = chunk_documents(sample)
    for chunk in chunks:
        assert chunk.metadata["company"] == "Tesla"
        assert chunk.metadata["source_file"] == "tsla-2024.htm"
