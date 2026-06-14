"""
rag/ingest.py

Reads the wellness protocol PDF, chunks it, embeds with Gemini,
and persists to ChromaDB. Safe to re-run (checks if collection already populated).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

import rag._compat  # noqa: F401 — patches sqlite3 before chromadb loads
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

from rag.embeddings import GeminiEmbeddingFunction

load_dotenv()
logger = logging.getLogger(__name__)

COLLECTION_NAME = "wellness_protocol"
CHUNK_SIZE = 600       # characters per chunk
CHUNK_OVERLAP = 100    # character overlap between chunks


def _load_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_chroma_client() -> chromadb.ClientAPI:
    db_path = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def ingest_documents(force: bool = False) -> int:
    """
    Ingest wellness_protocol.pdf into ChromaDB.

    Args:
        force: If True, re-ingest even if the collection already has documents.

    Returns:
        Number of chunks stored.
    """
    pdf_path = os.environ.get("WELLNESS_PDF_PATH", "./data/wellness_protocol.pdf")

    if not Path(pdf_path).exists():
        logger.error("PDF not found at %s — run scripts/generate_pdf.py first", pdf_path)
        raise FileNotFoundError(f"Wellness protocol PDF not found: {pdf_path}")

    client = get_chroma_client()
    embed_fn = GeminiEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if not force and collection.count() > 0:
        logger.info("Collection already has %d chunks — skipping ingestion.", collection.count())
        return collection.count()

    if force:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    logger.info("Loading PDF from %s …", pdf_path)
    text = _load_pdf_text(pdf_path)
    chunks = _chunk_text(text)

    logger.info("Embedding and storing %d chunks …", len(chunks))
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    logger.info("Ingestion complete — %d chunks stored.", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = ingest_documents()
    print(f"Ingested {n} chunks into ChromaDB.")
